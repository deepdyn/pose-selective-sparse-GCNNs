import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split, ConcatDataset
import numpy as np

# Special transform for Rotated MNIST/FashionMNIST
class RandomRotation:
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, img):
        angle = np.random.uniform(-self.degrees, self.degrees)
        return transforms.functional.rotate(img, angle)

def get_dataloaders_with_fixed_splits(name: str, batch_size: int, path: str = "./data"):
    """
    Creates train, validation, and test data loaders with specific, fixed splits
    as defined by the experimental protocol.
    """
    is_rotated = 'Rot' in name
    use_flips = '+' in name

    # --- Define Transformations ---
    # Define a generic ToTensor transform for non-augmented sets
    to_tensor_transform = transforms.ToTensor()
    
    # Define training transforms based on dataset type
    if "CIFAR" in name or "GTSRB" in name:
        # Normalization for 3-channel images
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        flip_prob = 0.5 if use_flips else 0.0
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=flip_prob),
            transforms.ToTensor(),
            normalize
        ])
        test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    else: # MNIST / FashionMNIST
        train_transform = transforms.Compose([
            RandomRotation(180) if is_rotated else transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ToTensor()
        ])
        test_transform = to_tensor_transform

    # --- Define Splits and Load Datasets ---
    if "Fashion" in name or "CIFAR" in name or "GTSRB" in name:
        # These datasets use a standard Train/Val split from the official training set
        if "Fashion" in name:
            DatasetClass = datasets.FashionMNIST
            splits = {'train': 55000, 'val': 5000}
        elif "CIFAR" in name:
            DatasetClass = datasets.CIFAR10
            splits = {'train': 40000, 'val': 10000}
        elif "GTSRB" in name:
            DatasetClass = datasets.GTSRB
            # Use split='train' for the combined train/val pool
            full_train_dataset = DatasetClass(root=path, split='train', download=True, transform=train_transform)
            test_dataset = DatasetClass(root=path, split='test', download=True, transform=test_transform)
            splits = {'train': 21312, 'val': 5328}

        # For GTSRB, the datasets are already loaded. For others, load them now.
        if "GTSRB" not in name:
            full_train_dataset = DatasetClass(root=path, train=True, download=True, transform=train_transform)
            test_dataset = DatasetClass(root=path, train=False, download=True, transform=test_transform)
            # Important: Apply non-augmenting transform to the validation set
            val_dataset_full = DatasetClass(root=path, train=True, download=True, transform=test_transform)
            full_train_dataset.transform = train_transform # Ensure train has augmentations
        else: # For GTSRB, we create a second copy for validation transforms
            val_dataset_full = datasets.GTSRB(root=path, split='train', download=True, transform=test_transform)
        
        # Split the training data into a new train and a validation set
        # Use a fixed generator for reproducibility
        generator = torch.Generator().manual_seed(42)
        train_subset, val_subset, _ = random_split(full_train_dataset, [splits['train'], splits['val'], len(full_train_dataset) - sum(splits.values())], generator=generator)
        
        # We need to get the indices from the split and apply them to the non-augmented dataset for validation
        val_indices = val_subset.indices
        val_dataset = Subset(val_dataset_full, val_indices)
        train_dataset = train_subset

    elif "MNIST" in name:
        # MNIST uses a non-standard split by merging train and test sets
        DatasetClass = datasets.MNIST
        splits = {'train': 12000, 'val': 2000, 'test': 56000}
        
        # Load and combine the official train and test sets
        d1 = DatasetClass(root=path, train=True, download=True, transform=train_transform)
        d2 = DatasetClass(root=path, train=False, download=True, transform=train_transform)
        combined_dataset = ConcatDataset([d1, d2])
        
        # Create a second copy for validation/testing with no augmentations
        d1_noaug = DatasetClass(root=path, train=True, transform=test_transform)
        d2_noaug = DatasetClass(root=path, train=False, transform=test_transform)
        combined_dataset_noaug = ConcatDataset([d1_noaug, d2_noaug])
        
        # Split the combined dataset according to the specified sizes
        generator = torch.Generator().manual_seed(42)
        train_subset, val_subset, test_subset = random_split(combined_dataset, [splits['train'], splits['val'], splits['test']], generator=generator)

        # Get indices and create subsets with appropriate transforms
        train_dataset = train_subset
        val_dataset = Subset(combined_dataset_noaug, val_subset.indices)
        test_dataset = Subset(combined_dataset_noaug, test_subset.indices)

    else:
        raise ValueError(f"Dataset {name} not supported.")

    # --- Create DataLoaders ---
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Loaded {name}: {len(train_dataset)} train, {len(val_dataset)} validation, {len(test_dataset)} test images.")
    return train_loader, val_loader, test_loader