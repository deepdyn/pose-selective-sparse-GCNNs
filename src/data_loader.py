import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split, ConcatDataset

# Define normalization constants for clarity
NORM_STATS = {
    'MNIST': {'mean': (0.1307,), 'std': (0.3081,)},
    'FashionMNIST': {'mean': (0.2860,), 'std': (0.3530,)},
    'CIFAR10': {'mean': (0.4914, 0.4822, 0.4465), 'std': (0.2023, 0.1994, 0.2010)},
    'GTSRB': {'mean': (0.3582, 0.3628, 0.3567), 'std': (0.2075, 0.2134, 0.2099)}
}

def get_dataloaders_with_fixed_splits(name: str, batch_size: int, path: str = "./data"):
    """
    Creates train, validation, and test data loaders with specific, fixed splits
    and data augmentation pipelines as defined by the experimental protocol.
    """
    is_rotated = 'Rot' in name
    use_flips = '+' in name

    # --- Define Transformations based on dataset name ---
    if "MNIST" in name:
        stats_key = "FashionMNIST" if "Fashion" in name else "MNIST"
        stats = NORM_STATS[stats_key]
        
        train_transforms_list = []
        if is_rotated:
            train_transforms_list.append(transforms.RandomRotation(degrees=(-180, 180)))
        
        train_transforms_list.extend([transforms.ToTensor(), transforms.Normalize(stats['mean'], stats['std'])])
        train_transform = transforms.Compose(train_transforms_list)
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(stats['mean'], stats['std'])])

    elif "CIFAR10" in name:
        stats = NORM_STATS['CIFAR10']
        flip_prob = 0.5 if use_flips else 0.0
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=flip_prob),
            transforms.ToTensor(),
            transforms.Normalize(stats['mean'], stats['std'])
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(stats['mean'], stats['std'])
        ])

    elif "GTSRB" in name:
        stats = NORM_STATS['GTSRB']
        flip_prob = 0.5 if use_flips else 0.0
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=flip_prob),
            transforms.RandomRotation(degrees=(-180, 180)),
            transforms.ToTensor(),
            transforms.Normalize(stats['mean'], stats['std'])
        ])
        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomRotation(degrees=(-180, 180)),
            transforms.ToTensor(),
            transforms.Normalize(stats['mean'], stats['std'])
        ])
    else:
        raise ValueError(f"Dataset name '{name}' not recognized for transformation pipeline.")


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
            full_train_dataset = DatasetClass(root=path, split='train', download=True, transform=train_transform)
            test_dataset = DatasetClass(root=path, split='test', download=True, transform=test_transform)
            splits = {'train': 21312, 'val': 5328}
        
        if "GTSRB" not in name:
            full_train_dataset = DatasetClass(root=path, train=True, download=True, transform=train_transform)
            test_dataset = DatasetClass(root=path, train=False, download=True, transform=test_transform)
        
        # Create a second copy of the training data with test-time transforms for the validation set
        val_dataset_full = DatasetClass(root=path, train=True, download=True, transform=test_transform) if "GTSRB" not in name else datasets.GTSRB(root=path, split='train', download=True, transform=test_transform)
        
        # Split the training data into a new train and a validation set
        generator = torch.Generator().manual_seed(42)
        train_subset, val_subset, _ = random_split(full_train_dataset, [splits['train'], splits['val'], len(full_train_dataset) - sum(splits.values())], generator=generator)
        
        # Create subsets with the correct transforms
        train_dataset = train_subset
        val_dataset = Subset(val_dataset_full, val_subset.indices)

    elif "MNIST" in name:
        # MNIST uses a non-standard split by merging train and test sets
        DatasetClass = datasets.MNIST
        splits = {'train': 12000, 'val': 2000, 'test': 56000}
        
        # Load and combine the official train and test sets
        d1_train = DatasetClass(root=path, train=True, download=True, transform=train_transform)
        d2_train = DatasetClass(root=path, train=False, download=True, transform=train_transform)
        combined_dataset_train = ConcatDataset([d1_train, d2_train])
        
        # Create a second copy for validation/testing with no training augmentations
        d1_test = DatasetClass(root=path, train=True, download=True, transform=test_transform)
        d2_test = DatasetClass(root=path, train=False, download=True, transform=test_transform)
        combined_dataset_test = ConcatDataset([d1_test, d2_test])
        
        # Split the combined dataset according to the specified sizes
        generator = torch.Generator().manual_seed(42)
        # Note: The split is on indices, so we apply it once
        train_subset_indices, val_subset_indices, test_subset_indices = random_split(range(len(combined_dataset_train)), [splits['train'], splits['val'], splits['test']], generator=generator)

        # Create subsets with the appropriate transforms
        train_dataset = Subset(combined_dataset_train, train_subset_indices.indices)
        val_dataset = Subset(combined_dataset_test, val_subset_indices.indices)
        test_dataset = Subset(combined_dataset_test, test_subset_indices.indices)
        
    else:
        raise ValueError(f"Dataset {name} not supported for splitting.")

    # --- Create DataLoaders ---
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Loaded {name}: {len(train_dataset)} train, {len(val_dataset)} validation, {len(test_dataset)} test images.")
    return train_loader, val_loader, test_loader