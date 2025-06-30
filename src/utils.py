import logging
import json
import torch
import random
import numpy as np
import torch.nn.functional as F
from functools import partial

def set_seed(seed: int):
    """Sets the random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info(f"Random seed set to {seed}")

def setup_logging(log_file: str):
    """Configures logging to both file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def save_results(results: dict, save_path: str):
    """Saves final results dictionary to a JSON file."""
    file_path = f"{save_path}/results.json"
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4)
    logging.info(f"Final results saved to {file_path}")

def get_gate_weights(model: torch.nn.Module) -> dict:
    """
    Extracts and formats the current gate values (π_g) from 
    DifferentiableMaskGate modules.
    """
    from src.models.pose_gcnn import DifferentiableMaskGate # Local import to avoid circular dependency
    
    gate_weights = {}
    for name, module in model.named_modules():
        if isinstance(module, DifferentiableMaskGate):
            # Get the deterministic mask value for logging
            with torch.no_grad():
                module.eval() # Use deterministic forward pass
                weights = module.get_mask().cpu().numpy()
                module.train() # Set back to train mode
            gate_weights[name] = [round(w, 4) for w in weights]
    return gate_weights

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def expected_calibration_error(conf, correct, n_bins=15):
    bins = torch.linspace(0, 1, n_bins + 1)
    ece  = torch.zeros(1, device=conf.device)
    for i in range(n_bins):
        mask   = (conf > bins[i]) & (conf <= bins[i + 1])
        if mask.any():
            acc   = correct[mask].float().mean()
            bin_conf = conf[mask].mean()
            ece  += mask.float().mean() * (acc - bin_conf).abs()
    return ece.item()

@torch.no_grad()
def compute_ece(model, test_loader, device="cuda", n_bins=15):
    model.eval()
    all_conf, all_correct = [], []

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)                       # (batch, C)
        prob   = F.softmax(logits, dim=1)       # convert to probabilities
        conf, pred = prob.max(dim=1)            # highest prob per sample
        all_conf.append(conf)
        all_correct.append(pred.eq(y))          # Boolean tensor

    conf_tensor    = torch.cat(all_conf)        # shape (N,)
    correct_tensor = torch.cat(all_correct)     # shape (N,)

    return expected_calibration_error(conf_tensor,
                                      correct_tensor,
                                      n_bins=n_bins)


def calculate_gflops(model, input_shape, device):
    """
    Calculates the GFLOPs of a sparse model by temporarily hardening the gates
    to reflect the pruned state at test time.
    """
    try:
        from thop import profile
        from src.models.pose_gcnn import DifferentiableMaskGate
    except ImportError:
        logging.warning("`thop` is not installed. Skipping FLOPs calculation. Please run `pip install thop`.")
        return -1

    dummy_input = torch.randn(1, *input_shape).to(device)
    
    original_get_mask_methods = {}

    # Define the hard mask function that mimics the zero-temperature limit
    def hard_get_mask(self):
        with torch.no_grad():
            soft_mask = torch.sigmoid(self.b_logits / self.temp)
            # Convert to a hard 0/1 mask
            return (soft_mask > 0.5).float()

    # Temporarily replace the get_mask method in all gate modules
    for name, module in model.named_modules():
        if isinstance(module, DifferentiableMaskGate):
            original_get_mask_methods[name] = module.get_mask
            module.get_mask = partial(hard_get_mask, module)

    try:
        model.eval()
        total_ops, _ = profile(model, inputs=(dummy_input,), verbose=False)
        gflops = total_ops / 1e9
    except Exception as e:
        logging.error(f"An error occurred during FLOPs calculation: {e}")
        gflops = -1
    finally:
        # IMPORTANT: Restore the original methods to not affect subsequent evaluations
        for name, module in model.named_modules():
            if name in original_get_mask_methods:
                module.get_mask = original_get_mask_methods[name]
    
    return gflops
