import logging
import json
import torch
import random
import numpy as np

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

def get_alpha_weights(model: torch.nn.Module):
    """Extracts and formats all alpha_logits from OrientationGate modules."""
    alpha_weights = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Parameter):
             if 'alpha_logits' in name:
                  weights = torch.sigmoid(module.data).cpu().numpy()
                  alpha_weights[name] = [round(w, 4) for w in weights]
    return alpha_weights

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