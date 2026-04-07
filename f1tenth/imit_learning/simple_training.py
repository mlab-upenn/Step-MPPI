import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
import torch.nn.functional as F

from NN import *

class ImitationDataset(Dataset):
    def __init__(self, mu_tilde, r_samples, mu_star, per_step_normalize=True):
        self.mu_tilde = mu_tilde
        self.r_samples = r_samples
        self.mu_star = mu_star
        self.per_step_normalize = per_step_normalize

    def __len__(self):
        return self.mu_tilde.shape[0]

    def __getitem__(self, idx):
        mu_tilde = torch.from_numpy(self.mu_tilde[idx]).float()  
        costs    = torch.from_numpy(self.r_samples[idx]).float() 
        mu_star  = torch.from_numpy(self.mu_star[idx]).float() 
        return mu_tilde, costs, mu_star
    
def load_dataset_npz(path="dataset.npz"):
    """
    Loads dataset saved by save_dataset_npz(...).
    Returns:
      mu_tilde:  (K, 20, 2) float32
      r_samples: (K, 256)   float32
      mu_star:   (K, 20, 2) float32
    """
    data = np.load(path)
    mu_tilde = data["mu_tilde"].astype(np.float32)
    r_samples = data["r_samples"].astype(np.float32)
    mu_star = data["mu_star"].astype(np.float32)

    print(f"Loaded {path}")
    return mu_tilde, r_samples, mu_star


def model_train(dataset, H, nu, M, training_params, hidden=1024,
                device=None, save_path="trained_NN.pt"):

    batch_size = int(training_params.get("batch_size", 256))
    epochs = int(training_params.get("epochs", 20))
    lr = float(training_params.get("lr", 1e-3))
    val_frac = float(training_params.get("val_frac", 0.1))
    print_after_epoch = int(training_params.get("print_after_epoch", 1))
    seed = int(training_params.get("seed", 12))

    # Plateau scheduler params, usually just use the default
    sched_factor = training_params.get("plateau_factor", 0.5)
    sched_patience = training_params.get("plateau_patience", 50)
    sched_threshold = training_params.get("plateau_threshold", 1e-6)
    sched_min_lr = training_params.get("plateau_min_lr", 1e-6)
    sched_cooldown = training_params.get("plateau_cooldown", 0)

    # Early stopping params, usually just use the default
    early_stop_patience = training_params.get("early_stop_patience", 20)  # epochs w/o improvement
    best_val_loss = float("inf")
    early_stop_counter = 0

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)

    n_total = len(dataset)
    n_val = int(n_total * val_frac)
    n_train = n_total - n_val
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device == "cuda"))
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=(device == "cuda"))

    model = L2O_Update_Net(H=H, nu=nu, M=M, hidden=hidden, dropout_p=0.1, per_step_cost_norm=True).to(device)

    opti = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opti, mode="min", factor=sched_factor,
        patience=sched_patience, threshold=sched_threshold, threshold_mode="rel",
        cooldown=sched_cooldown, min_lr=sched_min_lr,
    )

    def run_eval():
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for mu_t, c, mu_y in val_loader:
                mu_t = mu_t.to(device, non_blocking=True)
                c = c.to(device, non_blocking=True)
                mu_y = mu_y.to(device, non_blocking=True)

                pred = model(mu_t, c)
                loss = loss_fn(pred, mu_y)

                total += loss.item() * mu_t.shape[0]
                count += mu_t.shape[0]
        return total / max(count, 1)

    for ep in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0

        for mu_t, c, mu_y in train_loader:
            mu_t = mu_t.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)
            mu_y = mu_y.to(device, non_blocking=True)

            opti.zero_grad(set_to_none=True)
            pred = model(mu_t, c)
            loss = loss_fn(pred, mu_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opti.step()

            total += loss.item() * mu_t.shape[0]
            count += mu_t.shape[0]

        train_loss = total / max(count, 1)
        val_loss = run_eval()

        scheduler.step(val_loss)
        current_lr = opti.param_groups[0]["lr"]

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print(f"Early stopping triggered at epoch {ep}")
                break

        if (ep == 1) or (ep % print_after_epoch == 0) or (ep == epochs):
            print(f"epoch {ep:03d} | train {train_loss:.6e} | val {val_loss:.6e} | lr {current_lr:.2e}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "H": H, "nu": nu, "M": M, "hidden": hidden,
        },
        save_path,
    )
    print(f"Saved model to {save_path}")

    return model

def main():
    mu_tilde, r_samples, mu_star = load_dataset_npz("dataset.npz")
    dataset = ImitationDataset(mu_tilde, r_samples, mu_star, per_step_normalize=True)

    train_params = {
        "batch_size": 256,
        "epochs": 1000,
        "lr": 1e-4,
        "val_frac": 0.1,
        "print_after_epoch": 100,
        "seed": 0,
    }

    model = model_train(dataset, H = 40, nu = 2, M = 256, training_params = train_params)

if __name__ == "__main__":
    main()
