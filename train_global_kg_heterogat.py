import os
import json
import math
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from torch_geometric.nn import HeteroConv, GATConv


# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Metrics
# ============================================================
def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return {"r2": r2, "mae": mae, "rmse": rmse}


# ============================================================
# Model
# ============================================================
class GlobalKGHeteroGAT(nn.Module):
    """
    Attention-based heterogeneous GNN for node regression on experiment nodes.
    Works on the saved global KG built with build_global_ree_kg.py.
    """

    def __init__(self, metadata, hidden_dim=64, heads=4, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dropout = dropout

        node_types, edge_types = metadata

        # project each node type into shared latent space
        self.input_lin = nn.ModuleDict()
        for ntype in node_types:
            self.input_lin[ntype] = nn.LazyLinear(hidden_dim)

        out_per_head = hidden_dim // heads
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads.")

        self.conv1 = HeteroConv(
            {
                etype: GATConv(
                    (-1, -1),
                    out_per_head,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    add_self_loops=False,
                )
                for etype in edge_types
            },
            aggr="sum",
        )

        self.conv2 = HeteroConv(
            {
                etype: GATConv(
                    (-1, -1),
                    out_per_head,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    add_self_loops=False,
                )
                for etype in edge_types
            },
            aggr="sum",
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        # type-specific input projection
        x_dict = {
            ntype: self.input_lin[ntype](data[ntype].x)
            for ntype in data.node_types
        }

        # layer 1
        x_dict = self.conv1(x_dict, data.edge_index_dict)
        x_dict = {
            k: F.dropout(F.elu(v), p=self.dropout, training=self.training)
            for k, v in x_dict.items()
        }

        # layer 2
        x_dict = self.conv2(x_dict, data.edge_index_dict)
        x_dict = {
            k: F.elu(v)
            for k, v in x_dict.items()
        }

        pred = self.regressor(x_dict["experiment"]).view(-1)
        return pred, x_dict

    @torch.no_grad()
    def get_experiment_embeddings(self, data):
        self.eval()
        _, x_dict = self.forward(data)
        return x_dict["experiment"].cpu().numpy()


# ============================================================
# Train / eval
# ============================================================
def train_one_epoch(model, data, train_idx, optimizer, device):
    model.train()
    optimizer.zero_grad()

    pred, _ = model(data)
    y = data["experiment"].y.to(device)

    loss = F.mse_loss(pred[train_idx], y[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model, data, idx, device):
    model.eval()
    pred, x_dict = model(data)
    y = data["experiment"].y.to(device)

    y_true = y[idx].cpu().numpy()
    y_pred = pred[idx].cpu().numpy()
    metrics = regression_metrics(y_true, y_pred)

    exp_emb = x_dict["experiment"].cpu().numpy()
    return metrics, y_true, y_pred, exp_emb


def fit_model(
    model,
    data,
    train_idx,
    val_idx,
    device,
    lr=1e-3,
    weight_decay=1e-4,
    max_epochs=300,
    patience=30,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val_mae = float("inf")
    best_epoch = -1
    history = []

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(model, data, train_idx, optimizer, device)
        val_metrics, _, _, _ = evaluate(model, data, val_idx, device)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_r2": val_metrics["r2"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
        })

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) - best_epoch >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, pd.DataFrame(history), best_epoch, best_val_mae


# ============================================================
# Main
# ============================================================
def main():
    set_seed(42)

    graph_path = "ree_global_kg.pt"
    meta_path = "ree_global_kg_metadata.json"

    if not os.path.exists(graph_path):
        raise FileNotFoundError("Could not find ree_global_kg.pt. Run build_global_ree_kg.py first.")
    if not os.path.exists(meta_path):
        raise FileNotFoundError("Could not find ree_global_kg_metadata.json. Run build_global_ree_kg.py first.")

    # PyTorch 2.6+ safe load note
    data = torch.load(graph_path, map_location="cpu", weights_only=False)

    with open(meta_path, "r") as f:
        meta_json = json.load(f)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data = data.to(device)

    num_exp = data["experiment"].num_nodes
    all_idx = np.arange(num_exp)

    # 70/10/20 split
    trainval_idx, test_idx = train_test_split(all_idx, test_size=0.20, random_state=42)
    train_idx, val_idx = train_test_split(trainval_idx, test_size=0.125, random_state=42)

    train_idx = torch.tensor(train_idx, dtype=torch.long, device=device)
    val_idx = torch.tensor(val_idx, dtype=torch.long, device=device)
    test_idx = torch.tensor(test_idx, dtype=torch.long, device=device)

    model = GlobalKGHeteroGAT(
        metadata=data.metadata(),
        hidden_dim=64,
        heads=4,
        dropout=0.2,
    ).to(device)

    model, history_df, best_epoch, best_val_mae = fit_model(
        model=model,
        data=data,
        train_idx=train_idx,
        val_idx=val_idx,
        device=device,
        lr=1e-3,
        weight_decay=1e-4,
        max_epochs=300,
        patience=30,
    )

    print(f"Best epoch: {best_epoch}")
    print(f"Best val MAE: {best_val_mae:.4f}")

    val_metrics, y_val, y_val_pred, exp_emb = evaluate(model, data, val_idx, device)
    test_metrics, y_test, y_test_pred, exp_emb = evaluate(model, data, test_idx, device)

    print("\nValidation metrics:")
    for k, v in val_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nTest metrics:")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

    history_df.to_csv("global_kg_heterogat_training_history.csv", index=False)
    pd.DataFrame({"y_true": y_test, "y_pred": y_test_pred}).to_csv(
        "global_kg_heterogat_test_predictions.csv", index=False
    )
    np.save("global_kg_heterogat_experiment_embeddings.npy", exp_emb)
    torch.save(model.state_dict(), "global_kg_heterogat_model.pt")

    print("\nSaved:")
    print("- global_kg_heterogat_training_history.csv")
    print("- global_kg_heterogat_test_predictions.csv")
    print("- global_kg_heterogat_experiment_embeddings.npy")
    print("- global_kg_heterogat_model.pt")

    # --------------------------------------------------------
    # 5-fold CV
    # --------------------------------------------------------
    print("\nRunning 5-fold cross-validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    fold_results = []
    for fold, (tr_idx, te_idx) in enumerate(kf.split(all_idx), start=1):
        tr_idx = np.array(tr_idx)
        te_idx = np.array(te_idx)

        tr_sub_idx, va_idx = train_test_split(tr_idx, test_size=0.125, random_state=42)

        tr_sub_idx = torch.tensor(tr_sub_idx, dtype=torch.long, device=device)
        va_idx = torch.tensor(va_idx, dtype=torch.long, device=device)
        te_idx_t = torch.tensor(te_idx, dtype=torch.long, device=device)

        model = GlobalKGHeteroGAT(
            metadata=data.metadata(),
            hidden_dim=64,
            heads=4,
            dropout=0.2,
        ).to(device)

        model, _, _, _ = fit_model(
            model=model,
            data=data,
            train_idx=tr_sub_idx,
            val_idx=va_idx,
            device=device,
            lr=1e-3,
            weight_decay=1e-4,
            max_epochs=250,
            patience=25,
        )

        fold_metrics, _, _, _ = evaluate(model, data, te_idx_t, device)
        fold_metrics["fold"] = fold
        fold_results.append(fold_metrics)

        print(
            f"Fold {fold}: "
            f"R2={fold_metrics['r2']:.4f}, "
            f"MAE={fold_metrics['mae']:.4f}, "
            f"RMSE={fold_metrics['rmse']:.4f}"
        )

    cv_df = pd.DataFrame(fold_results)
    cv_df.to_csv("global_kg_heterogat_cv_results.csv", index=False)

    print("\nCross-validation summary:")
    print(cv_df.describe().loc[["mean", "std"]])

    print("\nSaved:")
    print("- global_kg_heterogat_cv_results.csv")


if __name__ == "__main__":
    main()