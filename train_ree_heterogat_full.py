import os
import math
import copy
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import HeteroConv, GATConv


# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed: int = 42):
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
# Category maps
# ============================================================
def build_category_maps(df):
    maps = {}
    for col in ["material", "solution", "ree_class"]:
        cats = sorted(df[col].astype(str).unique().tolist())
        maps[col] = {c: i for i, c in enumerate(cats)}
    return maps


# ============================================================
# Graph builder: full heterogeneous graph
# ============================================================
class REELeachingFullHeteroGraphBuilder:
    """
    One graph per experiment.

    Node types:
      - material         : categorical embedding id
      - solution         : categorical embedding id
      - ree              : categorical embedding id
      - chemistry        : [si, al, fe]
      - operation        : [temperature, solid_liquid_ratio, stirring_speed, leaching_time, pretreatment]
      - solution_num     : [ph]
      - experiment       : dummy [1.0]

    Full graph edges:
      Core:
        material   -> experiment
        solution   -> experiment
        ree        -> experiment
        chemistry  -> experiment
        operation  -> experiment
        solution_num -> experiment

      Reverse:
        experiment -> material / solution / ree / chemistry / operation / solution_num

      Mechanistic:
        material   -> chemistry
        solution   -> chemistry
        solution_num -> operation
        operation  -> chemistry
        material   -> ree

      Reverse mechanistic:
        chemistry  -> material
        chemistry  -> solution
        operation  -> solution_num
        chemistry  -> operation
        ree        -> material
    """

    def __init__(self, category_maps, scaler):
        self.category_maps = category_maps
        self.scaler = scaler

    def transform_numeric(self, row):
        vals = np.array([[
            row["si"],
            row["al"],
            row["fe"],
            row["temperature"],
            row["solid_liquid_ratio"],
            row["stirring_speed"],
            row["leaching_time"],
            row["pretreatment"],
            row["ph"],
        ]], dtype=float)

        vals_scaled = self.scaler.transform(vals)[0]
        chem = vals_scaled[0:3]
        op = vals_scaled[3:8]
        sol_num = vals_scaled[8:9]
        return chem, op, sol_num

    def build_graph(self, row):
        data = HeteroData()

        # categorical node ids
        mat_idx = self.category_maps["material"][str(row["material"])]
        sol_idx = self.category_maps["solution"][str(row["solution"])]
        ree_idx = self.category_maps["ree_class"][str(row["ree_class"])]

        data["material"].x = torch.tensor([mat_idx], dtype=torch.long)
        data["solution"].x = torch.tensor([sol_idx], dtype=torch.long)
        data["ree"].x = torch.tensor([ree_idx], dtype=torch.long)

        # numeric node features
        chem, op, sol_num = self.transform_numeric(row)
        data["chemistry"].x = torch.tensor([chem], dtype=torch.float)
        data["operation"].x = torch.tensor([op], dtype=torch.float)
        data["solution_num"].x = torch.tensor([sol_num], dtype=torch.float)

        # experiment node: dummy constant feature
        data["experiment"].x = torch.tensor([[1.0]], dtype=torch.float)

        edge = torch.tensor([[0], [0]], dtype=torch.long)

        # core edges
        data["material", "to", "experiment"].edge_index = edge
        data["solution", "to", "experiment"].edge_index = edge
        data["ree", "to", "experiment"].edge_index = edge
        data["chemistry", "to", "experiment"].edge_index = edge
        data["operation", "to", "experiment"].edge_index = edge
        data["solution_num", "to", "experiment"].edge_index = edge

        # reverse edges
        data["experiment", "rev_to", "material"].edge_index = edge
        data["experiment", "rev_to", "solution"].edge_index = edge
        data["experiment", "rev_to", "ree"].edge_index = edge
        data["experiment", "rev_to", "chemistry"].edge_index = edge
        data["experiment", "rev_to", "operation"].edge_index = edge
        data["experiment", "rev_to", "solution_num"].edge_index = edge

        # mechanistic edges
        data["material", "interacts", "chemistry"].edge_index = edge
        data["solution", "acts_on", "chemistry"].edge_index = edge
        data["solution_num", "controls", "operation"].edge_index = edge
        data["operation", "affects", "chemistry"].edge_index = edge
        data["material", "hosts", "ree"].edge_index = edge

        # reverse mechanistic edges
        data["chemistry", "rev_interacts", "material"].edge_index = edge
        data["chemistry", "rev_acts_on", "solution"].edge_index = edge
        data["operation", "rev_controls", "solution_num"].edge_index = edge
        data["chemistry", "rev_affects", "operation"].edge_index = edge
        data["ree", "rev_hosts", "material"].edge_index = edge

        # target
        data.y = torch.tensor([float(row["recovery"])], dtype=torch.float)

        return data


def build_graph_list(df, builder):
    return [builder.build_graph(row) for _, row in df.iterrows()]


# ============================================================
# Encoders
# ============================================================
class MLPEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Full heterogeneous attention graph model
# ============================================================
class REEHeteroGAT(nn.Module):
    def __init__(
        self,
        n_materials: int,
        n_solutions: int,
        n_ree_classes: int,
        hidden_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dropout = dropout

        # embeddings for categorical nodes
        self.material_emb = nn.Embedding(n_materials, hidden_dim)
        self.solution_emb = nn.Embedding(n_solutions, hidden_dim)
        self.ree_emb = nn.Embedding(n_ree_classes, hidden_dim)

        # encoders for numeric nodes
        self.chem_encoder = MLPEncoder(3, hidden_dim)
        self.op_encoder = MLPEncoder(5, hidden_dim)
        self.solnum_encoder = MLPEncoder(1, hidden_dim)
        self.exp_encoder = MLPEncoder(1, hidden_dim)

        # GAT output per head -> hidden_dim // heads
        # concat=True means final size = out_channels * heads = hidden_dim
        out_per_head = hidden_dim // heads
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads for this implementation.")

        edge_types = [
            ("material", "to", "experiment"),
            ("solution", "to", "experiment"),
            ("ree", "to", "experiment"),
            ("chemistry", "to", "experiment"),
            ("operation", "to", "experiment"),
            ("solution_num", "to", "experiment"),

            ("experiment", "rev_to", "material"),
            ("experiment", "rev_to", "solution"),
            ("experiment", "rev_to", "ree"),
            ("experiment", "rev_to", "chemistry"),
            ("experiment", "rev_to", "operation"),
            ("experiment", "rev_to", "solution_num"),

            ("material", "interacts", "chemistry"),
            ("solution", "acts_on", "chemistry"),
            ("solution_num", "controls", "operation"),
            ("operation", "affects", "chemistry"),
            ("material", "hosts", "ree"),

            ("chemistry", "rev_interacts", "material"),
            ("chemistry", "rev_acts_on", "solution"),
            ("operation", "rev_controls", "solution_num"),
            ("chemistry", "rev_affects", "operation"),
            ("ree", "rev_hosts", "material"),
        ]

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

    def encode_inputs(self, data):
        x_dict = {}

        x_dict["material"] = self.material_emb(data["material"].x.view(-1))
        x_dict["solution"] = self.solution_emb(data["solution"].x.view(-1))
        x_dict["ree"] = self.ree_emb(data["ree"].x.view(-1))

        x_dict["chemistry"] = self.chem_encoder(data["chemistry"].x)
        x_dict["operation"] = self.op_encoder(data["operation"].x)
        x_dict["solution_num"] = self.solnum_encoder(data["solution_num"].x)
        x_dict["experiment"] = self.exp_encoder(data["experiment"].x)

        return x_dict

    def forward(self, data):
        x_dict = self.encode_inputs(data)

        x_dict = self.conv1(x_dict, data.edge_index_dict)
        x_dict = {
            k: F.dropout(F.elu(v), p=self.dropout, training=self.training)
            for k, v in x_dict.items()
        }

        x_dict = self.conv2(x_dict, data.edge_index_dict)
        x_dict = {k: F.elu(v) for k, v in x_dict.items()}

        out = self.regressor(x_dict["experiment"])
        return out.view(-1)


# ============================================================
# Train / eval
# ============================================================
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_n = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(batch)
        target = batch.y.view(-1)

        loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * target.size(0)
        total_n += target.size(0)

    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds = []
    trues = []

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        target = batch.y.view(-1)

        preds.extend(pred.cpu().numpy().tolist())
        trues.extend(target.cpu().numpy().tolist())

    metrics = regression_metrics(trues, preds)
    return metrics, np.array(trues), np.array(preds)


def fit_model(
    model,
    train_loader,
    val_loader,
    device,
    lr=1e-3,
    weight_decay=1e-4,
    max_epochs=250,
    patience=30,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val_mae = float("inf")
    best_epoch = -1
    history = []

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics, _, _ = evaluate(model, val_loader, device)

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
# Data prep
# ============================================================
def prepare_graph_splits(df_train, df_val, df_test):
    category_maps = build_category_maps(
        pd.concat([df_train, df_val, df_test], axis=0, ignore_index=True)
    )

    scale_cols = [
        "si", "al", "fe",
        "temperature", "solid_liquid_ratio", "stirring_speed", "leaching_time",
        "pretreatment", "ph",
    ]
    scaler = StandardScaler()
    scaler.fit(df_train[scale_cols].values.astype(float))

    builder = REELeachingFullHeteroGraphBuilder(category_maps, scaler)

    train_graphs = build_graph_list(df_train, builder)
    val_graphs = build_graph_list(df_val, builder)
    test_graphs = build_graph_list(df_test, builder)

    return train_graphs, val_graphs, test_graphs, category_maps, scaler


# ============================================================
# Main
# ============================================================
def main():
    set_seed(42)

    csv_path = "ree_leaching_gnn_ready.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = [
        "material", "solution", "ree_class", "pretreatment",
        "si", "al", "fe", "temperature", "ph",
        "solid_liquid_ratio", "stirring_speed", "leaching_time", "recovery",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # basic cleaning
    df = df.copy()
    for c in ["material", "solution", "ree_class"]:
        df[c] = df[c].astype(str).str.strip().str.lower()

    numeric_cols = [
        "pretreatment", "si", "al", "fe", "temperature", "ph",
        "solid_liquid_ratio", "stirring_speed", "leaching_time", "recovery",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=required_cols).reset_index(drop=True)

    print(f"Usable rows: {len(df)}")

    # 70/10/20 split
    df_trainval, df_test = train_test_split(df, test_size=0.20, random_state=42)
    df_train, df_val = train_test_split(df_trainval, test_size=0.125, random_state=42)

    train_graphs, val_graphs, test_graphs, maps, scaler = prepare_graph_splits(
        df_train, df_val, df_test
    )

    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=64, shuffle=False)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    model = REEHeteroGAT(
        n_materials=len(maps["material"]),
        n_solutions=len(maps["solution"]),
        n_ree_classes=len(maps["ree_class"]),
        hidden_dim=64,
        heads=4,
        dropout=0.2,
    ).to(device)

    model, history_df, best_epoch, best_val_mae = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=1e-3,
        weight_decay=1e-4,
        max_epochs=250,
        patience=30,
    )

    print(f"Best epoch: {best_epoch}")
    print(f"Best val MAE: {best_val_mae:.4f}")

    val_metrics, _, _ = evaluate(model, val_loader, device)
    test_metrics, y_test, y_pred = evaluate(model, test_loader, device)

    print("\nValidation metrics:")
    for k, v in val_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nTest metrics:")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

    history_df.to_csv("heterogat_full_training_history.csv", index=False)
    pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(
        "heterogat_full_test_predictions.csv", index=False
    )
    torch.save(model.state_dict(), "ree_heterogat_full_model.pt")

    print("\nSaved:")
    print("- heterogat_full_training_history.csv")
    print("- heterogat_full_test_predictions.csv")
    print("- ree_heterogat_full_model.pt")

    # --------------------------------------------------------
    # 5-fold CV
    # --------------------------------------------------------
    print("\nRunning 5-fold cross-validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(df), start=1):
        df_fold_train = df.iloc[train_idx].reset_index(drop=True)
        df_fold_test = df.iloc[test_idx].reset_index(drop=True)

        df_fold_train_sub, df_fold_val = train_test_split(
            df_fold_train, test_size=0.125, random_state=42
        )

        train_graphs, val_graphs, test_graphs, maps, scaler = prepare_graph_splits(
            df_fold_train_sub, df_fold_val, df_fold_test
        )

        train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_graphs, batch_size=64, shuffle=False)

        model = REEHeteroGAT(
            n_materials=len(maps["material"]),
            n_solutions=len(maps["solution"]),
            n_ree_classes=len(maps["ree_class"]),
            hidden_dim=64,
            heads=4,
            dropout=0.2,
        ).to(device)

        model, _, _, _ = fit_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            lr=1e-3,
            weight_decay=1e-4,
            max_epochs=200,
            patience=25,
        )

        fold_metrics, _, _ = evaluate(model, test_loader, device)
        fold_metrics["fold"] = fold
        fold_results.append(fold_metrics)

        print(
            f"Fold {fold}: "
            f"R2={fold_metrics['r2']:.4f}, "
            f"MAE={fold_metrics['mae']:.4f}, "
            f"RMSE={fold_metrics['rmse']:.4f}"
        )

    fold_df = pd.DataFrame(fold_results)
    fold_df.to_csv("heterogat_full_cv_results.csv", index=False)

    print("\nCross-validation summary:")
    print(fold_df.describe().loc[["mean", "std"]])

    print("\nSaved:")
    print("- heterogat_full_cv_results.csv")


if __name__ == "__main__":
    main()