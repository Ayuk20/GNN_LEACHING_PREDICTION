import os
import json
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from torch_geometric.data import HeteroData
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
# Cleaning / normalization
# ============================================================
def normalize_text(val):
    return str(val).strip().lower()


def normalize_solution(val):
    v = normalize_text(val)
    replacements = {
        "h₂so₄": "h2so4",
        "h₂so4": "h2so4",
        "h2so₄": "h2so4",
        "hno₃": "hno3",
        "h₃po₄": "h3po4",
        "h3po₄": "h3po4",
    }
    return replacements.get(v, v)


def normalize_ree_class(val):
    v = normalize_text(val)

    lree_elements = {"la", "ce", "pr", "nd", "pm", "sm", "eu"}
    hree_elements = {"gd", "tb", "dy", "ho", "er", "tm", "yb", "lu"}
    mixed_labels = {
        "mixed", "mix", "lree+hree", "lree/hree", "ree", "rees",
        "light+heavy", "light/heavy", "light and heavy", "both"
    }

    if v in {"lree", "light ree", "light rees", "light rare earth", "light rare earths"}:
        return "lree"
    if v in {"hree", "heavy ree", "heavy rees", "heavy rare earth", "heavy rare earths"}:
        return "hree"
    if v in mixed_labels:
        return "mixed"
    if v in lree_elements:
        return "lree"
    if v in hree_elements:
        return "hree"

    return v


def clean_dataframe(df):
    df = df.copy()
    df["material"] = df["material"].apply(normalize_text)
    df["solution"] = df["solution"].apply(normalize_solution)
    df["ree_class"] = df["ree_class"].apply(normalize_ree_class)

    numeric_cols = [
        "pretreatment", "si", "al", "fe", "temperature", "ph",
        "solid_liquid_ratio", "stirring_speed", "leaching_time", "recovery"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


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
# Full hetero graph builder for a single query
# ============================================================
class QueryFullGraphBuilder:
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

        mat_idx = self.category_maps["material"][row["material"]]
        sol_idx = self.category_maps["solution"][row["solution"]]
        ree_idx = self.category_maps["ree_class"][row["ree_class"]]

        data["material"].x = torch.tensor([mat_idx], dtype=torch.long)
        data["solution"].x = torch.tensor([sol_idx], dtype=torch.long)
        data["ree"].x = torch.tensor([ree_idx], dtype=torch.long)

        chem, op, sol_num = self.transform_numeric(row)
        data["chemistry"].x = torch.tensor([chem], dtype=torch.float)
        data["operation"].x = torch.tensor([op], dtype=torch.float)
        data["solution_num"].x = torch.tensor([sol_num], dtype=torch.float)
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

        return data


# ============================================================
# Full hetero-GAT model
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


class REEHeteroGAT(nn.Module):
    def __init__(self, n_materials, n_solutions, n_ree_classes, hidden_dim=64, heads=4, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dropout = dropout

        self.material_emb = nn.Embedding(n_materials, hidden_dim)
        self.solution_emb = nn.Embedding(n_solutions, hidden_dim)
        self.ree_emb = nn.Embedding(n_ree_classes, hidden_dim)

        self.chem_encoder = MLPEncoder(3, hidden_dim)
        self.op_encoder = MLPEncoder(5, hidden_dim)
        self.solnum_encoder = MLPEncoder(1, hidden_dim)
        self.exp_encoder = MLPEncoder(1, hidden_dim)

        out_per_head = hidden_dim // heads
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads.")

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
            aggr="sum"
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
            aggr="sum"
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
        x_dict = {k: F.dropout(F.elu(v), p=self.dropout, training=self.training) for k, v in x_dict.items()}

        x_dict = self.conv2(x_dict, data.edge_index_dict)
        x_dict = {k: F.elu(v) for k, v in x_dict.items()}

        out = self.regressor(x_dict["experiment"]).view(-1)
        return out

    @torch.no_grad()
    def get_experiment_embedding(self, data):
        self.eval()
        x_dict = self.encode_inputs(data)
        x_dict = self.conv1(x_dict, data.edge_index_dict)
        x_dict = {k: F.elu(v) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, data.edge_index_dict)
        x_dict = {k: F.elu(v) for k, v in x_dict.items()}
        return x_dict["experiment"].cpu().numpy()


# ============================================================
# Global KG-based recommender support
# ============================================================
class GlobalKGRecommender:
    def __init__(self, ref_df):
        self.ref_df = ref_df.copy()

        self.feature_cols = [
            "pretreatment", "si", "al", "fe", "temperature", "ph",
            "solid_liquid_ratio", "stirring_speed", "leaching_time"
        ]
        self.scaler = StandardScaler()
        self.X_num = self.scaler.fit_transform(self.ref_df[self.feature_cols].values.astype(float))

    def query_vector(self, row):
        vals = np.array([[
            row["pretreatment"], row["si"], row["al"], row["fe"],
            row["temperature"], row["ph"], row["solid_liquid_ratio"],
            row["stirring_speed"], row["leaching_time"]
        ]], dtype=float)
        return self.scaler.transform(vals)

    def retrieve_similar(self, row, k=8):
        q = self.query_vector(row)
        sims = cosine_similarity(q, self.X_num)[0]

        df = self.ref_df.copy()
        df["similarity"] = sims

        # soft categorical filtering
        material_mask = df["material"] == row["material"]
        ree_mask = df["ree_class"] == row["ree_class"]

        # favor matching material + ree
        df["similarity_adjusted"] = df["similarity"]
        df.loc[material_mask, "similarity_adjusted"] += 0.05
        df.loc[ree_mask, "similarity_adjusted"] += 0.03

        return df.sort_values("similarity_adjusted", ascending=False).head(k).copy()

    def better_neighbors(self, row, predicted_recovery, k=8, min_gain=3.0):
        nbrs = self.retrieve_similar(row, k=k)
        better = nbrs[nbrs["recovery"] >= predicted_recovery + min_gain].copy()

        if len(better) < 3:
            better = nbrs.sort_values("recovery", ascending=False).head(min(5, len(nbrs))).copy()

        return nbrs, better

    def recommend_changes(self, row, better_df, top_n=3):
        actionable_num = [
            "ph", "temperature", "solid_liquid_ratio",
            "stirring_speed", "leaching_time"
        ]
        actionable_cat = ["solution", "pretreatment"]

        recs = []

        for var in actionable_num:
            current = float(row[var])
            median_target = float(np.median(better_df[var].values))
            rel_gap = abs(median_target - current) / (abs(current) + 1e-6)

            if rel_gap < 0.15:
                continue

            direction = "increase" if median_target > current else "decrease"
            recs.append({
                "variable": var,
                "direction": direction,
                "current": current,
                "suggested_median": median_target,
                "score": rel_gap,
            })

        for var in actionable_cat:
            mode_vals = better_df[var].mode(dropna=True)
            if len(mode_vals) == 0:
                continue
            mode_val = mode_vals.iloc[0]
            if str(mode_val).strip().lower() != str(row[var]).strip().lower():
                recs.append({
                    "variable": var,
                    "direction": "switch",
                    "current": row[var],
                    "suggested_mode": mode_val,
                    "score": 1.0,
                })

        recs = sorted(recs, key=lambda x: x["score"], reverse=True)
        return recs[:top_n]


# ============================================================
# Main predictor-recommender
# ============================================================
class HeteroGATGlobalKGFramework:
    def __init__(
        self,
        data_csv="ree_leaching_gnn_ready.csv",
        model_path="ree_heterogat_full_model.pt",
        hidden_dim=64,
        heads=4,
        dropout=0.2,
        device=None,
    ):
        if not os.path.exists(data_csv):
            raise FileNotFoundError(f"Could not find {data_csv}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Could not find {model_path}")

        self.df = clean_dataframe(pd.read_csv(data_csv))

        required_cols = [
            "material", "solution", "ree_class", "pretreatment",
            "si", "al", "fe", "temperature", "ph",
            "solid_liquid_ratio", "stirring_speed", "leaching_time", "recovery"
        ]
        self.df = self.df.dropna(subset=required_cols).reset_index(drop=True)

        self.category_maps = build_category_maps(self.df)

        scale_cols = [
            "si", "al", "fe",
            "temperature", "solid_liquid_ratio", "stirring_speed",
            "leaching_time", "pretreatment", "ph"
        ]
        self.scaler = StandardScaler()
        self.scaler.fit(self.df[scale_cols].values.astype(float))

        self.graph_builder = QueryFullGraphBuilder(self.category_maps, self.scaler)

        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        self.model = REEHeteroGAT(
            n_materials=len(self.category_maps["material"]),
            n_solutions=len(self.category_maps["solution"]),
            n_ree_classes=len(self.category_maps["ree_class"]),
            hidden_dim=hidden_dim,
            heads=heads,
            dropout=dropout,
        ).to(self.device)

        state = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state)
        self.model.eval()

        self.recommender = GlobalKGRecommender(self.df)

    def normalize_query(self, user_input):
        row = dict(user_input)

        row["material"] = normalize_text(row["material"])
        row["solution"] = normalize_solution(row["solution"])
        row["ree_class"] = normalize_ree_class(row["ree_class"])

        numeric_cols = [
            "pretreatment", "si", "al", "fe", "temperature", "ph",
            "solid_liquid_ratio", "stirring_speed", "leaching_time"
        ]
        for c in numeric_cols:
            row[c] = float(row[c])

        # basic vocabulary check
        for c in ["material", "solution", "ree_class"]:
            if row[c] not in self.category_maps[c]:
                raise ValueError(
                    f"Unknown {c}: {row[c]}. Known values include: "
                    f"{list(self.category_maps[c].keys())[:10]}"
                )

        return row

    def predict_recovery(self, user_input):
        row = self.normalize_query(user_input)
        graph = self.graph_builder.build_graph(row).to(self.device)

        with torch.no_grad():
            pred = self.model(graph).item()

        return pred, row

    def recommend(self, user_input, k_neighbors=8, min_gain=3.0):
        pred, row = self.predict_recovery(user_input)
        neighbors, better = self.recommender.better_neighbors(
            row, pred, k=k_neighbors, min_gain=min_gain
        )
        recs = self.recommender.recommend_changes(row, better, top_n=3)

        result = {
            "predicted_recovery": pred,
            "similar_neighbors": neighbors[
                [
                    "material", "solution", "ree_class", "pretreatment",
                    "si", "al", "fe", "temperature", "ph",
                    "solid_liquid_ratio", "stirring_speed", "leaching_time",
                    "recovery", "similarity", "similarity_adjusted"
                ]
            ].to_dict(orient="records"),
            "better_neighbors": better[
                [
                    "material", "solution", "ree_class", "pretreatment",
                    "si", "al", "fe", "temperature", "ph",
                    "solid_liquid_ratio", "stirring_speed", "leaching_time",
                    "recovery", "similarity", "similarity_adjusted"
                ]
            ].to_dict(orient="records"),
            "recommendations": recs,
        }
        return result


# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":
    set_seed(42)

    framework = HeteroGATGlobalKGFramework(
        data_csv="ree_leaching_gnn_ready.csv",
        model_path="ree_heterogat_full_model.pt",
        hidden_dim=64,
        heads=4,
        dropout=0.2,
    )

    user_input = {
        "material": "iron residue",
        "solution": "H2SO4",
        "ree_class": "LREE",
        "pretreatment": 0,
        "si": 2.91,
        "al": 2.29,
        "fe": 59.55,
        "temperature": 180,
        "ph": 0.66,
        "solid_liquid_ratio": 0.05,
        "stirring_speed": 60,
        "leaching_time": 180,
    }

    result = framework.recommend(user_input, k_neighbors=8, min_gain=3.0)

    print(f"\nPredicted recovery: {result['predicted_recovery']:.3f}%")

    print("\nRecommendations:")
    if len(result["recommendations"]) == 0:
        print("No strong recommendation found.")
    else:
        for rec in result["recommendations"]:
            if rec["direction"] == "switch":
                print(f"- switch {rec['variable']} from {rec['current']} to {rec['suggested_mode']}")
            else:
                print(
                    f"- {rec['direction']} {rec['variable']} "
                    f"from {rec['current']:.3f} toward {rec['suggested_median']:.3f}"
                )

    print("\nTop similar experiments:")
    sim_df = pd.DataFrame(result["similar_neighbors"])
    print(sim_df[["material", "solution", "ree_class", "recovery", "similarity_adjusted"]].head(5))