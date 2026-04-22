import os
import json
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
from torch_geometric.data import HeteroData


# ============================================================
# Helpers
# ============================================================
def clean_str(x):
    return str(x).strip().lower()


def normalize_solution(val):
    v = clean_str(val)
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
    v = clean_str(val)

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


def interval_label(val, bins, prefix):
    for low, high in bins:
        low_ok = val > low if np.isfinite(low) else True
        high_ok = val <= high if np.isfinite(high) else True
        if low_ok and high_ok:
            low_txt = str(low).replace(".0", "") if np.isfinite(low) else "neg_inf"
            high_txt = str(high).replace(".0", "") if np.isfinite(high) else "pos_inf"
            return f"{prefix}_{low_txt}_to_{high_txt}"
    return f"{prefix}_unknown"


def make_quantile_bins(series, q=4):
    s = pd.to_numeric(series, errors="coerce").dropna().values
    if len(s) == 0:
        return [(-np.inf, np.inf)]

    qs = np.unique(np.quantile(s, np.linspace(0, 1, q + 1)))
    if len(qs) < 2:
        return [(-np.inf, np.inf)]

    bins = []
    bins.append((-np.inf, qs[1]))
    for i in range(1, len(qs) - 2):
        bins.append((qs[i], qs[i + 1]))
    bins.append((qs[-2], np.inf))
    return bins


def add_reverse_edges(data):
    existing = list(data.edge_types)
    for src, rel, dst in existing:
        rev_rel = f"rev_{rel}"
        if (dst, rev_rel, src) not in data.edge_types:
            edge_index = data[(src, rel, dst)].edge_index
            rev_edge = torch.stack([edge_index[1], edge_index[0]], dim=0)
            data[(dst, rev_rel, src)].edge_index = rev_edge


# ============================================================
# Main builder
# ============================================================
def build_global_kg(
    csv_path="ree_leaching_gnn_ready.csv",
    save_pt="ree_global_kg.pt",
    save_meta="ree_global_kg_metadata.json",
):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = [
        "material", "solution", "ree_class", "pretreatment",
        "si", "al", "fe", "temperature", "ph",
        "solid_liquid_ratio", "stirring_speed", "leaching_time", "recovery"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # clean categorical columns
    df["material"] = df["material"].astype(str).map(clean_str)
    df["solution"] = df["solution"].astype(str).map(normalize_solution)
    df["ree_class"] = df["ree_class"].astype(str).map(normalize_ree_class)

    # numeric coercion
    num_cols = [
        "pretreatment", "si", "al", "fe", "temperature", "ph",
        "solid_liquid_ratio", "stirring_speed", "leaching_time", "recovery"
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=required_cols).reset_index(drop=True)
    print(f"Usable rows: {len(df)}")

    # --------------------------------------------------------
    # Bin definitions
    # --------------------------------------------------------
    ph_bins = [
        (-np.inf, 0.0),
        (0.0, 0.5),
        (0.5, 1.0),
        (1.0, 2.0),
        (2.0, np.inf),
    ]
    temp_bins = [
        (-np.inf, 50.0),
        (50.0, 100.0),
        (100.0, 180.0),
        (180.0, 400.0),
        (400.0, 900.0),
        (900.0, np.inf),
    ]
    stir_bins = [
        (-np.inf, 100.0),
        (100.0, 300.0),
        (300.0, 500.0),
        (500.0, np.inf),
    ]
    time_bins = [
        (-np.inf, 30.0),
        (30.0, 60.0),
        (60.0, 120.0),
        (120.0, 240.0),
        (240.0, np.inf),
    ]

    slr_bins = make_quantile_bins(df["solid_liquid_ratio"], q=4)
    si_bins = make_quantile_bins(df["si"], q=4)
    al_bins = make_quantile_bins(df["al"], q=4)
    fe_bins = make_quantile_bins(df["fe"], q=4)

    # --------------------------------------------------------
    # Bin node labels
    # --------------------------------------------------------
    df["ph_bin_node"] = df["ph"].apply(lambda x: interval_label(x, ph_bins, "ph"))
    df["temp_bin_node"] = df["temperature"].apply(lambda x: interval_label(x, temp_bins, "temp"))
    df["stir_bin_node"] = df["stirring_speed"].apply(lambda x: interval_label(x, stir_bins, "stir"))
    df["time_bin_node"] = df["leaching_time"].apply(lambda x: interval_label(x, time_bins, "time"))
    df["slr_bin_node"] = df["solid_liquid_ratio"].apply(lambda x: interval_label(x, slr_bins, "slr"))
    df["si_bin_node"] = df["si"].apply(lambda x: interval_label(x, si_bins, "si"))
    df["al_bin_node"] = df["al"].apply(lambda x: interval_label(x, al_bins, "al"))
    df["fe_bin_node"] = df["fe"].apply(lambda x: interval_label(x, fe_bins, "fe"))

    # --------------------------------------------------------
    # Node maps
    # --------------------------------------------------------
    exp_ids = [f"exp_{i}" for i in range(len(df))]
    material_nodes = sorted(df["material"].unique().tolist())
    solution_nodes = sorted(df["solution"].unique().tolist())
    ree_nodes = sorted(df["ree_class"].unique().tolist())
    pretreat_nodes = ["pretreatment_0", "pretreatment_1"]

    ph_bin_nodes = sorted(df["ph_bin_node"].unique().tolist())
    temp_bin_nodes = sorted(df["temp_bin_node"].unique().tolist())
    stir_bin_nodes = sorted(df["stir_bin_node"].unique().tolist())
    time_bin_nodes = sorted(df["time_bin_node"].unique().tolist())
    slr_bin_nodes = sorted(df["slr_bin_node"].unique().tolist())
    si_bin_nodes = sorted(df["si_bin_node"].unique().tolist())
    al_bin_nodes = sorted(df["al_bin_node"].unique().tolist())
    fe_bin_nodes = sorted(df["fe_bin_node"].unique().tolist())

    node_maps = {
        "experiment": {k: i for i, k in enumerate(exp_ids)},
        "material": {k: i for i, k in enumerate(material_nodes)},
        "solution": {k: i for i, k in enumerate(solution_nodes)},
        "ree_class": {k: i for i, k in enumerate(ree_nodes)},
        "pretreatment": {k: i for i, k in enumerate(pretreat_nodes)},
        "ph_bin": {k: i for i, k in enumerate(ph_bin_nodes)},
        "temp_bin": {k: i for i, k in enumerate(temp_bin_nodes)},
        "stir_bin": {k: i for i, k in enumerate(stir_bin_nodes)},
        "time_bin": {k: i for i, k in enumerate(time_bin_nodes)},
        "slr_bin": {k: i for i, k in enumerate(slr_bin_nodes)},
        "si_bin": {k: i for i, k in enumerate(si_bin_nodes)},
        "al_bin": {k: i for i, k in enumerate(al_bin_nodes)},
        "fe_bin": {k: i for i, k in enumerate(fe_bin_nodes)},
    }

    # --------------------------------------------------------
    # Experiment node features
    # --------------------------------------------------------
    exp_feature_cols = [
        "pretreatment", "si", "al", "fe", "temperature", "ph",
        "solid_liquid_ratio", "stirring_speed", "leaching_time"
    ]
    scaler = StandardScaler()
    exp_x = scaler.fit_transform(df[exp_feature_cols].values.astype(float))
    y = df["recovery"].values.astype(float)

    # --------------------------------------------------------
    # Build graph
    # --------------------------------------------------------
    data = HeteroData()

    data["experiment"].x = torch.tensor(exp_x, dtype=torch.float)
    data["experiment"].y = torch.tensor(y, dtype=torch.float)

    # one-hot identity features for shared attribute nodes
    for ntype, mapping in node_maps.items():
        if ntype == "experiment":
            continue
        n = len(mapping)
        data[ntype].x = torch.eye(n, dtype=torch.float)

    edge_store = {
        ("experiment", "has_material", "material"): [],
        ("experiment", "has_solution", "solution"): [],
        ("experiment", "has_ree_class", "ree_class"): [],
        ("experiment", "has_pretreatment", "pretreatment"): [],
        ("experiment", "has_ph_bin", "ph_bin"): [],
        ("experiment", "has_temp_bin", "temp_bin"): [],
        ("experiment", "has_stir_bin", "stir_bin"): [],
        ("experiment", "has_time_bin", "time_bin"): [],
        ("experiment", "has_slr_bin", "slr_bin"): [],
        ("experiment", "has_si_bin", "si_bin"): [],
        ("experiment", "has_al_bin", "al_bin"): [],
        ("experiment", "has_fe_bin", "fe_bin"): [],
    }

    for i, row in df.iterrows():
        exp_idx = node_maps["experiment"][f"exp_{i}"]

        edge_store[("experiment", "has_material", "material")].append(
            [exp_idx, node_maps["material"][row["material"]]]
        )
        edge_store[("experiment", "has_solution", "solution")].append(
            [exp_idx, node_maps["solution"][row["solution"]]]
        )
        edge_store[("experiment", "has_ree_class", "ree_class")].append(
            [exp_idx, node_maps["ree_class"][row["ree_class"]]]
        )
        edge_store[("experiment", "has_pretreatment", "pretreatment")].append(
            [exp_idx, node_maps["pretreatment"][f"pretreatment_{int(row['pretreatment'])}"]]
        )
        edge_store[("experiment", "has_ph_bin", "ph_bin")].append(
            [exp_idx, node_maps["ph_bin"][row["ph_bin_node"]]]
        )
        edge_store[("experiment", "has_temp_bin", "temp_bin")].append(
            [exp_idx, node_maps["temp_bin"][row["temp_bin_node"]]]
        )
        edge_store[("experiment", "has_stir_bin", "stir_bin")].append(
            [exp_idx, node_maps["stir_bin"][row["stir_bin_node"]]]
        )
        edge_store[("experiment", "has_time_bin", "time_bin")].append(
            [exp_idx, node_maps["time_bin"][row["time_bin_node"]]]
        )
        edge_store[("experiment", "has_slr_bin", "slr_bin")].append(
            [exp_idx, node_maps["slr_bin"][row["slr_bin_node"]]]
        )
        edge_store[("experiment", "has_si_bin", "si_bin")].append(
            [exp_idx, node_maps["si_bin"][row["si_bin_node"]]]
        )
        edge_store[("experiment", "has_al_bin", "al_bin")].append(
            [exp_idx, node_maps["al_bin"][row["al_bin_node"]]]
        )
        edge_store[("experiment", "has_fe_bin", "fe_bin")].append(
            [exp_idx, node_maps["fe_bin"][row["fe_bin_node"]]]
        )

    for etype, pairs in edge_store.items():
        edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
        data[etype].edge_index = edge_index

    add_reverse_edges(data)

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------
    metadata = {
        "num_rows": len(df),
        "node_counts": {ntype: int(data[ntype].num_nodes) for ntype in data.node_types},
        "edge_counts": {
            str(etype): int(data[etype].edge_index.shape[1]) for etype in data.edge_types
        },
        "experiment_feature_cols": exp_feature_cols,
        "binning": {
            "ph_bins": ph_bins,
            "temp_bins": temp_bins,
            "stir_bins": stir_bins,
            "time_bins": time_bins,
            "slr_bins": slr_bins,
            "si_bins": si_bins,
            "al_bins": al_bins,
            "fe_bins": fe_bins,
        },
        "node_maps": node_maps,
    }

    torch.save(data, save_pt)
    with open(save_meta, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved graph to: {save_pt}")
    print(f"Saved metadata to: {save_meta}")
    print("\nNode types:", data.node_types)
    print("Edge types:", data.edge_types)
    print("\nNode counts:")
    for ntype in data.node_types:
        print(f"  {ntype}: {data[ntype].num_nodes}")

    return data, metadata


if __name__ == "__main__":
    build_global_kg()