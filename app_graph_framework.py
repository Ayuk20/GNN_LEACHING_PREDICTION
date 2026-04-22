import pandas as pd
import streamlit as st

from full_heterogat_plus_globalkg_attention_recommender import (
    FullHeteroGATPlusGlobalKGAttentionFramework,
)

st.set_page_config(
    page_title="REE Leaching Graph Predictor–Recommender",
    layout="wide"
)

st.title("REE Leaching Graph Predictor–Recommender")
st.write(
    "Prediction uses the full heterogeneous attention graph. "
    "Recommendation uses global knowledge-graph attention embeddings."
)

# ------------------------------------------------------------
# Cached framework loader
# ------------------------------------------------------------
@st.cache_resource
def load_framework():
    return FullHeteroGATPlusGlobalKGAttentionFramework(
        data_csv="ree_leaching_gnn_ready.csv",
        full_model_path="ree_heterogat_full_model.pt",
        global_kg_graph_path="ree_global_kg.pt",
        global_kg_model_path="global_kg_heterogat_model.pt",
        hidden_dim=64,
        heads=4,
        dropout=0.2,
    )

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:
    st.header("Model configuration")
    k_neighbors = st.slider(
        "Number of similar experiments",
        min_value=3,
        max_value=15,
        value=8
    )
    min_gain = st.slider(
        "Minimum recovery gain threshold",
        min_value=0.0,
        max_value=15.0,
        value=3.0,
        step=0.5
    )

    st.markdown("---")
    st.write("Required files in the same folder:")
    st.code(
        "\n".join([
            "ree_leaching_gnn_ready.csv",
            "ree_heterogat_full_model.pt",
            "ree_global_kg.pt",
            "global_kg_heterogat_model.pt",
            "full_heterogat_plus_globalkg_attention_recommender.py",
        ]),
        language="text"
    )

# ------------------------------------------------------------
# Load framework
# ------------------------------------------------------------
framework = None
load_error = None
try:
    framework = load_framework()
except Exception as e:
    load_error = str(e)

if load_error is not None:
    st.error("Failed to load the graph framework.")
    st.code(load_error)
    st.stop()

# ------------------------------------------------------------
# Build choices from framework data
# ------------------------------------------------------------
materials = sorted(framework.category_maps["material"].keys())
solutions = sorted(framework.category_maps["solution"].keys())
ree_classes = sorted(framework.category_maps["ree_class"].keys())

# ------------------------------------------------------------
# Input form
# ------------------------------------------------------------
st.subheader("Input conditions")

with st.form("query_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        material = st.selectbox(
            "Material",
            options=materials,
            index=materials.index("iron residue") if "iron residue" in materials else 0
        )
        solution = st.selectbox(
            "Solution",
            options=solutions,
            index=solutions.index("h2so4") if "h2so4" in solutions else 0
        )
        ree_class = st.selectbox(
            "REE class",
            options=ree_classes,
            index=ree_classes.index("lree") if "lree" in ree_classes else 0
        )
        pretreatment = st.selectbox("Pretreatment", options=[0, 1], index=0)

    with c2:
        si = st.number_input("Si (%)", value=2.91, format="%.4f")
        al = st.number_input("Al (%)", value=2.29, format="%.4f")
        fe = st.number_input("Fe (%)", value=59.55, format="%.4f")
        temperature = st.number_input("Temperature (°C)", value=180.0, format="%.4f")

    with c3:
        ph = st.number_input("pH", value=0.66, format="%.4f")
        solid_liquid_ratio = st.number_input("Solid-liquid ratio", value=0.05, format="%.6f")
        stirring_speed = st.number_input("Stirring speed (rpm)", value=60.0, format="%.4f")
        leaching_time = st.number_input("Leaching time (min)", value=180.0, format="%.4f")

    submitted = st.form_submit_button("Predict and Recommend", type="primary")

# ------------------------------------------------------------
# Run inference
# ------------------------------------------------------------
if submitted:
    user_input = {
        "material": material,
        "solution": solution,
        "ree_class": ree_class,
        "pretreatment": pretreatment,
        "si": si,
        "al": al,
        "fe": fe,
        "temperature": temperature,
        "ph": ph,
        "solid_liquid_ratio": solid_liquid_ratio,
        "stirring_speed": stirring_speed,
        "leaching_time": leaching_time,
    }

    try:
        with st.spinner("Running full hetero-GAT prediction and KG-embedding recommendation..."):
            result = framework.recommend(
                user_input=user_input,
                k_neighbors=k_neighbors,
                min_gain=min_gain
            )
    except Exception as e:
        st.error("Inference failed.")
        st.code(str(e))
        st.stop()

    # --------------------------------------------------------
    # Top metrics
    # --------------------------------------------------------
    pred = result["predicted_recovery"]

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Predicted recovery (%)", f"{pred:.2f}")
    with m2:
        better_df = pd.DataFrame(result["better_neighbors"])
        if not better_df.empty:
            st.metric("Mean better-neighbor recovery (%)", f"{better_df['recovery'].mean():.2f}")
        else:
            st.metric("Mean better-neighbor recovery (%)", "N/A")

    # --------------------------------------------------------
    # Recommendations
    # --------------------------------------------------------
    st.subheader("Recommended changes")
    if len(result["recommendations"]) == 0:
        st.info("No strong recommendation found from better-performing similar experiments.")
    else:
        for rec in result["recommendations"]:
            if rec["direction"] == "switch":
                st.write(
                    f"- **Switch {rec['variable']}** from `{rec['current']}` "
                    f"to `{rec['suggested_mode']}`"
                )
            else:
                st.write(
                    f"- **{rec['direction'].capitalize()} {rec['variable']}** "
                    f"from `{rec['current']:.3f}` toward `{rec['suggested_median']:.3f}`"
                )

    # --------------------------------------------------------
    # Similar experiments
    # --------------------------------------------------------
    st.subheader("Top similar experiments from global KG embeddings")
    sim_df = pd.DataFrame(result["similar_neighbors"])
    if not sim_df.empty:
        sim_show = [
            "material", "solution", "ree_class", "pretreatment",
            "ph", "temperature", "solid_liquid_ratio",
            "stirring_speed", "leaching_time",
            "recovery", "kg_similarity_adj"
        ]
        sim_show = [c for c in sim_show if c in sim_df.columns]
        st.dataframe(sim_df[sim_show], use_container_width=True)

    # --------------------------------------------------------
    # Better-performing neighbors
    # --------------------------------------------------------
    st.subheader("Better-performing neighbors used for recommendation")
    better_df = pd.DataFrame(result["better_neighbors"])
    if not better_df.empty:
        better_show = [
            "material", "solution", "ree_class", "pretreatment",
            "ph", "temperature", "solid_liquid_ratio",
            "stirring_speed", "leaching_time",
            "recovery", "kg_similarity_adj"
        ]
        better_show = [c for c in better_show if c in better_df.columns]
        st.dataframe(better_df[better_show], use_container_width=True)

    # --------------------------------------------------------
    # Quick comparison panel
    # --------------------------------------------------------
    st.subheader("Query summary")
    qdf = pd.DataFrame([user_input])
    st.dataframe(qdf, use_container_width=True)