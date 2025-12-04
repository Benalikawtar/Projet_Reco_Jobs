# ============================================================
#             JOBMATCH AI — VERSION FUTURISTE PREMIUM
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import io
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------
# STYLING — THEME FUTURISTE
# ------------------------------------------------------------
st.markdown("""
    <style>
        body {
            background-color: #0d0f17;
        }
        .main {
            background-color: #0d0f17;
            color: #ffffff;
        }
        .stTextInput>div>div>input {
            background: #11131c;
            color: white;
            border: 1px solid #444;
            border-radius: 8px;
        }
        .stButton>button {
            background: linear-gradient(90deg, #6a00ff, #9c4dff);
            color: white;
            border-radius: 8px;
            border: none;
            padding: 0.6rem 1rem;
            font-weight: bold;
        }
        .card {
            padding: 15px;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.05);
            margin-bottom: 15px;
            border-left: 3px solid #6a00ff;
        }
        .badge {
            background: #6a00ff;
            padding: 4px 10px;
            border-radius: 8px;
            margin-right: 5px;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="JobMatch AI", layout="wide")


# ------------------------------------------------------------
# DOWNLOAD FUNCTION
# ------------------------------------------------------------
@st.cache_resource
def download_file(url):
    response = requests.get(url)
    return io.BytesIO(response.content)


# ------------------------------------------------------------
# LOAD FILES FROM GOOGLE DRIVE
# ------------------------------------------------------------
URL_DF = "https://drive.google.com/uc?export=download&id=1wNihuzXv1xTX9MD_aNANIpnolvBtzwei"
URL_EMB = "https://drive.google.com/uc?export=download&id=1r0Z9gY5eocJ3C6HQ0yNE7F1qzBywPutp"
URL_KMEANS = "https://drive.google.com/uc?export=download&id=1p1XRqitrdF6NImC8jA3x7BfHICpWUoVb"
URL_TOPSKILLS = "https://drive.google.com/uc?export=download&id=1UZoSfnUa3LnHFfgm7sVIO-Y_pB3Bb1qt"
URL_UMAP = "https://drive.google.com/uc?export=download&id=1TKQtUaj85CxFnahpkR8R7HOs1AGgkMRV"

st.sidebar.info("Chargement des modèles...")

df = pd.read_parquet(download_file(URL_DF))
embeddings = np.load(download_file(URL_EMB))
kmeans = pickle.load(download_file(URL_KMEANS))
top_skills_par_profil = pickle.load(download_file(URL_TOPSKILLS))
umap_bytes = download_file(URL_UMAP)
umap_df = pd.read_parquet(umap_bytes)

st.sidebar.success("Modèles chargés ✔")


# ------------------------------------------------------------
# PROFIL NAMES
# ------------------------------------------------------------
PROFILE_NAMES = {
    0: "Data Engineer / Cloud Engineer",
    1: "Machine Learning Engineer / AI Engineer",
    2: "Business Analyst / Project Manager",
    3: "Machine Learning Engineer / AI Engineer",
    4: "Business Analyst / Project Manager",
    5: "Data Engineer / Cloud Engineer",
}

df["profil_metier"] = df["cluster_label"].map(PROFILE_NAMES)
umap_df["profil_metier"] = df["profil_metier"]


# ------------------------------------------------------------
# NAVIGATION MENU
# ------------------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Dashboard", "🔍 Recommandation d'offres", "🧠 Analyse d'un profil", "🌐 Visualisation IA des profils"]
)


# ============================================================
# PAGE 0 — DASHBOARD (HOME)
# ============================================================
if page == "🏠 Dashboard":

    st.markdown("<h1 style='color:#9c4dff;'>🌌 JobMatch AI</h1>", unsafe_allow_html=True)
    st.markdown("### Votre copilote intelligent pour explorer le marché de l'emploi Data & IA.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='card'><h3>🔍 Recommandation</h3>Découvrez les offres correspondant à votre profil.</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'><h3>👤 Analyse Profil</h3>Analysez vos compétences et votre positionnement.</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='card'><h3>🌐 Visualisation</h3>Explorez le paysage IA/Data en 2D.</div>", unsafe_allow_html=True)



# ============================================================
# PAGE 1 — RECOMMANDATION
# ============================================================
elif page == "🔍 Recommandation d'offres":

    st.header("🔍 Recommandation intelligente")

    user_text = st.text_area("Décrivez votre profil / compétences :")

    if st.button("🔎 Trouver des offres similaires"):
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        emb = model.encode([user_text])

        sims = cosine_similarity(emb, embeddings)[0]
        top_idx = sims.argsort()[-5:][::-1]

        for i in top_idx:
            st.markdown(f"""
                <div class='card'>
                    <h3 style='color:#9c4dff;'>{df.iloc[i]['title']}</h3>
                    <p>{df.iloc[i]['description'][:250]}...</p>
                    <p><b>Similarité :</b> {sims[i]:.2f}</p>
                </div>
            """, unsafe_allow_html=True)



# ============================================================
# PAGE 2 — ANALYSE PROFIL
# ============================================================
elif page == "🧠 Analyse d'un profil":

    st.header("🧠 Analyse automatisée d'un profil Data/IA")

    user_text = st.text_area("Collez votre CV ou décrivez votre profil :")

    if st.button("Analyser"):
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        emb = model.encode([user_text])

        cluster = kmeans.predict(emb)[0]
        profil = PROFILE_NAMES[cluster]

        st.success(f"🎯 Votre profil correspond à : **{profil}**")

        st.subheader("Compétences importantes pour ce profil :")

        for sk, score in top_skills_par_profil[cluster][:10]:
            st.markdown(f"<span class='badge'>{sk}</span>", unsafe_allow_html=True)



# ============================================================
# PAGE 3 — VISUALISATION UMAP
# ============================================================
elif page == "🌐 Visualisation IA des profils":

    st.header("🌐 Visualisation IA des Profils (UMAP)")

    fig = px.scatter(
        umap_df,
        x="x", y="y",
        color="profil_metier",
        template="plotly_dark",
        hover_data=["profil_metier"],
        title="🧭 Projection UMAP des Profils Métiers",
        color_discrete_sequence=px.colors.qualitative.Plotly
    )

    st.plotly_chart(fig, use_container_width=True)
