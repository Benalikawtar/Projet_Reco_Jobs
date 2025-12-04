import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import io
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="JobMatch AI", layout="wide")

# -------------------------
# 1) DOWNLOAD FUNCTION
# -------------------------
@st.cache_resource
def download_file(url):
    response = requests.get(url)
    return io.BytesIO(response.content)


# -------------------------
# 2) Load files from Google Drive
# -------------------------
URL_DF = "https://drive.google.com/uc?export=download&id=1wNihuzXv1xTX9MD_aNANIpnolvBtzwei"
URL_EMB = "https://drive.google.com/uc?export=download&id=1r0Z9gY5eocJ3C6HQ0yNE7F1qzBywPutp"
URL_KMEANS = "https://drive.google.com/uc?export=download&id=1p1XRqitrdF6NImC8jA3x7BfHICpWUoVb"
URL_TOPSKILLS = "https://drive.google.com/uc?export=download&id=1UZoSfnUa3LnHFfgm7sVIO-Y_pB3Bb1qt"

st.sidebar.success("Chargement des modèles en cours...")

df_bytes = download_file(URL_DF)
emb_bytes = download_file(URL_EMB)
kmeans_bytes = download_file(URL_KMEANS)
topskills_bytes = download_file(URL_TOPSKILLS)

df = pd.read_parquet(df_bytes)
embeddings = np.load(emb_bytes)
kmeans = pickle.load(kmeans_bytes)
top_skills_par_profil = pickle.load(topskills_bytes)

st.sidebar.success("Modèles chargés ✔")

# -------------------------
# 3) Mapping profil métier
# -------------------------
PROFILE_NAMES = {
    0: "Data Engineer / Cloud Engineer",
    1: "Machine Learning Engineer / AI Engineer",
    2: "Business Analyst / Project Manager",
    3: "Machine Learning Engineer / AI Engineer",
    4: "Business Analyst / Project Manager",
    5: "Data Engineer / Cloud Engineer",
}

df["profil_metier"] = df["cluster_label"].map(PROFILE_NAMES)

# -------------------------
# 4) Page Navigation
# -------------------------
st.title("💼 JobMatch AI — Votre assistant carrière intelligent")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏷️ Explorer par profil métier",
        "🔍 Recommandation d'offres",
        "🧠 Analyse d'un profil"
    ]
)

# -------------------------
# PAGE 1 — Explorer un profil métier
# -------------------------
if page == "🏷️ Explorer par profil métier":
    st.header("🏷️ Explorer les offres par profil métier")

    profil = st.selectbox("Choisissez un profil :", df["profil_metier"].unique())

    subset = df[df["profil_metier"] == profil]

    st.subheader(f"📌 Top compétences pour ce profil :")
    for sk, score in top_skills_par_profil[df[df['profil_metier']==profil]['cluster_label'].iloc[0]]:
        st.write(f"• {sk} — **{score}**")

    st.subheader(f"📄 Exemples d'offres ({len(subset)})")
    for idx, row in subset.head(10).iterrows():
        st.write(f"### 🔹 {row['title']}")
        st.write(row["description"][:300] + "...")


# -------------------------
# PAGE 2 — Recommandation
# -------------------------
elif page == "🔍 Recommandation d'offres":
    st.header("🔍 Recommandation d'offres d'emploi")

    user_text = st.text_area("Décrivez votre profil / compétences :")

    if st.button("Trouver des offres similaires"):
        if len(user_text.strip()) < 5:
            st.error("Veuillez entrer une description.")
        else:
            # encoding SBERT already computed → use embeddings directly
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            user_emb = model.encode([user_text])

            sims = cosine_similarity(user_emb, embeddings)[0]
            top_idx = sims.argsort()[-5:][::-1]

            st.subheader("🎯 Offres recommandées :")
            for i in top_idx:
                st.write(f"### 🔹 {df.iloc[i]['title']}")
                st.write(df.iloc[i]["description"][:300] + "...")
                st.write("---")


# -------------------------
# PAGE 3 — Analyse
# -------------------------
elif page == "🧠 Analyse d'un profil":
    st.header("🧠 Analyse automatique d'un profil métier")

    user_text = st.text_area("Entrez votre CV / description :")

    if st.button("Analyser"):
        if len(user_text) < 5:
            st.error("Entrez un texte valide.")
        else:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            user_emb = model.encode([user_text])

            cluster = kmeans.predict(user_emb)[0]
            profil = PROFILE_NAMES[cluster]

            st.success(f"🎯 Vous correspondez au profil : **{profil}**")
            st.subheader("Compétences importantes pour ce profil :")

            for sk, score in top_skills_par_profil[cluster][:10]:
                st.write(f"• {sk} — **{score}**")
