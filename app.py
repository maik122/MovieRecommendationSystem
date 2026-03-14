"""
app.py — CineMatch · Movie Recommendation System
Run with:  streamlit run app.py
"""

import json
import os
import pickle

import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split as surprise_split

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch · Movie Recommender",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0C0C0C;
    color: #EDEBE3;
}

#MainMenu, footer, header { visibility: hidden; }

.hero { padding: 2.4rem 0 1.2rem; }
.hero-eyebrow {
    font-size: 0.7rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #F5C518;
    font-weight: 500;
    margin-bottom: 0.5rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 700;
    color: #F5C518;
    line-height: 1.05;
    margin-bottom: 0.4rem;
    letter-spacing: -0.5px;
}
.hero-sub {
    font-size: 0.95rem;
    color: #666;
    font-weight: 300;
    font-style: italic;
    margin-bottom: 0;
}

.rule { border: none; border-top: 1px solid #1E1E1E; margin: 1.6rem 0; }

.label {
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #F5C518;
    font-weight: 500;
    margin-bottom: 0.6rem;
}

.picks-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1rem; }
.pick-pill {
    background: #1A1A1A;
    border: 1px solid #F5C518;
    color: #F5C518;
    border-radius: 20px;
    padding: 0.3rem 0.9rem;
    font-size: 0.82rem;
    font-weight: 500;
}

.rec-list { display: flex; flex-direction: column; gap: 0.5rem; margin-top: 0.8rem; }
.rec-card {
    background: #141414;
    border: 1px solid #252525;
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    transition: border-color 0.2s, background 0.2s;
}
.rec-card:hover { border-color: #F5C518; background: #191919; }
.rec-rank {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    color: #F5C518;
    min-width: 1.8rem;
    text-align: center;
    flex-shrink: 0;
}
.rec-title { font-weight: 400; font-size: 0.95rem; color: #EDEBE3; flex: 1; }
.rec-badge {
    font-size: 0.72rem;
    color: #444;
    background: #1A1A1A;
    border: 1px solid #2A2A2A;
    border-radius: 4px;
    padding: 0.15rem 0.5rem;
    white-space: nowrap;
    flex-shrink: 0;
}

div[data-testid="stRadio"] > div { display: flex; gap: 0.5rem; flex-wrap: wrap; }
div[data-testid="stRadio"] label {
    background: #141414 !important;
    border: 1px solid #2A2A2A !important;
    border-radius: 6px !important;
    padding: 0.35rem 0.9rem !important;
    color: #888 !important;
    font-size: 0.85rem !important;
    cursor: pointer;
    transition: all 0.15s;
}
div[data-testid="stRadio"] label:has(input:checked) {
    background: #F5C518 !important;
    border-color: #F5C518 !important;
    color: #0C0C0C !important;
    font-weight: 500 !important;
}

.stMultiSelect [data-baseweb="tag"] {
    background-color: #F5C518 !important;
    color: #0C0C0C !important;
    font-weight: 500;
}
.stMultiSelect [data-baseweb="tag"] span { color: #0C0C0C !important; }

div.stButton > button {
    background: #F5C518;
    color: #0C0C0C;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    padding: 0.65rem 0;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    width: 100%;
    transition: opacity 0.2s;
}
div.stButton > button:hover { opacity: 0.84; background: #F5C518; border: none; }
div.stButton > button:disabled { opacity: 0.3; background: #F5C518; }

.info-strip {
    background: #111;
    border-left: 3px solid #F5C518;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    color: #666;
    margin: 0.5rem 0 1rem;
}

.metrics-row { display: flex; gap: 1rem; margin: 0.5rem 0 1.2rem; }
.metric-box {
    flex: 1;
    background: #141414;
    border: 1px solid #1E1E1E;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    text-align: center;
}
.metric-value {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    color: #F5C518;
    line-height: 1;
    margin-bottom: 0.25rem;
}
.metric-label { font-size: 0.72rem; color: #555; letter-spacing: 0.1em; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
# SVD is retrained at startup from the raw data file + saved
# JSON params. This avoids pickle's Cython binary incompatibility
# when the model is trained on a different Python/OS than Cloud.
# The other artifacts (pure pandas/numpy) are safe to pickle.

DATA_PATH  = "ml-100k"
MODELS_DIR = "models"


@st.cache_resource(show_spinner="Loading recommendation engine …")
def load_models():
    # ── Pandas / numpy artifacts ──────────────────────────────
    user_item_matrix  = pd.read_pickle(os.path.join(MODELS_DIR, "user_item_matrix.pkl"))
    item_similarity_df = pd.read_pickle(os.path.join(MODELS_DIR, "item_similarity.pkl"))
    movies_df         = pd.read_pickle(os.path.join(MODELS_DIR, "movies.pkl"))

    # ── SVD: retrain from data + saved best params ────────────
    with open(os.path.join(MODELS_DIR, "svd_params.json")) as f:
        params = json.load(f)

    reader       = Reader(line_format="user item rating timestamp", sep="\t")
    surprise_data = Dataset.load_from_file(
        os.path.join(DATA_PATH, "u.data"), reader=reader
    )
    trainset, _ = surprise_split(surprise_data, test_size=0.2, random_state=42)

    svd_model = SVD(**params)
    svd_model.fit(trainset)

    return user_item_matrix, item_similarity_df, movies_df, svd_model


try:
    user_item_matrix, item_similarity_df, movies_df, svd_model = load_models()
    movie_titles = sorted(movies_df["title"].tolist())
except FileNotFoundError as e:
    st.error(
        f"⚠️ Missing file: `{e.filename}`. "
        "Make sure `models/` and `ml-100k/` are both committed to your repo."
    )
    st.stop()


# ─────────────────────────────────────────────
# RECOMMENDATION FUNCTIONS
# ─────────────────────────────────────────────
def recommend_item_based(favorite_movies, item_similarity_df, top_n=10):
    scores = pd.Series(dtype=float)
    for movie in favorite_movies:
        if movie in item_similarity_df.columns:
            scores = scores.add(item_similarity_df[movie], fill_value=0)
    scores = scores.drop(favorite_movies, errors="ignore")
    top  = scores.sort_values(ascending=False).head(top_n)
    norm = top / top.max() if top.max() > 0 else top
    return list(zip(top.index.tolist(), norm.values.tolist()))


def recommend_user_based(favorite_movies, user_item_matrix, top_n=10):
    pseudo = pd.Series(0.0, index=user_item_matrix.columns)
    for movie in favorite_movies:
        if movie in pseudo.index:
            pseudo[movie] = 5.0
    sims = cosine_similarity([pseudo.values], user_item_matrix.values)[0]
    weighted = pd.Series(0.0, index=user_item_matrix.columns)
    for i, sim in enumerate(sims):
        weighted += user_item_matrix.iloc[i] * sim
    weighted /= sims.sum()
    weighted  = weighted.drop(favorite_movies, errors="ignore")
    top  = weighted.sort_values(ascending=False).head(top_n)
    norm = top / top.max() if top.max() > 0 else top
    return list(zip(top.index.tolist(), norm.values.tolist()))


def recommend_svd(favorite_movies, svd_model, movies_df, top_n=10):
    candidates = movies_df[~movies_df["title"].isin(favorite_movies)].copy()
    candidates["pred"] = candidates["movie_id"].apply(
        lambda x: svd_model.predict(0, x).est
    )
    top = candidates.sort_values("pred", ascending=False).head(top_n)
    lo, hi = top["pred"].min(), top["pred"].max()
    top = top.copy()
    top["norm"] = (top["pred"] - lo) / (hi - lo + 1e-9)
    return list(zip(top["title"].tolist(), top["norm"].tolist()))


# Internal mode keys → (display label, short metric label, plain-English explanation)
MODES = {
    "Similar Movies":   (
        "Similar movies",
        '"If you liked those, you\'ll like these" — finds movies that are '
        "rated similarly across all users. Great for staying in a genre or mood."
    ),
    "Similar Audience": (
        "Similar audience",
        '"People who loved your picks also loved…" — finds viewers with the '
        "same taste and borrows their highest-rated films."
    ),
    "Smart Pick":       (
        "Smart pick",
        '"Our best guess for you" — a machine-learning model trained on '
        "100,000 ratings predicts exactly how much you\'d enjoy each film."
    ),
}


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">MovieLens 100k · Collaborative Filtering</div>
  <div class="hero-title">CineMatch</div>
  <div class="hero-sub">Three films you love. Ten you haven't seen yet.</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="rule">', unsafe_allow_html=True)

st.markdown('<div class="label">Your three favourites</div>', unsafe_allow_html=True)
selected = st.multiselect(
    label="picks",
    options=movie_titles,
    max_selections=3,
    placeholder="Start typing a title …",
    label_visibility="collapsed",
)

remaining = 3 - len(selected)
if 0 < remaining < 3:
    st.markdown(
        f'<div class="info-strip">Pick {remaining} more movie{"s" if remaining > 1 else ""} to unlock recommendations.</div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="label" style="margin-top:1.2rem">How would you like us to choose?</div>', unsafe_allow_html=True)
mode = st.radio(
    label="mode",
    options=list(MODES.keys()),
    horizontal=True,
    label_visibility="collapsed",
)

# Plain-English explanation of the chosen mode
short_label, explanation = MODES[mode]
st.markdown(f'<div class="info-strip">💡 {explanation}</div>', unsafe_allow_html=True)

top_n = st.slider("How many recommendations?", min_value=5, max_value=20, value=10, step=1)
st.markdown("")
go = st.button("✦  Find My Movies", disabled=(len(selected) != 3))

if go and len(selected) == 3:
    with st.spinner("Curating your watchlist …"):
        if mode == "Similar Movies":
            results = recommend_item_based(selected, item_similarity_df, top_n)
        elif mode == "Similar Audience":
            results = recommend_user_based(selected, user_item_matrix, top_n)
        else:
            results = recommend_svd(selected, svd_model, movies_df, top_n)

    st.markdown('<hr class="rule">', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metrics-row">
      <div class="metric-box">
        <div class="metric-value">{len(results)}</div>
        <div class="metric-label">Recommendations</div>
      </div>
      <div class="metric-box">
        <div class="metric-value">{len(movie_titles):,}</div>
        <div class="metric-label">Movies indexed</div>
      </div>
      <div class="metric-box">
        <div class="metric-value">{short_label}</div>
        <div class="metric-label">Method used</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="label">Based on</div>', unsafe_allow_html=True)
    pills = "".join(f'<span class="pick-pill">🎬 {m}</span>' for m in selected)
    st.markdown(f'<div class="picks-row">{pills}</div>', unsafe_allow_html=True)

    st.markdown('<div class="label">Your watchlist</div>', unsafe_allow_html=True)
    cards = '<div class="rec-list">'
    for rank, (title, score) in enumerate(results, 1):
        pct = f"{score * 100:.0f}% match"
        cards += f"""
        <div class="rec-card">
          <div class="rec-rank">{rank}</div>
          <div class="rec-title">{title}</div>
          <div class="rec-badge">{pct}</div>
        </div>"""
    cards += "</div>"
    st.markdown(cards, unsafe_allow_html=True)

elif not go:
    st.markdown(
        '<div class="info-strip" style="margin-top:1.5rem">'
        '👆 Pick 3 movies above, choose an algorithm, then hit <strong>Find My Movies</strong>.'
        '</div>',
        unsafe_allow_html=True,
    )