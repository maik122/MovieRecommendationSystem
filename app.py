# app.py
import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Load models (cached for speed)
# -------------------------------
@st.cache_resource
def load_models():
    user_item_matrix = pd.read_pickle("models/user_item_matrix.pkl")
    item_similarity_df = pd.read_pickle("models/item_similarity.pkl")
    movies_df = pd.read_pickle("models/movies.pkl")
    with open("models/svd_model.pkl", "rb") as f:
        svd_model = pickle.load(f)
    return user_item_matrix, item_similarity_df, movies_df, svd_model

user_item_matrix, item_similarity_df, movies_df, svd_model = load_models()
movie_titles = movies_df['title'].tolist()

# -------------------------------
# Recommendation functions
# -------------------------------
def recommend_from_favorites(favorite_movies, item_similarity_df, top_n=10):
    scores = pd.Series(dtype=float)
    for movie in favorite_movies:
        if movie in item_similarity_df.columns:
            scores = scores.add(item_similarity_df[movie], fill_value=0)
    scores = scores.drop(favorite_movies, errors="ignore")
    return scores.sort_values(ascending=False).head(top_n).index.tolist()

def recommend_user_based(favorite_movies, user_item_matrix, top_n=10):
    pseudo_user = pd.Series(0, index=user_item_matrix.columns)
    for movie in favorite_movies:
        if movie in pseudo_user.index:
            pseudo_user[movie] = 5
    similarities = cosine_similarity([pseudo_user.values], user_item_matrix.values)[0]
    weighted_ratings = pd.Series(0, index=user_item_matrix.columns)
    for i, sim in enumerate(similarities):
        weighted_ratings += user_item_matrix.iloc[i] * sim
    weighted_ratings /= similarities.sum()
    weighted_ratings = weighted_ratings.drop(favorite_movies, errors="ignore")
    return weighted_ratings.sort_values(ascending=False).head(top_n).index.tolist()

def recommend_svd(favorite_movies, svd_model, movies_df, user_item_matrix, top_n=10):
    pseudo_user_id = 0
    user_unrated = movies_df[~movies_df["title"].isin(favorite_movies)].copy()
    user_unrated["pred_rating"] = user_unrated["movie_id"].apply(
        lambda x: svd_model.predict(pseudo_user_id, x).est
    )
    return user_unrated.sort_values("pred_rating", ascending=False).head(top_n)["title"].tolist()

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Recommendation System")
st.markdown("""
Welcome! Select **3 of your favourite movies** below, choose a recommendation mode, 
and click **Recommend** to get your personalised top 10 movie suggestions.
""")

# -------------------------------
# Sidebar: Movie selection & mode
# -------------------------------
st.sidebar.header("Your Preferences")
favorite_movies = st.sidebar.multiselect(
    "Select exactly 3 favourite movies",
    options=movie_titles,
    default=movie_titles[:3]
)
if len(favorite_movies) != 3:
    st.sidebar.warning("Please select exactly 3 movies.")

st.sidebar.header("Recommendation Mode")
mode = st.sidebar.radio("Choose mode", ["User-based CF", "Item-based CF", "SVD"])

# -------------------------------
# Main: Recommendations
# -------------------------------
if st.button(" Recommend"):
    if len(favorite_movies) != 3:
        st.warning("Select exactly 3 favourite movies to get recommendations!")
    else:
        with st.spinner("Generating recommendations..."):
            if mode == "User-based CF":
                recs = recommend_user_based(favorite_movies, user_item_matrix, top_n=10)
            elif mode == "Item-based CF":
                recs = recommend_from_favorites(favorite_movies, item_similarity_df, top_n=10)
            else:
                recs = recommend_svd(favorite_movies, svd_model, movies_df, user_item_matrix, top_n=10)

        st.subheader(" Recommended Movies:")
        # Display in two columns as cards
        cols = st.columns(2)
        for i, movie in enumerate(recs):
            with cols[i % 2]:
                st.markdown(
                    f"""
                    <div style='
                        padding:15px; 
                        margin:5px; 
                        border:1px solid #ccc; 
                        border-radius:10px; 
                        background:#f9f9f9; 
                        color:#111; 
                        font-weight:bold;
                        font-size:16px;
                        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
                    '>
                    {i+1}. {movie}
                    </div>
                    """,
                    unsafe_allow_html=True
                )