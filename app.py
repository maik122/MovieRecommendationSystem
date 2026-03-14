import streamlit as st
import pandas as pd
import pickle

# -------------------------------
# Load saved models safely
# -------------------------------
user_item_matrix = pd.read_pickle("models/user_item_matrix.pkl")
item_similarity_df = pd.read_pickle("models/item_similarity.pkl")
movies_df = pd.read_pickle("models/movies.pkl")

with open("models/svd_model.pkl", "rb") as f:
    svd_model = pickle.load(f)

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
    from sklearn.metrics.pairwise import cosine_similarity
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
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")
st.title("🎬 Movie Recommendation System")
st.markdown("""
Welcome! Select **3 of your favourite movies** below, choose a recommendation mode, 
and click **Recommend** to get your personalised top 10 movie suggestions.
""")

# Sidebar for mode selection
mode = st.sidebar.radio("Recommendation Mode", ["User-based CF", "Item-based CF", "SVD"])
st.sidebar.markdown("**Select 3 favorite movies:**")

# Favorite movies selection in sidebar
favorite_movies = []
for i in range(3):
    movie = st.sidebar.selectbox(f"Movie #{i+1}", options=movie_titles, key=i)
    favorite_movies.append(movie)

st.write("---")

if st.button("🎯 Recommend"):
    with st.spinner("Generating recommendations..."):
        if mode == "User-based CF":
            recs = recommend_user_based(favorite_movies, user_item_matrix, top_n=10)
        elif mode == "Item-based CF":
            recs = recommend_from_favorites(favorite_movies, item_similarity_df, top_n=10)
        else:
            recs = recommend_svd(favorite_movies, svd_model, movies_df, user_item_matrix, top_n=10)

    st.subheader("Recommended Movies:")
    # Display recommendations in two columns with cards
    cols = st.columns(2)
    for i, movie in enumerate(recs):
        with cols[i % 2]:
            st.markdown(f"**{i+1}. {movie}**")