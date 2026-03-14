# 🎬 Movie Recommendation App – MoSCoW Roadmap

## Must-Have
These are essential to make your app functional, stable, and user-ready.

- **Core Recommendation Engine**
  - [x] User-based CF, Item-based CF, and SVD models work reliably.
  - Handle unseen movies gracefully.
  - [x] Return top-N recommendations based on 3 favourite movies.

- **Streamlit UI/UX Basics**
  - Clean interface: title, instructions, movie selectors, mode selection, and “Recommend” button.
  - Display recommendations in an ordered, readable list.
  - Prevent errors when users select the same movie twice.

- **Model Persistence**
  - Save and load models (`user_item_matrix`, `user_similarity_df`, `item_similarity_df`, `svd_model`) using `pickle` or `joblib`.
  - Ensure compatibility across Python/pandas versions.

- **Error Handling**
  - Handle missing movies or ratings.
  - Provide clear feedback if recommendation fails.

- **Environment & Dependencies**
  - `requirements.txt` with exact versions.
  - Instructions to set up virtual environment.

---

## Should-Have
Important for professional UX, but not blocking core functionality.

- **Improved UI**
  - Multi-select dropdown for favourite movies instead of 3 separate selectboxes.
  - Use sidebar or tabs for mode selection.
  - Display movie posters for recommendations.
  - Highlight top recommendation visually.

- **Performance**
  - Cache model loads with `st.cache_data` or `st.cache_resource`.
  - Optimise similarity calculations for larger datasets.

- **User Feedback**
  - Show progress or loading spinner while generating recommendations.
  - Option to reset selections without refreshing the page.

- **Recommendation Details**
  - Show predicted rating scores.
  - Provide explanation of why a movie is recommended.

---

## Could-Have
Nice-to-have features that enhance user engagement.

- **Personalised Experience**
  - Save user history or favourites locally.
  - Option to log in and maintain preferences.

- **Advanced Recommendation**
  - Combine User/Item CF and SVD for hybrid recommendations.
  - Allow users to input more than 3 favourites.

- **Analytics & Insights**
  - Display top trending movies.
  - Precision@K or RMSE metrics for SVD visualisation.

- **Visual Enhancements**
  - Dark/light mode toggle.
  - Movie posters or thumbnails in recommendation list.
  - Tooltips or popups with movie description.

---

## Won’t-Have (for now)
Features outside current scope, for future versions.

- Full account system with backend database.
- Large-scale deployment optimisation (cloud hosting, containerisation).
- Real-time collaborative filtering updates.
- Automatic dataset updates or integration with APIs (TMDb, IMDb, etc.).