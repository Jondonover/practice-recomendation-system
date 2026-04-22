import streamlit as st
st.title("My First Recommnendation System")

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# --- FUNCTIONS ---
@st.cache_resource
def prepare_data():
    ratings = pd.read_csv("Data/ml-latest-small/ratings.csv")
    movies = pd.read_csv("Data/ml-latest-small/movies.csv")
    
    # Create the matrix
    matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
    
    # Create the similarity
    sim = cosine_similarity(matrix)
    sim_df = pd.DataFrame(sim, index=matrix.index, columns=matrix.index)
    
    return ratings, movies, matrix, sim_df

# Load everything once
ratings, movies, user_item_matrix, user_similarity_df = prepare_data()

def get_recommendations_for_new_user(new_user_ratings, n=5):
    # 1. Convert the new ratings into a Series that matches our matrix columns
    new_user_series = pd.Series(0, index=user_item_matrix.columns)
    for movie_id, rating in new_user_ratings.items():
        new_user_series[movie_id] = rating

    # 2. Calculate Similarity between the New User and all existing users
    # We use cosine_similarity between the new vector and our matrix
    from sklearn.metrics.pairwise import cosine_similarity
    sim_scores = cosine_similarity(new_user_series.values.reshape(1, -1), user_item_matrix)
    sim_weights = pd.Series(sim_scores.flatten(), index=user_item_matrix.index)

    # 3. Weighted average of all movie ratings based on similarity
    recommendation_scores = user_item_matrix.T.dot(sim_weights)
    
    # 4. Filter out movies the user just rated
    unseen_scores = recommendation_scores.drop(new_user_ratings.keys(), errors='ignore')
    
    # 5. Get top titles
    top_ids = unseen_scores.sort_values(ascending=False).head(n).index
    return movies[movies['movieId'].isin(top_ids)][['title', 'genres']]

# --- USER INTERFACE ---
st.title("🔍 Search & Rate Movies")

# 1. Search Bar (Multiselect)
# Users can type to filter movie titles
selected_titles = st.multiselect(
    "Search for movies you have seen:",
    options=movies['title'].values,
    placeholder="Type a movie title (e.g., Toy Story)"
)

# 2. Dynamic Sliders
new_ratings = {}
if selected_titles:
    st.write("### Rate these movies:")
    # We create a slider for every movie selected in the search bar
    for title in selected_titles:
        rating = st.slider(f"How much did you like **{title}**?", 1.0, 5.0, 3.5, step=0.5)
        
        # Get the ID for this movie title
        m_id = movies[movies['title'] == title]['movieId'].values[0]
        new_ratings[m_id] = rating

    # 3. Submit Button
    if st.button("Get My Recommendations"):
        if len(new_ratings) > 0:
            with st.spinner("Finding matches..."):
                # Use the function we built in the previous step
                results = get_recommendations_for_new_user(new_ratings)
                st.write("---")
                st.subheader("Personalized for you:")
                st.table(results)
        else:
            st.warning("Please rate at least one movie first!")
else:
    st.info("Start typing in the search bar above to select movies you've watched.")
    