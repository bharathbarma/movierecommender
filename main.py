import streamlit as st
from processing import preprocess
from processing.display import Main
import pandas as pd
import requests

# Page configuration
st.set_page_config(
    page_title="Movie Recommender",
    layout="centered",  # Adjust layout to 'centered' for a cleaner look
    page_icon="🎥",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        background-color: #1a1a2e;
        color: #ffffff;
    }
    .main-container {
        padding: 2rem;
        border-radius: 8px;
        background-color: #282c34;
    }
    .title {
        text-align: center;
        font-weight: 600;
        color: #ff9f43;
    }
    .recommend-section, .details-section {
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state variables
if "selected_movie_name" not in st.session_state:
    st.session_state["selected_movie_name"] = None

if "show_recommendations" not in st.session_state:
    st.session_state["show_recommendations"] = True

# TMDB API key
TMDB_API_KEY = "6bc3065293c944b5ad11cb7cd15c076e"

def fetch_posters(movie_id):
    """
    Fetch the poster URL for a given movie ID using TMDB API.
    """
    response = requests.get(
        f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
    )
    data = response.json()
    try:
        return f"https://image.tmdb.org/t/p/w780/{data['poster_path']}"
    except KeyError:
        return "https://via.placeholder.com/150"

def recommend_display(new_df):
    """
    Display movie recommendations and handle movie selection with a predictive dropdown.
    """
    st.markdown("<h1 class='title'>🎥 Movie Recommender</h1>", unsafe_allow_html=True)
    st.markdown("<div class='recommend-section'>", unsafe_allow_html=True)

    # Get unique movie titles for the dropdown
    movie_titles = new_df["title"].dropna().unique().tolist()

    # Predictive dropdown for movie search
    search_query = st.selectbox(
        "Search for a Movie:",
        options=[""] + movie_titles,
        format_func=lambda x: "Type to search..." if x == "" else x,
    )

    recommend_button = st.button("Get Recommendations")

    if recommend_button:
        if search_query and search_query != "Type to search...":
            movie_found = new_df[new_df["title"].str.contains(search_query, case=False, na=False)]
            if not movie_found.empty:
                st.session_state["selected_movie_name"] = movie_found["title"].iloc[0]
                st.session_state["show_recommendations"] = True
                st.experimental_rerun()
            else:
                st.error("Movie not found. Please try another name.")
        else:
            st.warning("Please select or enter a movie name before clicking 'Get Recommendations'.")

    if st.session_state["selected_movie_name"]:
        st.subheader(f"Recommended Movies for '{st.session_state['selected_movie_name']}'")
        st.write("---")
        try:
            movies, posters = preprocess.recommend(new_df, st.session_state["selected_movie_name"], r"Files/similarity_tags_tags.pkl")
            for i in range(0, len(movies), 4):
                cols = st.columns(4)
                for j, col in enumerate(cols):
                    if i + j < len(movies):
                        with col:
                            # Use a unique key by combining movie title and index
                            st.image(posters[i + j], width=140)
                            if st.button(movies[i + j], key=f"movie_{movies[i + j]}_{i + j}"):
                                st.session_state["selected_movie_name"] = movies[i + j]
                                st.session_state["show_recommendations"] = False
                                st.experimental_rerun()
        except Exception as e:
            st.error(f"Error fetching recommendations: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

def display_movie_details():
    """
    Display details for the selected movie.
    """
    selected_movie_name = st.session_state.get("selected_movie_name", None)

    if not selected_movie_name:
        st.warning("No movie selected. Please select a movie from recommendations.")
        return

    st.markdown("<div class='details-section'>", unsafe_allow_html=True)
    st.title(f"🎥 Details for {selected_movie_name}")
    st.write("---")

    try:
        info = preprocess.get_details(selected_movie_name)
        if not info:
            st.warning("Details not available for the selected movie.")
            return

        image_col, text_col = st.columns((1, 2))
        with image_col:
            st.image(info[0], width=140)

        with text_col:
            st.subheader(selected_movie_name)
            st.text(f"Rating: {info[8]} | Runtime: {info[6]}")
            st.write("Overview:")
            st.write(info[3])

        if st.button("Back to Recommendations"):
            st.session_state["show_recommendations"] = True
            st.experimental_rerun()
    except Exception as e:
        st.error(f"Error fetching movie details: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

def main():
    """
    Main function to run the app.
    """
    with Main() as bot:
        bot.main_()
        new_df, movies, _ = bot.getter()

        if st.session_state["show_recommendations"]:
            recommend_display(new_df)
        else:
            display_movie_details()

if __name__ == "__main__":
    main()
