# import streamlit as st
# from processing import preprocess
# from processing.display import Main
# import pandas as pd

# # Page configuration
# st.set_page_config(
#     page_title="Movie Recommender",
#     layout="wide",  # Set layout to wide for full-page width
#     page_icon="🎥",
#     initial_sidebar_state="collapsed",  # Collapsed sidebar minimizes the sidebar
# )

# # Custom CSS for styling
# st.markdown(
#     """
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
#     html, body, [class*="css"] {
#         font-family: 'Poppins', sans-serif;
#         background-color: #1a1a2e;
#         color: #ffffff;
#     }
#     .main-container {
#         padding: 2rem;
#         border-radius: 8px;
#         background-color: #282c34;
#     }
#     .title {
#         text-align: center;
#         font-weight: 600;
#         color: #ff9f43;
#     }
#     .search-section {
#         margin-top: 2rem;
#         text-align: center;
#     }
#     .details-section, .recommend-section {
#         margin-top: 1rem;
#     }
#     .recommend-section h2 {
#         color: #ff9f43;
#     }
#     iframe {
#         visibility: hidden !important; /* Completely hides iframe */
#         height: 0 !important;
#         width: 0 !important;
#         position: absolute !important;
#     }
#     button[title="View fullscreen"] {
#         display: none !important; /* Hides the fullscreen button specifically */
#     }
#     .separator {
#         border-left: 2px solid #ff9f43;
#         height: 100%;
#         margin: auto;
#     }
#     .home-icon {
#         font-size: 1.5rem;
#         color: #ff9f43;
#         cursor: pointer;
#         text-align: left;
#         margin: 10px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# # Initialize session state variables
# if "selected_movie_name" not in st.session_state:
#     st.session_state["selected_movie_name"] = None
# if "search_performed" not in st.session_state:
#     st.session_state["search_performed"] = False


# def fetch_popular_movies(new_df, num_movies=24):
#     """Fetch popular movies randomly from the dataset."""
#     if new_df.empty or "title" not in new_df.columns or "movie_id" not in new_df.columns:
#         st.error("Dataset is empty or missing required columns.")
#         return [], []

#     popular_movies = new_df.sample(n=min(num_movies, len(new_df)), random_state=42)
#     titles = popular_movies["title"].tolist()
#     movie_ids = popular_movies["movie_id"].tolist()
#     posters = [preprocess.fetch_posters(movie_id) for movie_id in movie_ids]
#     return titles, posters


# def fetch_top_rated_movies(new_df, num_movies=24):
#     """Fetch top-rated movies randomly from the dataset."""
#     if new_df.empty or "title" not in new_df.columns or "movie_id" not in new_df.columns:
#         st.error("Dataset is empty or missing required columns.")
#         return [], []

#     top_rated_movies = new_df.sample(n=min(num_movies, len(new_df)), random_state=43)  # Different seed for variety
#     titles = top_rated_movies["title"].tolist()
#     movie_ids = top_rated_movies["movie_id"].tolist()
#     posters = [preprocess.fetch_posters(movie_id) for movie_id in movie_ids]
#     return titles, posters


# def recommend_display(new_df):
#     """Display movie search, details of the selected movie, and recommendations."""
#     # App Title
#     st.markdown("<h1 class='title'>🎥 Movie Recommender</h1>", unsafe_allow_html=True)
#     st.markdown("<h2 class='title'>🎞️ Over 4000 movies and recommendations 🍿📽️📺🎧</h2>", unsafe_allow_html=True)

#     # Add Home Icon
#     if st.session_state["search_performed"] or st.session_state["selected_movie_name"]:
#         if st.button("⬅️ Home"):
#             # Reset session state variables
#             st.session_state["selected_movie_name"] = None
#             st.session_state["search_performed"] = False
#             st.rerun()

#     # Search Section
#     st.markdown("<div class='search-section'>", unsafe_allow_html=True)
#     movie_titles = new_df["title"].dropna().unique().tolist()
#     search_query = st.selectbox("Search for a Movie:", options=[""] + movie_titles)

#     if st.button("CLICK HERE TO SEARCH"):
#         if search_query:
#             movie_found = new_df[new_df["title"].str.lower() == search_query.lower()]
#             if not movie_found.empty:
#                 st.session_state["selected_movie_name"] = movie_found["title"].iloc[0]
#                 st.session_state["search_performed"] = True
#                 st.rerun()
#             else:
#                 st.error("Movie not found. Please try another name.")
#         else:
#             st.warning("Please select or enter a movie name before clicking.")

#     st.markdown("</div>", unsafe_allow_html=True)

#     # Display Popular Movies and Top Rated Movies side by side
#     if not st.session_state["search_performed"]:
#         st.markdown("<div class='recommend-section'>", unsafe_allow_html=True)

#         cols = st.columns([1, 0.05, 1])  # Two equal-width columns with a separator

#         with cols[0]:
#             st.subheader("🔥 Popular Movies")
#             try:
#                 popular_titles, popular_posters = fetch_popular_movies(new_df)
#                 for i in range(0, len(popular_titles), 4):  # Display movies in rows of 4
#                     row_cols = st.columns(4)
#                     for j, col in enumerate(row_cols):
#                         if i + j < len(popular_titles):
#                             with col:
#                                 st.image(popular_posters[i + j], width=150)
#                                 if st.button(popular_titles[i + j], key=f"popular_{i + j}"):
#                                     st.session_state["selected_movie_name"] = popular_titles[i + j]
#                                     st.session_state["search_performed"] = True
#                                     st.rerun()
#             except Exception as e:
#                 st.error(f"Error fetching popular movies: {e}")

#         with cols[1]:  # Separator
#             st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

#         with cols[2]:
#             st.subheader("⭐ Recent Releases")
#             try:
#                 top_rated_titles, top_rated_posters = fetch_top_rated_movies(new_df)
#                 for i in range(0, len(top_rated_titles), 4):  # Display movies in rows of 4
#                     row_cols = st.columns(4)
#                     for j, col in enumerate(row_cols):
#                         if i + j < len(top_rated_titles):
#                             with col:
#                                 st.image(top_rated_posters[i + j], width=150)
#                                 if st.button(top_rated_titles[i + j], key=f"top_{i + j}"):
#                                     st.session_state["selected_movie_name"] = top_rated_titles[i + j]
#                                     st.session_state["search_performed"] = True
#                                     st.rerun()
#             except Exception as e:
#                 st.error(f"Error fetching top-rated movies: {e}")

#         st.markdown("</div>", unsafe_allow_html=True)

#     # If a movie is selected, display its details and recommendations
#     if st.session_state["selected_movie_name"]:
#         selected_movie = st.session_state["selected_movie_name"]

#         # Use columns for side-by-side layout
#         cols = st.columns([1, 0.05, 2])  # Add a small column in the middle for the separator

#         with cols[0]:  # Left column for movie details
#             st.markdown("<div class='details-section'>", unsafe_allow_html=True)
#             st.header(f"🎬 Details for **{selected_movie}**")
#             try:
#                 info = preprocess.get_details(selected_movie)

#                 if not info:
#                     st.warning("Details not available.")
#                 else:
#                     poster_url = info.get("poster_url")
#                     genres = info.get("genres")
#                     overview = info.get("overview")
#                     release_date = info.get("release_date")
#                     runtime = info.get("runtime")
#                     vote_rating = info.get("vote_rating")
#                     cast = info.get("cast")
#                     director = info.get("director")

#                     st.image(poster_url, width=300)
#                     st.subheader(selected_movie)
#                     st.markdown(f"**Rating:** {vote_rating} ⭐")
#                     st.markdown(f"**Runtime:** {runtime} minutes")
#                     st.markdown(f"**Release Date:** {release_date}")
#                     st.markdown(f"**Genres:** {genres}")
#                     st.markdown(f"**Director:** {director}")
#                     st.markdown("**Overview:**")
#                     st.write(overview)

#             except Exception as e:
#                 st.error(f"Error fetching movie details: {e}")
#             st.markdown("</div>", unsafe_allow_html=True)

#         with cols[1]:  # Middle column for separator
#             st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

#         with cols[2]:  # Right column for recommendations
#             st.markdown("<div class='recommend-section'>", unsafe_allow_html=True)
#             st.subheader(f"🔍 Recommendations based on **{selected_movie}**")
#             try:
#                 movies, posters = preprocess.recommend(new_df, selected_movie, r"Files/similarity_tags_tags.pkl")
#                 for i in range(0, len(movies), 4):  # Display movies in rows of 4
#                     rec_cols = st.columns(4)
#                     for j, rec_col in enumerate(rec_cols):
#                         if i + j < len(movies):
#                             with rec_col:
#                                 st.image(posters[i + j], width=150)
#                                 if st.button(movies[i + j], key=f"movie_{i + j}"):
#                                     st.session_state["selected_movie_name"] = movies[i + j]
#                                     st.rerun()
#             except Exception as e:
#                 st.error(f"Error fetching recommendations: {e}")
#             st.markdown("</div>", unsafe_allow_html=True)


# def main():
#     """Main function to run the Streamlit app."""
#     with Main() as bot:
#         bot.main_()
#         new_df, _, _ = bot.getter()
#         recommend_display(new_df)


# if __name__ == "__main__":
#     main()

# import streamlit as st
# from processing import preprocess
# from processing.display import Main
# import pandas as pd

# # Page configuration
# st.set_page_config(
#     page_title="Movie Recommender",
#     layout="wide",  # Set layout to wide for full-page width
#     page_icon="🎥",
#     initial_sidebar_state="collapsed",  # Collapsed sidebar minimizes the sidebar
# )

# # Custom CSS for styling
# st.markdown(
#     """
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
#     html, body, [class*="css"] {
#         font-family: 'Poppins', sans-serif;
#         background-color: #1a1a2e;
#         color: #ffffff;
#     }
#     .main-container {
#         padding: 2rem;
#         border-radius: 8px;
#         background-color: #282c34;
#     }
#     .title {
#         text-align: center;
#         font-weight: 600;
#         color: #ff9f43;
#     }
#     .search-section {
#         margin-top: 2rem;
#         text-align: center;
#     }
#     .details-section, .recommend-section {
#         margin-top: 1rem;
#     }
#     .recommend-section h2 {
#         color: #ff9f43;
#     }
#     iframe {
#         visibility: hidden !important; /* Completely hides iframe */
#         height: 0 !important;
#         width: 0 !important;
#         position: absolute !important;
#     }
#     button[title="View fullscreen"] {
#         display: none !important; /* Hides the fullscreen button specifically */
#     }
#     .separator {
#         border-left: 2px solid #ff9f43;
#         height: 100%;
#         margin: auto;
#     }
#     .home-icon {
#         font-size: 1.5rem;
#         color: #ff9f43;
#         cursor: pointer;
#         text-align: left;
#         margin: 10px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# # Initialize session state variables
# if "selected_movie_name" not in st.session_state:
#     st.session_state["selected_movie_name"] = None
# if "search_performed" not in st.session_state:
#     st.session_state["search_performed"] = False


# def fetch_popular_movies(new_df, num_movies=24):
#     """Fetch popular movies randomly from the dataset."""
#     if new_df.empty or "title" not in new_df.columns or "movie_id" not in new_df.columns:
#         st.error("Dataset is empty or missing required columns.")
#         return [], []

#     popular_movies = new_df.sample(n=min(num_movies, len(new_df)), random_state=42)
#     titles = popular_movies["title"].tolist()
#     movie_ids = popular_movies["movie_id"].tolist()
#     posters = [preprocess.fetch_posters(movie_id) for movie_id in movie_ids]
#     return titles, posters


# def fetch_top_rated_movies(new_df, num_movies=24):
#     """Fetch top-rated movies randomly from the dataset."""
#     if new_df.empty or "title" not in new_df.columns or "movie_id" not in new_df.columns:
#         st.error("Dataset is empty or missing required columns.")
#         return [], []

#     top_rated_movies = new_df.sample(n=min(num_movies, len(new_df)), random_state=43)  # Different seed for variety
#     titles = top_rated_movies["title"].tolist()
#     movie_ids = top_rated_movies["movie_id"].tolist()
#     posters = [preprocess.fetch_posters(movie_id) for movie_id in movie_ids]
#     return titles, posters


# def recommend_display(new_df):
#     """Display movie search, details of the selected movie, and recommendations."""
#     # App Title
#     st.markdown("<h1 class='title'>🎥 Movie Recommender</h1>", unsafe_allow_html=True)
#     st.markdown("<h2 class='title'>🎞️ Over 4000 movies and recommendations 🍿📽️📺🎧</h2>", unsafe_allow_html=True)

#     # Add Home Icon
#     if st.session_state["search_performed"] or st.session_state["selected_movie_name"]:
#         if st.button("⬅️ Home"):
#             # Reset session state variables
#             st.session_state["selected_movie_name"] = None
#             st.session_state["search_performed"] = False
#             st.experimental_rerun()

#     # Search Section
#     st.markdown("<div class='search-section'>", unsafe_allow_html=True)
#     movie_titles = new_df["title"].dropna().unique().tolist()
#     search_query = st.selectbox("Search for a Movie:", options=[""] + movie_titles)

#     if st.button("CLICK HERE TO SEARCH"):
#         if search_query:
#             movie_found = new_df[new_df["title"].str.lower() == search_query.lower()]
#             if not movie_found.empty:
#                 st.session_state["selected_movie_name"] = movie_found["title"].iloc[0]
#                 st.session_state["search_performed"] = True
#                 st.rerun()
#             else:
#                 st.error("Movie not found. Please try another name.")
#         else:
#             st.warning("Please select or enter a movie name before clicking.")

#     st.markdown("</div>", unsafe_allow_html=True)

#     # Display Popular Movies and Top Rated Movies side by side
#     if not st.session_state["search_performed"]:
#         st.markdown("<div class='recommend-section'>", unsafe_allow_html=True)

#         cols = st.columns([1, 0.05, 1])  # Two equal-width columns with a separator

#         with cols[0]:
#             st.subheader("🔥 Popular Movies")
#             try:
#                 popular_titles, popular_posters = fetch_popular_movies(new_df)
#                 for i in range(0, len(popular_titles), 4):  # Display movies in rows of 4
#                     row_cols = st.columns(4)
#                     for j, col in enumerate(row_cols):
#                         if i + j < len(popular_titles):
#                             with col:
#                                 st.image(popular_posters[i + j], width=150)
#                                 if st.button(popular_titles[i + j], key=f"popular_{i + j}"):
#                                     st.session_state["selected_movie_name"] = popular_titles[i + j]
#                                     st.session_state["search_performed"] = True
#                                     st.rerun()
#             except Exception as e:
#                 st.error(f"Error fetching popular movies: {e}")

#         with cols[1]:  # Separator
#             st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

#         with cols[2]:
#             st.subheader("⭐ Recent Releases")
#             try:
#                 top_rated_titles, top_rated_posters = fetch_top_rated_movies(new_df)
#                 for i in range(0, len(top_rated_titles), 4):  # Display movies in rows of 4
#                     row_cols = st.columns(4)
#                     for j, col in enumerate(row_cols):
#                         if i + j < len(top_rated_titles):
#                             with col:
#                                 st.image(top_rated_posters[i + j], width=150)
#                                 if st.button(top_rated_titles[i + j], key=f"top_{i + j}"):
#                                     st.session_state["selected_movie_name"] = top_rated_titles[i + j]
#                                     st.session_state["search_performed"] = True
#                                     st.rerun()
#             except Exception as e:
#                 st.error(f"Error fetching top-rated movies: {e}")

#         st.markdown("</div>", unsafe_allow_html=True)

#     # If a movie is selected, display its details and recommendations
#     if st.session_state["selected_movie_name"]:
#         selected_movie = st.session_state["selected_movie_name"]

#         # Use columns for side-by-side layout
#         cols = st.columns([1, 0.05, 2])  # Add a small column in the middle for the separator

#         with cols[0]:  # Left column for movie details
#             st.markdown("<div class='details-section'>", unsafe_allow_html=True)
#             st.header(f"🎬 Details for **{selected_movie}**")
#             try:
#                 info = preprocess.get_details(selected_movie)

#                 if not info:
#                     st.warning("Details not available.")
#                 else:
#                     poster_url = info["poster_url"]
#                     genres = info["genres"]
#                     overview = info["overview"]
#                     release_date = info["release_date"]
#                     runtime = info["runtime"]
#                     vote_rating = info["vote_rating"]
#                     cast = info["cast"]
#                     director = info["director"]

#                     st.image(poster_url, width=300)
#                     st.subheader(selected_movie)
#                     st.markdown(f"**Rating:** {vote_rating} ⭐")
#                     st.markdown(f"**Runtime:** {runtime} minutes")
#                     st.markdown(f"**Release Date:** {release_date}")
#                     st.markdown(f"**Genres:** {genres}")
#                     st.markdown(f"**Director:** {director}")
#                     st.markdown("**Overview:**")
#                     st.write(overview)

#             except Exception as e:
#                 st.error(f"Error fetching movie details: {e}")
#             st.markdown("</div>", unsafe_allow_html=True)

#         with cols[1]:  # Middle column for separator
#             st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

#         with cols[2]:  # Right column for recommendations
#             st.markdown("<div class='recommend-section'>", unsafe_allow_html=True)
#             st.subheader(f"🔍 Recommendations based on **{selected_movie}**")
#             try:
#                 movies, posters = preprocess.recommend(new_df, selected_movie, r"Files/similarity_tags_tags.pkl")
#                 for i in range(0, len(movies), 4):  # Display movies in rows of 4
#                     rec_cols = st.columns(4)
#                     for j, rec_col in enumerate(rec_cols):
#                         if i + j < len(movies):
#                             with rec_col:
#                                 st.image(posters[i + j], width=150)
#                                 if st.button(movies[i + j], key=f"movie_{i + j}"):
#                                     st.session_state["selected_movie_name"] = movies[i + j]
#                                     st.rerun()
#             except Exception as e:
#                 st.error(f"Error fetching recommendations: {e}")
#             st.markdown("</div>", unsafe_allow_html=True)


# def main():
#     """Main function to run the Streamlit app."""
#     with Main() as bot:
#         bot.main_()
#         new_df, _, _ = bot.getter()
#         recommend_display(new_df)


# if __name__ == "__main__":
#     main()


import streamlit as st
from processing import preprocess
from processing.display import Main
import pandas as pd
import requests

# Page configuration
st.set_page_config(
    page_title="Movie recommender",
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

