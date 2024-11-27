import ast
import string
import pickle
import pandas as pd
import requests
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

nltk.download("stopwords")
ps = PorterStemmer()


def get_genres(obj):
    return [i["name"] for i in ast.literal_eval(obj)]


def get_cast(obj):
    return [member["name"] for member in ast.literal_eval(obj)[:10]]


def get_crew(obj):
    return [member["name"] for member in ast.literal_eval(obj) if member["job"] == "Director"]


def stemming_stopwords(text):
    words = [ps.stem(word.lower()) for word in text if word.lower() not in stopwords.words("english")]
    return " ".join(words).translate(str.maketrans("", "", string.punctuation))


def fetch_posters(movie_id):
    response = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=6bc3065293c944b5ad11cb7cd15c076e")
    data = response.json()
    return f"https://image.tmdb.org/t/p/w780/{data.get('poster_path', '')}"


def recommend(new_df, movie, pickle_file_path):
    with open(pickle_file_path, "rb") as pickle_file:
        similarity_tags = pickle.load(pickle_file)

    try:
        movie_idx = new_df[new_df["title"].str.lower() == movie.lower()].index[0]
    except IndexError:
        return [], []

    similar_movies = sorted(enumerate(similarity_tags[movie_idx]), key=lambda x: x[1], reverse=True)[1:13]
    rec_movie_list = [new_df.iloc[i[0]]["title"] for i in similar_movies]
    rec_poster_list = [fetch_posters(new_df.iloc[i[0]]["movie_id"]) for i in similar_movies]
    return rec_movie_list, rec_poster_list


def get_details(selected_movie_name):
    # Load the DataFrame from pickled dictionaries
    with open(r"Files/movies_dict.pkl", "rb") as pickle_file:
        loaded_dict = pickle.load(pickle_file)
    movies = pd.DataFrame.from_dict(loaded_dict)

    with open(r"Files/movies2_dict.pkl", "rb") as pickle_file:
        loaded_dict_2 = pickle.load(pickle_file)
    movies2 = pd.DataFrame.from_dict(loaded_dict_2)

    # Find the movie row
    try:
        movie_data = movies2[movies2["title"].str.lower() == selected_movie_name.lower()].iloc[0]
    except IndexError:
        return None  # Movie not found

    # Extract details with fallback values
    info = {
        "poster_url": fetch_posters(movie_data.get("movie_id", "")),
        "budget": movie_data.get("budget", "N/A"),
        "genres": movie_data.get("genres", "N/A"),
        "overview": movie_data.get("overview", "No overview available."),
        "release_date": movie_data.get("release_date", "N/A"),
        "revenue": movie_data.get("revenue", "N/A"),
        "runtime": movie_data.get("runtime", "Unknown"),  # Default to "Unknown" if not found
        "available_languages": [lang["name"] for lang in ast.literal_eval(movie_data.get("spoken_languages", "[]"))],
        "vote_rating": movie_data.get("vote_average", "N/A"),
        "vote_count": movie_data.get("vote_count", "N/A"),
        "cast": movies.loc[movies["title"].str.lower() == selected_movie_name.lower(), "cast"].iloc[0],
        "director": movies.loc[movies["title"].str.lower() == selected_movie_name.lower(), "director"].iloc[0],
    }

    return info
