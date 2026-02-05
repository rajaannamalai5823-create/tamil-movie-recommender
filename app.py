from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import requests
API_KEY = "638d402"
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ===== ML PART =====
df = pd.read_csv("tamilmovies.csv")

df = df[['MovieName', 'Genre', 'Director', 'Actor']]
df['Genre'] = df['Genre'].str.lower()
df['Director'] = df['Director'].str.lower()
df['Actor'] = df['Actor'].str.lower()

df['tags'] = df['Genre'] + ' ' + df['Director'] + ' ' + df['Actor']

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(df['tags']).toarray()
similarity = cosine_similarity(vectors)
def fetch_poster(movie_name):
    url = f"http://www.omdbapi.com/?t={movie_name}&apikey={API_KEY}"
    response = requests.get(url)
    data = response.json()

    if data.get("Poster") and data["Poster"] != "N/A":
        return data["Poster"]
    return None


def recommend(movie):
    movie = movie.lower()
    if movie not in df['MovieName'].str.lower().values:
        return []

    index = df[df['MovieName'].str.lower() == movie].index[0]
    distances = similarity[index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:9]

    results = []

    results = []

    for i in movies_list:
      movie_title = df.iloc[i[0]].MovieName
      poster = fetch_poster(movie_title)

      results.append({
        "title": movie_title,
        "poster": poster
    })

    return results




# ===== FLASK PART =====
@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = []
    if request.method == 'POST':
        movie_name = request.form['movie']
        recommendations = recommend(movie_name)

    return render_template('index.html', recommendations=recommendations)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

