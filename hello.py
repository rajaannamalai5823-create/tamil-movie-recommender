from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
df = pd.read_csv("tamilmovies.csv")
print(df.head())
print(df.columns)
print(df.shape)
print(df.info())
df = df[['MovieName', 'Genre', 'Director', 'Actor']]
df['Genre'] = df['Genre'].str.lower()
df['Director'] = df['Director'].str.lower()
df['Actor'] = df['Actor'].str.lower()
df['tags'] = df['Genre'] + ' ' + df['Director'] + ' ' + df['Actor']
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(df['tags']).toarray()
similarity = cosine_similarity(vectors)
def recommend(movie_name):
    movie_index = df[df['MovieName'] == movie_name].index[0]
    distances = similarity[movie_index]
    
    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]
    
    print("Recommended Movies:")
    for i in movies_list:
        print(df.iloc[i[0]].MovieName)
print(df.head())
print(vectors.shape)
print(similarity.shape)
recommend()







