from flask import Flask, request, render_template, redirect, url_for, session
import spotipy
import pandas as pd
from forms import MusicForm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict
from scipy.spatial.distance import cdist
import numpy as np


# load data using pandas
music_data = pd.read_csv("/Users/krishna/Desktop/ecs171/group_project/data.csv")

# data preprocessing pipeline
clustering_pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('kmeans', KMeans(n_clusters=20, verbose=False, n_init=10))
])

numerical_columns = music_data.select_dtypes(np.number).columns.tolist()
clustering_pipeline.fit(music_data[numerical_columns])
music_data['cluster_label'] = clustering_pipeline.predict(music_data[numerical_columns])

# defining our client object which contains API key and secret
spotify_client = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id="6a8146dae2da400daa1f9dcbf1e58a3c",
    client_secret="4b4d40f840944fea9354e28d3f3e88af"
))

# function that gets song from spotify given the name and year
def fetch_song(name, year):
    search_results = spotify_client.search(q=f'track: {name} year: {year}', limit=1)
    if not search_results['tracks']['items']:
        return None

    track_info = search_results['tracks']['items'][0]
    track_id = track_info['id']
    audio_features = spotify_client.audio_features(track_id)[0]

    song_info = {
        'name': [name],
        'year': [year],
        'explicit': [int(track_info['explicit'])],
        'duration_ms': [track_info['duration_ms']],
        'popularity': [track_info['popularity']]
    }
    song_info.update(audio_features)
    
    return pd.DataFrame(song_info)

# returns song name and year so we can display it on the frontend
def retrieve_song_data(song, data):
    song_data = data[(data['name'] == song['name']) & (data['year'] == song['year'])]
    if not song_data.empty:
        return song_data.iloc[0]
    return fetch_song(song['name'], song['year'])

#ensures duplicate song is not recommended
def flatten_dict(dict_list):
    flattened = defaultdict(list)
    for d in dict_list:
        for key, value in d.items():
            flattened[key].append(value)
    return flattened

def compute_mean_vector(songs, data):
    song_vectors = [retrieve_song_data(song, data)[numerical_columns].values for song in songs]
    song_matrix = np.array(song_vectors)
    return np.mean(song_matrix, axis=0)

# core recommendation function
def recommend_songs(songs, data, n_recommendations=10):
    metadata_columns = ['name', 'year', 'artists']
    song_dict = flatten_dict(songs)

    mean_vector = compute_mean_vector(songs, data)
    scaler = clustering_pipeline.steps[0][1]
    scaled_data = scaler.transform(data[numerical_columns])
    scaled_vector = scaler.transform(mean_vector.reshape(1, -1))
    distances = cdist(scaled_vector, scaled_data, 'cosine')
    recommended_indices = np.argsort(distances)[:, :n_recommendations][0]

    recommended_songs = data.iloc[recommended_indices]
    recommended_songs = recommended_songs[~recommended_songs['name'].isin(song_dict['name'])]
    recommendations = recommended_songs[metadata_columns].to_dict(orient='records')

    for song in recommendations:
        search_results = spotify_client.search(q=f'track:{song["name"]} year:{song["year"]}', limit=1)
        if search_results['tracks']['items']:
            track = search_results['tracks']['items'][0]
            song['album_art'] = track['album']['images'][0]['url']
            song['spotify_url'] = track['external_urls']['spotify']
        else:
            song['album_art'] = ''
            song['spotify_url'] = ''

    return recommendations

app = Flask(__name__)
app.config['SECRET_KEY'] = 'KK123//'

#backend recommendation route
@app.route('/', methods=['GET', 'POST'])
def recommend():
    form = MusicForm()
    recommendations = []
    if request.method == 'POST' and form.validate_on_submit():
        song_name = form.song_name.data
        song_year = form.song_year.data
        try:
            song_year = int(song_year)
        except ValueError:
            return "Invalid input. Enter valid year."
        try:
            recommendations = recommend_songs([{'name': song_name, 'year': song_year}], music_data)
            session['recommendations'] = recommendations 
            return redirect(url_for('success'))
        except Exception as e:
            return f"Unexpected error: {e}"
    return render_template("test.html", form=form, songs=recommendations)

# if search is successful, display 10 recommended songs
@app.route('/success', methods=['GET', 'POST'])
def success():
    recommendations = session.get('recommendations', [])
    return render_template("success.html", songs=recommendations)

if __name__ == '__main__':
    app.run(debug=True, port=5200)