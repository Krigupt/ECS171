from flask import Flask, request, jsonify,render_template,redirect,url_for, session
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances
import difflib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from forms import MusicForm

data = pd.read_csv("/Users/krishna/Desktop/ecs171/group_project/data.csv")


# Set up the song clustering pipeline
song_cluster_pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('kmeans', KMeans(n_clusters=20, verbose=False, n_init=10))
])
X = data.select_dtypes(np.number)
number_cols = list(X.columns)
song_cluster_pipeline.fit(X)
song_cluster_labels = song_cluster_pipeline.predict(X)
data['cluster_label'] = song_cluster_labels

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="6a8146dae2da400daa1f9dcbf1e58a3c",
                                                           client_secret="4b4d40f840944fea9354e28d3f3e88af"))

def find_song(name, year):
    song_data = defaultdict()
    results = sp.search(q= 'track: {} year: {}'.format(name,year), limit=1)
    if results['tracks']['items'] == []:
        return None

    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]

    song_data['name'] = [name]
    song_data['year'] = [year]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)





number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']


def get_song_data(song, spotify_data):
    
    try:
        song_data = spotify_data[(spotify_data['name'] == song['name']) 
                                & (spotify_data['year'] == song['year'])].iloc[0]
        return song_data
    
    except IndexError:
        return find_song(song['name'], song['year'])
        

def get_mean_vector(song_list, spotify_data):
    
    song_vectors = []
    
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)  
    
    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)


def flatten_dict_list(dict_list):
    
    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []
    
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
            
    return flattened_dict




def recommend_songs( song_list, spotify_data, n_songs=10):
    
    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)
    
    song_center = get_mean_vector(song_list, spotify_data)
    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])
    
    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    return rec_songs[metadata_cols].to_dict(orient='records')



app = Flask(__name__)
app.config['SECRET_KEY'] = 'KK123//'



@app.route('/', methods=['GET', 'POST'])
def recommend():
    form = MusicForm()
    recommended_songs = []
    if request.method == 'POST' and form.validate_on_submit():
        # Process the form data
        song_name = form.song_name.data
        song_year = form.song_year.data
        # Add your logic for recommendation here
        try:
            song_year = int(song_year)
        except ValueError:
            return "Invalid year input. Please enter a valid year."
        try:
            recommended_songs = recommend_songs([{'name': song_name, 'year': song_year}], data)
            # print(f'Recommended Songs: {recommended_songs}')
            session['recommended_songs'] = recommended_songs 
            return redirect(url_for('success'))
        except Exception as e:
            print(f'Error: {e}')
            return "An error occurred while processing your request."

    return render_template("test.html", form=form,songs=recommended_songs)



@app.route('/success', methods=['GET', 'POST'])
def success():
    # return "Form successfully submitted!"
    recommended_songs = session.get('recommended_songs', [])
    return render_template("success.html",songs=recommended_songs)





if __name__ == '__main__':
    app.run(debug=True)