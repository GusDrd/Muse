from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from joblib import load
import pandas as pd
import numpy as np
import json
import os


# Declare flask API and apply CORS
app = Flask(__name__)
CORS(app, supports_credentials=True, resources={r"*": {"origins": "*"}})


# Load representations & features GMM models
reps_gmm_model = load('/model/Reps_GMM_Model.joblib')
feature_gmm_model = load('/model/Feature_GMM_Model.joblib')

# Used for representations recommendation
dataset = None
tracks_dataset = None

# Used for features recommendation
loc_dataset = None
genre_dataset = None
feature_dataset = np.zeros((22492, 6))



##########################
# Return recommendations #
##########################
@app.route('/music', methods=['POST'])
def get_recommendations():

    # Check if request is not json for some reason
    if not request.is_json:
        return jsonify({'code': 400, 'error': 'Input format not in json'})

    content = request.get_json()
    print(content)

    # ----- Catch payload format error ----------
    if content is None or 'n_song' not in content or 'song_ids' not in content:
        return jsonify({'code': 400, 'error': 'Missing valid inputs'})

    if not isinstance(content['n_song'], int) or not isinstance(content['song_ids'], list):
        return jsonify({'code': 400, 'error': 'Invalid input format'})
    
    for input in content['song_ids']:
        if not isinstance(input, str) or not input.isnumeric():
            return jsonify({'code': 400, 'error': 'Invalid song input'})
    # ----- -------------------------- ----------

    # Obtain recommendations from both reps & features models
    recs_1, recs_2 = find_recommendations(content['n_song'], content['song_ids'])


    # Check if recommendations worked correctly. This should be unreachable but still for safety
    if recs_1 is None or recs_2 is None:
        return jsonify({'error': 400, 'message': 'Failed to process recommendations'})


    return jsonify({'song_ids_1': recs_1, 'song_ids_2': recs_2})


######################
# Save music ratings #
######################
@app.route('/ratings', methods=['POST'])
def save_ratings():

    # Check if request is not json for some reason
    if not request.is_json:
        return jsonify({'code': 400, 'error': 'Input format not in json'})

    content = request.get_json()
    print(content)

    # ----- Catch payload format error ----------
    if content is None or 'n_song' not in content or 'n_recs' not in content or ('rating_1' not in content and 'rating_2' not in content):
        return jsonify({'code': 400, 'error': 'Missing valid inputs'})

    if not isinstance(content['n_song'], int) or not isinstance(content['n_recs'], int):
        return jsonify({'code': 400, 'error': 'Invalid input format'})

    if 'rating_1' in content and not isinstance(content['rating_1'], int):
        return jsonify({'code': 400, 'error': 'Invalid input format'})
    if 'rating_2' in content and not isinstance(content['rating_2'], int):
        return jsonify({'code': 400, 'error': 'Invalid input format'})
    # ----- -------------------------- ----------

    # Load results dictionnary
    ratings = {}
    with open('/model/Ratings.json', encoding='utf-8') as file:
        ratings = json.load(file)

    # Check which ratings are present and append to corresponding result chain
    if 'rating_1' in content:
        ratings['reps_model'].append([content['rating_1'], content['n_song'], content['n_recs']])
    if 'rating_2' in content:
        ratings['features_model'].append([content['rating_2'], content['n_song'], content['n_recs']])
        
    # Save file with new results
    with open('/model/Ratings.json', 'w', encoding='utf-8') as file:
        json.dump(ratings, file)
        

    return jsonify({'code': 200, 'message': 'Ratings sent successfuly !'})



#########################
# Init track dictionary #
#########################
def init_track_data():
    print("Initializing recommendation API...")

    global dataset
    global tracks_dataset
    global loc_dataset
    global genre_dataset
    global feature_dataset

    jamendo_data = np.load('/model/jamendo_data.npy')
    jamendo_ids = np.load('/model/jamendo_ids.npy')

    with open('/model/Complete_Tracks.json', encoding='utf-8') as file:
        tracks = json.load(file)


    # Add Jukebox (CALM) representations to track dictionary
    temp = {}

    for track in tracks:
        # Skip tracks for which representations have not been processed
        if int(track) not in jamendo_ids:
            continue

        # Locate track index using id to get representations from data array
        index = np.where(jamendo_ids == int(track))[0][0]

        temp[track] = {'title': tracks[track]['title'],
                 'genre' : tracks[track]['genre'],
                 'duration' : float(tracks[track]['duration']),
                 'year' : int(tracks[track]['year']),
                 'key': tracks[track]['key'],
                 'valence': float(tracks[track]['valence']),
                 'arousal': float(tracks[track]['arousal']),
                 'representations' : jamendo_data[index]}

    # Replace old dictionary to delete unused track entries
    tracks = temp

    # Create training set by removing 2000 test songs
    split = np.load('/model/Split2000.npy')

    train_tracks = {}

    # Remove tracks used in the test split
    for track in tracks:
        if int(track) not in split:
            train_tracks[track] = tracks[track]

    dataset = pd.DataFrame.from_dict(train_tracks)
    tracks_dataset = pd.DataFrame.from_dict(tracks)
    loc_dataset = tracks_dataset.loc['title']

    # Convert genre tags to binary and then to decimal
    complete_genre = tracks_dataset.loc['genre']
    mlb = MultiLabelBinarizer()

    labels = mlb.fit_transform(complete_genre)

    complete_genres = []
    for b in labels:
        complete_genres.append(sum(val*(2**idx) for idx, val in enumerate(reversed(b))))

    genre_dataset = np.array(complete_genres)

    genre = dataset.loc['genre']

    mlb = MultiLabelBinarizer()

    labels = mlb.fit_transform(genre)

    genres = []
    for b in labels:
        genres.append(sum(val*(2**idx) for idx, val in enumerate(reversed(b))))

    dura = np.array(list(dataset.loc['duration']))
    year = np.array(list(dataset.loc['year']))
    key = np.array(list(dataset.loc['key']))
    val = np.array(list(dataset.loc['valence']))
    arou = np.array(list(dataset.loc['arousal']))
    genres = np.array(genres)

    for i in range(len(dura)):
        feature_dataset[i] = np.array([dura[i], year[i], key[i], val[i], arou[i], genres[i]])

    print("API Initialized!")



########################
# Find recommendations #
########################
def find_recommendations(n_songs, track_ids):

    # ---- Using Representations ---------------
    centers = []
    for input in track_ids:
        centers.append(tracks_dataset[input]['representations'])

    song_center = np.mean(np.array(centers), axis=0)

    scaler = reps_gmm_model.steps[0][1]
    scaled_data = scaler.transform(list(dataset.loc['representations']))
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))


    dist = cosine_similarity(scaled_song_center, scaled_data)

    index = list(np.argsort(dist)[:,:n_songs][0])

    reps_recs = []

    for idx in index:
        rec_id = str(list(tracks_dataset)[idx])
        reps_recs.append(rec_id)

    # ---- Using Features ----------------------
    centers = []
    for input in track_ids:
        song_idx = loc_dataset.index.get_loc(input)
        center = np.array([tracks_dataset.loc['duration'][input], tracks_dataset.loc['year'][input],
                          tracks_dataset.loc['key'][input], tracks_dataset.loc['valence'][input],
                          tracks_dataset.loc['arousal'][input], genre_dataset[song_idx]])
        centers.append(center)

    song_center = np.mean(np.array(centers), axis=0)


    scaler = feature_gmm_model.steps[0][1]
    scaled_data = scaler.transform(feature_dataset)
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))


    dist = cosine_similarity(scaled_song_center, scaled_data)

    index = list(np.argsort(dist)[:,:n_songs][0])

    feature_recs = []

    for idx in index:
        rec_id = str(list(tracks_dataset)[idx])
        feature_recs.append(rec_id)

    return reps_recs, feature_recs



init_track_data()

# -- Main method ----------------
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=int(os.environ.get('PORT', 8080)))
