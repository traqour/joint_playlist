from flask import Flask, request, url_for, session, redirect, render_template, make_response, Response, jsonify
from flask_session import Session
import spotipy
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
from spotipy.exceptions import SpotifyException
import pandas as pd
import numpy as np
import umap
from scipy.spatial import distance as dst
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from config import SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET


app = Flask(__name__, static_folder='static')

app.secret_key = 'falkdjflasdkncxzoi12'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './.flask_session/'
Session(app)

@app.route('/')
def login():
    cache_handler = spotipy.cache_handler.FlaskSessionCacheHandler(session)
    auth_manager = spotipy.oauth2.SpotifyOAuth(client_id = SPOTIPY_CLIENT_ID,
        client_secret=SPOTIPY_CLIENT_SECRET,
        scope='user-top-read playlist-modify-public',
        cache_handler=cache_handler,
        redirect_uri=url_for('login', _external=True),
        show_dialog=True)
    if request.args.get("code"):
        auth_manager.get_access_token(request.args.get("code"))
        return redirect('/')

    if not auth_manager.validate_token(cache_handler.get_cached_token()):
        auth_url = auth_manager.get_authorize_url()
        return render_template('login.html', auth_url=auth_url)

    return redirect('/getTracks')

@app.route('/logout')
def logout():
    session.clear()
    return "Succesfully Logged Out"

@app.route('/getTracks')
def get_all_tracks():
    cache_handler = spotipy.cache_handler.FlaskSessionCacheHandler(session)
    auth_manager = spotipy.oauth2.SpotifyOAuth(
        client_id = SPOTIPY_CLIENT_ID,
        client_secret=SPOTIPY_CLIENT_SECRET,
        scope='user-top-read playlist-modify-public',
        cache_handler=cache_handler,
        redirect_uri=url_for('login', _external=True),
        show_dialog=True)
    if not auth_manager.validate_token(cache_handler.get_cached_token()):
        print(cache_handler.get_cached_token())
        return redirect('/')
    sp = spotipy.Spotify(auth_manager=auth_manager)
    columns = ['song_id', 'name', 'artist', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'loudness', 'speechiness', 'valence', 'liveness', 'key', 'tempo', 'genres']

    #recent has same priority, match current music tastes, find new songs matching current taste
    short_score = [x for x in range(70,20,-1)]
    #medium term has lowest, in between is awkward I feel like, less important than long and short
    medium_score = [x for x in range(50,0,-1)]
    #long score highest priority, throwback jams that both like is great
    long_score = [x for x in range(60,10,-1)]

    #Loading and cleaning user data
    arr1 = [['4sVp0rtlHVtcy65TJCxZXk', '춤', ['Damdamgugu'], 0.136, 0.687, 0.331, 0.769, -9.283, 0.0304, 0.355, 0.106, 6, 128.125, []], ['7sZCAHP2duHwr5M5K7lHsb', 'Selfish', ['Slum Village', 'John Legend', 'Kanye West'], 0.221, 0.71, 0.597, 0, -6.468, 0.239, 0.49, 0.432, 7, 95.769, ['neo soul', 'urban contemporary', 'hardcore hip hop', 'detroit hip hop', 'pop soul', 'pop', 'alternative hip hop', 'chicago rap', 'rap', 'hip hop']], ['1YrU8ExqF04ygegVoOOoFU', '中庭の少女たち', ['SHISHAMO'], 0.00227, 0.468, 0.837, 0, -3.841, 0.0358, 0.854, 0.174, 4, 174.848, ['j-pop', 'j-rock']], ['7B4XVwdxXFJ3yzz2BtJPmf', 'Mean It', ['6LACK'], 0.0694, 0.759, 0.66, 0.0539, -7.375, 0.0432, 0.844, 0.0903, 11, 104.039, ['atl hip hop', 'melodic rap', 'r&b', 'rap', 'trap']], ['3qnCr193RUK7qIuAKwiryh', 'About U', ['AstroHertz'], 0.0364, 0.803, 0.868, 0.496, -7.977, 0.13, 0.626, 0.0743, 0, 124.997, []], ['2vVUevSgxbWBub5zW7rQPO', 'Homicide (with Jessie Reyez)', ['6LACK', 'Jessie Reyez'], 0.11, 0.765, 0.74, 0, -7.192, 0.178, 0.592, 0.0748, 1, 105.013, ['trap', 'atl hip hop', 'canadian contemporary r&b', 'canadian pop', 'melodic rap', 'r&b', 'rap']], ['6m07gYVINo4QNYowLB3hUW', 'Searching For Yourself (feat. Raveena)', ['Yeek', 'Raveena'], 0.39, 0.703, 0.528, 0.0304, -8.975, 0.0335, 0.531, 0.142, 9, 89.035, ['indie hip hop', 'alternative r&b', 'hyperpop', 'indie soul']], ['1VXWhDwV2WG7tnlclMP1nk', 'Leave With Me', ['Yeek'], 0.128, 0.741, 0.842, 0.0162, -5.858, 0.0342, 0.758, 0.0858, 0, 125.089, ['hyperpop', 'indie hip hop']], ['0neC0jkiRxf0UNsb7SkJOR', 'Pyramid', ['ALYSS'], 0.0447, 0.655, 0.758, 0, -9.779, 0.272, 0.414, 0.0971, 3, 114.082, ['uk alternative pop']], ['1SkOOsHVAZrXv3KXdmVQE6', 'Mood Swings', ['Yeek'], 0.107, 0.618, 0.6, 0.0307, -7.124, 0.0396, 0.318, 0.0984, 11, 154.073, ['hyperpop', 'indie hip hop']], ['43G7LV6InhWQWdQGyQQ6vJ', 'Video Games', ['frvnk'], 0.000167, 0.534, 0.568, 0.0582, -9.802, 0.0517, 0.296, 0.0741, 2, 126.877, []], ['787Y2idwCU2Rk60Prv4wpr', 'Saving Up', ['Dom Dolla'], 0.00353, 0.747, 0.822, 0.132, -6.373, 0.0721, 0.578, 0.165, 3, 129.994, ['australian house', 'deep groove house', 'house']], ['08vlprFBmwh9TQnjXUtZDG', 'FAMJAM4000', ['Jordan Ward'], 0.467, 0.869, 0.574, 0.339, -7.383, 0.0623, 0.725, 0.15, 11, 116.57, ['chill abstract hip hop', 'indie hip hop', 'indie r&b']], ['42VsgItocQwOQC3XWZ8JNA', 'FE!N (feat. Playboi Carti)', ['Travis Scott', 'Playboi Carti'], 0.0316, 0.569, 0.882, 0, -2.777, 0.06, 0.201, 0.142, 3, 148.038, ['atl hip hop', 'slap house', 'pluggnb', 'rap', 'plugg', 'hip hop', 'rage rap']], ['7a3LbQFgp7NCuNcGlTgSsN', '忘れられないの', ['sakanaction'], 0.35, 0.642, 0.645, 0.00349, -7.358, 0.0375, 0.917, 0.191, 6, 172.1, ['hokkaido indie', 'j-pop', 'j-rock', 'japanese electropop']], ['7rbECVPkY5UODxoOUVKZnA', 'I Wonder', ['Kanye West'], 0.141, 0.542, 0.466, 0.000445, -8.665, 0.0831, 0.124, 0.125, 0, 191.385, ['chicago rap', 'hip hop', 'rap']], ['6RqtEJNI6pEiVaAlPeyr0R', 'Patience (feat. Don Toliver)', ['Lil Uzi Vert', 'Don Toliver'], 0.0502, 0.645, 0.732, 1.16e-06, -2.94, 0.0306, 0.121, 0.161, 4, 106.963, ['trap', 'pop rap', 'melodic rap', 'rap', 'hip hop', 'rage rap', 'philly rap']], ['7qLr3HMApUbyDkUvgIvHnB', 'Oh U Went (feat. Drake)', ['Young Thug', 'Drake'], 0.0517, 0.808, 0.789, 0, -5.88, 0.236, 0.465, 0.0819, 7, 136.01, ['trap', 'atl hip hop', 'canadian hip hop', 'canadian pop', 'pop rap', 'gangster rap', 'melodic rap', 'atl trap', 'rap', 'hip hop']], ['3VvMqZYJD4pWjlYvjRnuhy', '甜蜜蜜', ['Teresa Teng'], 0.267, 0.638, 0.387, 0.00157, -9.775, 0.0328, 0.412, 0.199, 2, 127.289, ['c-pop', 'classic mandopop', 'kayokyoku']], ['6wpDQGn3Gl0j9Wt6D6mYvQ', 'O Descobridor Dos Sete Mares', ['Tim Maia'], 0.0108, 0.643, 0.78, 0.00069, -8.306, 0.0718, 0.963, 0.305, 10, 136.891, ['brazilian boogie', 'brazilian soul', 'mpb']], ['6wsqVwoiVH2kde4k4KKAFU', 'I KNOW ?', ['Travis Scott'], 0.0186, 0.927, 0.619, 0, -4.441, 0.0539, 0.817, 0.104, 5, 117.995, ['hip hop', 'rap', 'slap house']], ['0dAfw35k2hBsnbSl74AVJF', 'dashstar*', ['Knock2'], 0.00872, 0.699, 0.949, 0.328, -2.994, 0.0739, 0.186, 0.151, 9, 126.07, ['bassline']], ['2TjnCxxQRYn56Ye8gkUKiW', 'Desperado - 2013 Remaster', ['Eagles'], 0.946, 0.228, 0.224, 0.000222, -12.749, 0.0311, 0.18, 0.273, 7, 60.3, ['album rock', 'classic rock', 'heartland rock', 'mellow gold', 'rock', 'soft rock', 'yacht rock']], ['0WCJSIy75ZVxqY51ce0enc', 'Pluto to Mars', ['Lil Uzi Vert'], 0.0225, 0.807, 0.627, 0, -6.829, 0.0771, 0.761, 0.0904, 11, 122.963, ['hip hop', 'melodic rap', 'philly rap', 'rage rap', 'rap']], ['1N8klHEbZ0bFOHnMVs8C9S', 'Fell In Love', ['Marshmello', 'Brent Faiyaz'], 0.0899, 0.732, 0.464, 3.1e-05, -9.722, 0.0527, 0.647, 0.0997, 1, 149.954, ['brostep', 'progressive electro house', 'pop', 'r&b', 'rap', 'edm']], ['2dFqK2ZkYB9Xc47gr3xXWl', 'Replay', ['Tems'], 0.097, 0.613, 0.549, 0, -8.187, 0.231, 0.78, 0.116, 9, 123.879, ['afro r&b', 'alte', 'nigerian pop']], ['4Tla2jt77nO70DgGwFejbK', 'Run It Up', ['Snakehips', 'EARTHGANG'], 0.00621, 0.838, 0.793, 0, -5.176, 0.145, 0.527, 0.0962, 2, 120.062, ['underground hip hop', 'atl hip hop', 'uk dance', 'indie poptimism', 'electropop', 'rap', 'indie hip hop', 'psychedelic hip hop', 'hip hop']], ['4Z59a9hIn8EMsni3OiqGso', 'Think Fast (feat. Weezer)', ['Dominic Fike', 'Weezer'], 0.368, 0.538, 0.452, 2.26e-05, -7.994, 0.0462, 0.209, 0.0635, 2, 168.102, ['permanent wave', 'alternative pop rock', 'modern rock', 'modern power pop', 'alternative rock', 'rock', 'pov: indie']], ['7fgmo2cHGzWLexkRhBeECj', 'WINDY SUMMER', ['Anri'], 0.175, 0.653, 0.762, 0.00113, -5.042, 0.0311, 0.838, 0.174, 2, 106.951, ['classic city pop']], ['1aa4wUXY3Bkc1uAYSi1W2z', 'WY@', ['Brent Faiyaz'], 0.596, 0.809, 0.338, 0.000111, -8.031, 0.189, 0.287, 0.122, 7, 119.934, ['r&b', 'rap']], ['2aQpISWUBToaF84DDiTeRV', 'Be My Lover (feat. La Bouche) - 2023 Mix', ['Hypaton', 'David Guetta', 'La Bouche'], 0.0347, 0.589, 0.973, 0.000945, -5.317, 0.0426, 0.115, 0.45, 8, 126.003, ['europop', 'diva house', 'dance pop', 'pop', 'german techno', 'eurodance', 'pop dance', 'big room', 'edm']], ['2ydwqnjPOcM24MV0kLThwi', 'Amped', ['Lil Uzi Vert'], 0.00038, 0.534, 0.699, 8.21e-06, -5.078, 0.0326, 0.0555, 0.362, 7, 147.169, ['hip hop', 'melodic rap', 'philly rap', 'rage rap', 'rap']], ['44KWbTVZev3SWdv1t5UoYE', 'How Much Is Weed?', ['Dominic Fike'], 0.201, 0.691, 0.813, 0.132, -4.49, 0.0393, 0.537, 0.128, 1, 154.122, ['alternative pop rock', 'pov: indie']], ['4DUmRDbkGK8eSCbnbrcpXo', 'Now I Know', ['Kenichiro Nishihara', 'Pismo'], 0.281, 0.681, 0.731, 0.315, -7.902, 0.0388, 0.898, 0.145, 6, 90.002, ['ambeat']], ['4oY2T9ur7Ll5b2kpBlcWcb', 'You Rock My World - Radio Edit', ['Michael Jackson'], 0.0297, 0.819, 0.774, 0.000626, -2.53, 0.101, 0.947, 0.0896, 4, 95.004, ['r&b', 'soul']], ['6GnhWMhgJb7uyiiPEiEkDA', 'Weekend (feat. Miguel)', ['Mac Miller', 'Miguel'], 0.47, 0.843, 0.435, 0, -8.442, 0.178, 0.19, 0.147, 8, 120.058, ['urban contemporary', 'pittsburgh rap', 'pop rap', 'r&b', 'rap', 'hip hop']], ['0Y71FEcRkyZOh4hySnEGB5', 'Baggage', ['Breakfast Santana', 'Khaji Beats'], 0.404, 0.629, 0.717, 3.33e-06, -8.505, 0.243, 0.655, 0.238, 0, 117.981, []], ['0bz5Ud9FSTJSm95ZNgvskj', 'Guilty Luv - Kenichiro Nishihara Remix', ['Kenichiro Nishihara'], 0.435, 0.815, 0.666, 0.000551, -6.026, 0.193, 0.842, 0.114, 8, 97.0, ['ambeat']], ['0cJTBlSiodwtdA5vdIbHhB', 'Pretend Lovers', ['Montell Fish'], 0.925, 0.647, 0.387, 0.0288, -7.173, 0.25, 0.397, 0.104, 8, 120.039, ['pittsburgh rap']], ['0dn6S6lJqAIQg90kMmWtVj', 'NO SZNS', ['Jean Dawson', 'SZA'], 0.647, 0.55, 0.34, 0, -7.988, 0.0311, 0.195, 0.121, 5, 112.077, ['rap', 'pop', 'r&b', 'modern indie pop', 'indie hip hop']], ['0n1VrelcxEj2QzOvAFTnST', 'Favorite Jeans', ['Free Party'], 0.104, 0.631, 0.685, 0.0146, -6.769, 0.0303, 0.503, 0.141, 5, 75.0, ['indie hip hop']], ['0rVJ6v23RQozOIvr1YotJP', 'Hell N Back', ['Bakar'], 0.314, 0.633, 0.684, 8.95e-05, -4.314, 0.591, 0.725, 0.112, 10, 209.688, ['uk alternative hip hop']], ['0wOtc2nY3NOohp4xSwOyTN', 'We Might Even Be Falling In Love (Duet) - Spotify Singles', ['Victoria Monét', 'Bryson Tiller'], 0.473, 0.731, 0.423, 0.000413, -10.147, 0.0784, 0.78, 0.129, 6, 76.964, ['rap', 'kentucky hip hop', 'alternative r&b', 'r&b']], ['1yIi7qRdybE4yY1V0YdOKG', 'Find My Way Home', ['Sammy Virji'], 0.116, 0.726, 0.728, 5.66e-05, -7.381, 0.259, 0.504, 0.061, 6, 136.09, ['bass house', 'bassline', 'old school bassline']], ['22lVCoZxqXlr5PzAU8onhA', 'Say You Love Me - 2017 Remix', ['Kenichiro Nishihara', 'Tamala'], 0.195, 0.64, 0.887, 0.0204, -5.203, 0.0296, 0.733, 0.251, 0, 147.98, ['ambeat']], ['277xBb6mGl8yNEY3tjntiP', 'Thru Your Mind', ['Bassboy'], 0.0905, 0.897, 0.827, 0.036, -5.465, 0.165, 0.529, 0.112, 11, 130.004, ['bass house', 'bassline', 'birmingham grime', 'old school bassline']], ['2KBo6O5rkNdtYT3wYjkEkq', 'Fall In Love', ['Slum Village'], 0.0486, 0.796, 0.543, 0.00584, -7.994, 0.164, 0.784, 0.124, 1, 91.212, ['alternative hip hop', 'detroit hip hop', 'hardcore hip hop', 'hip hop']], ['2vdposMtmLoanI0pflut4F', 'Circus', ['Summer Walker'], 0.638, 0.506, 0.601, 0, -8.647, 0.0975, 0.648, 0.129, 0, 74.111, ['r&b', 'rap']], ['3RaCGXCiiMufRPoexXxGkV', 'Slime You Out (feat. SZA)', ['Drake', 'SZA'], 0.508, 0.483, 0.408, 0, -9.243, 0.0502, 0.105, 0.259, 5, 88.88, ['canadian hip hop', 'pop rap', 'canadian pop', 'pop', 'r&b', 'rap', 'hip hop']], ['3VFGA65JnN8EDmnRb5SHW2', 'Zoom (Bonus Track)', ['Lil Uzi Vert'], 0.0784, 0.818, 0.547, 1.3e-06, -5.868, 0.39, 0.432, 0.119, 5, 79.483, ['hip hop', 'melodic rap', 'philly rap', 'rage rap', 'rap']]]
    user1_short = to_dataframe(arr1, columns, short_score)

    arr2 = [['0bz5Ud9FSTJSm95ZNgvskj', 'Guilty Luv - Kenichiro Nishihara Remix', ['Kenichiro Nishihara'], 0.435, 0.815, 0.666, 0.000551, -6.026, 0.193, 0.842, 0.114, 8, 97.0, ['ambeat']], ['4DUmRDbkGK8eSCbnbrcpXo', 'Now I Know', ['Kenichiro Nishihara', 'Pismo'], 0.281, 0.681, 0.731, 0.315, -7.902, 0.0388, 0.898, 0.145, 6, 90.002, ['ambeat']], ['1VXWhDwV2WG7tnlclMP1nk', 'Leave With Me', ['Yeek'], 0.128, 0.741, 0.842, 0.0162, -5.858, 0.0342, 0.758, 0.0858, 0, 125.089, ['hyperpop', 'indie hip hop']], ['44KWbTVZev3SWdv1t5UoYE', 'How Much Is Weed?', ['Dominic Fike'], 0.201, 0.691, 0.813, 0.132, -4.49, 0.0393, 0.537, 0.128, 1, 154.122, ['alternative pop rock', 'pov: indie']], ['1zZdEavQr1Vl769ZMqYUvk', 'Up All Night', ['Kenichiro Nishihara', 'SIRUP'], 0.322, 0.649, 0.8, 0.0147, -5.786, 0.0283, 0.561, 0.126, 4, 111.982, ['japanese r&b', 'japanese soul', 'ambeat']], ['4sVp0rtlHVtcy65TJCxZXk', '춤', ['Damdamgugu'], 0.136, 0.687, 0.331, 0.769, -9.283, 0.0304, 0.355, 0.106, 6, 128.125, []], ['1N8klHEbZ0bFOHnMVs8C9S', 'Fell In Love', ['Marshmello', 'Brent Faiyaz'], 0.0899, 0.732, 0.464, 3.1e-05, -9.722, 0.0527, 0.647, 0.0997, 1, 149.954, ['brostep', 'progressive electro house', 'pop', 'r&b', 'rap', 'edm']], ['0WCJSIy75ZVxqY51ce0enc', 'Pluto to Mars', ['Lil Uzi Vert'], 0.0225, 0.807, 0.627, 0, -6.829, 0.0771, 0.761, 0.0904, 11, 122.963, ['hip hop', 'melodic rap', 'philly rap', 'rage rap', 'rap']], ['7qLr3HMApUbyDkUvgIvHnB', 'Oh U Went (feat. Drake)', ['Young Thug', 'Drake'], 0.0517, 0.808, 0.789, 0, -5.88, 0.236, 0.465, 0.0819, 7, 136.01, ['trap', 'atl hip hop', 'canadian hip hop', 'canadian pop', 'pop rap', 'gangster rap', 'melodic rap', 'atl trap', 'rap', 'hip hop']], ['22lVCoZxqXlr5PzAU8onhA', 'Say You Love Me - 2017 Remix', ['Kenichiro Nishihara', 'Tamala'], 0.195, 0.64, 0.887, 0.0204, -5.203, 0.0296, 0.733, 0.251, 0, 147.98, ['ambeat']], ['0n1VrelcxEj2QzOvAFTnST', 'Favorite Jeans', ['Free Party'], 0.104, 0.631, 0.685, 0.0146, -6.769, 0.0303, 0.503, 0.141, 5, 75.0, ['indie hip hop']], ['3qnCr193RUK7qIuAKwiryh', 'About U', ['AstroHertz'], 0.0364, 0.803, 0.868, 0.496, -7.977, 0.13, 0.626, 0.0743, 0, 124.997, []], ['7sZCAHP2duHwr5M5K7lHsb', 'Selfish', ['Slum Village', 'John Legend', 'Kanye West'], 0.221, 0.71, 0.597, 0, -6.468, 0.239, 0.49, 0.432, 7, 95.769, ['neo soul', 'urban contemporary', 'hardcore hip hop', 'detroit hip hop', 'pop soul', 'pop', 'alternative hip hop', 'chicago rap', 'rap', 'hip hop']], ['2aQpISWUBToaF84DDiTeRV', 'Be My Lover (feat. La Bouche) - 2023 Mix', ['Hypaton', 'David Guetta', 'La Bouche'], 0.0347, 0.589, 0.973, 0.000945, -5.317, 0.0426, 0.115, 0.45, 8, 126.003, ['europop', 'diva house', 'dance pop', 'pop', 'german techno', 'eurodance', 'pop dance', 'big room', 'edm']], ['3CblJq8QQQ0bb7vwJu8c3v', '4EVA (feat. Pharrell Williams)', ['KAYTRAMINÉ', 'Aminé', 'KAYTRANADA', 'Pharrell Williams'], 0.0248, 0.83, 0.695, 0.0755, -9.445, 0.0716, 0.536, 0.0573, 6, 112.046, ['escape room', 'alternative r&b', 'underground hip hop', 'portland hip hop', 'lgbtq+ hip hop', 'dance pop', 'pop rap', 'indie soul', 'pop', 'rap']], ['2dFqK2ZkYB9Xc47gr3xXWl', 'Replay', ['Tems'], 0.097, 0.613, 0.549, 0, -8.187, 0.231, 0.78, 0.116, 9, 123.879, ['afro r&b', 'alte', 'nigerian pop']], ['1SkOOsHVAZrXv3KXdmVQE6', 'Mood Swings', ['Yeek'], 0.107, 0.618, 0.6, 0.0307, -7.124, 0.0396, 0.318, 0.0984, 11, 154.073, ['hyperpop', 'indie hip hop']], ['4OYXAD2OSy0RkSsQ0D9BEQ', 'Heartless', ['Kenichiro Nishihara', 'Michael Kaneko'], 0.599, 0.797, 0.79, 0.00271, -5.086, 0.0387, 0.824, 0.114, 5, 103.966, ['japanese r&b', 'ambeat']], ['6wsqVwoiVH2kde4k4KKAFU', 'I KNOW ?', ['Travis Scott'], 0.0186, 0.927, 0.619, 0, -4.441, 0.0539, 0.817, 0.104, 5, 117.995, ['hip hop', 'rap', 'slap house']], ['7g6FlLHxbVqMi1s7S8tDTp', 'Summer Too Hot', ['Chris Brown'], 0.155, 0.681, 0.71, 3.02e-05, -6.643, 0.0574, 0.254, 0.367, 11, 89.957, ['pop rap', 'r&b', 'rap']], ['39sDitIeCMrVX2QyXHY46t', 'Blue Hair', ['TV Girl'], 0.554, 0.751, 0.72, 0.0497, -6.376, 0.0303, 0.884, 0.258, 4, 135.73, ['pov: indie']], ['4Z59a9hIn8EMsni3OiqGso', 'Think Fast (feat. Weezer)', ['Dominic Fike', 'Weezer'], 0.368, 0.538, 0.452, 2.26e-05, -7.994, 0.0462, 0.209, 0.0635, 2, 168.102, ['permanent wave', 'alternative pop rock', 'modern rock', 'modern power pop', 'alternative rock', 'rock', 'pov: indie']], ['08vlprFBmwh9TQnjXUtZDG', 'FAMJAM4000', ['Jordan Ward'], 0.467, 0.869, 0.574, 0.339, -7.383, 0.0623, 0.725, 0.15, 11, 116.57, ['chill abstract hip hop', 'indie hip hop', 'indie r&b']], ['5mfGEs5n647v4WE3K5YmBy', 'on & on', ['piri', 'Tommy Villiers', 'piri & tommy'], 0.143, 0.682, 0.762, 0.0195, -7.622, 0.0526, 0.797, 0.0807, 4, 87.032, []], ['6RqtEJNI6pEiVaAlPeyr0R', 'Patience (feat. Don Toliver)', ['Lil Uzi Vert', 'Don Toliver'], 0.0502, 0.645, 0.732, 1.16e-06, -2.94, 0.0306, 0.121, 0.161, 4, 106.963, ['trap', 'pop rap', 'melodic rap', 'rap', 'hip hop', 'rage rap', 'philly rap']], ['3lsiqFV6SKhBgzQCpuM1JR', 'Miracle (with Ellie Goulding) - Mau P Remix', ['Calvin Harris', 'Ellie Goulding', 'Mau P'], 0.00134, 0.642, 0.943, 0.085, -6.87, 0.0519, 0.44, 0.155, 8, 128.005, ['uk pop', 'electro house', 'dance pop', 'indietronica', 'progressive house', 'metropopolis', 'pop', 'uk dance', 'tech house', 'house', 'pop dance', 'edm']], ['0h3BFerMyzUaYYh2nwGiX3', 'SIDEKICK (with Joyce Wrice) - BONUS', ['Jordan Ward', 'Joyce Wrice'], 0.136, 0.781, 0.623, 0.012, -6.064, 0.0515, 0.726, 0.141, 10, 107.732, ['indie hip hop', 'alternative r&b', 'indie r&b', 'chill abstract hip hop']], ['5cFREA8Fg75ytjbOC1NSOx', 'Pray', ['Kenichiro Nishihara', 'MARTER'], 0.0201, 0.616, 0.782, 0.0227, -6.301, 0.0318, 0.899, 0.0795, 0, 193.992, ['ambeat']], ['3lcfs9Qjfxzy00VSVkixDv', 'I Gotta', ['Lil Uzi Vert'], 0.209, 0.818, 0.51, 0, -7.05, 0.408, 0.416, 0.211, 2, 129.057, ['hip hop', 'melodic rap', 'philly rap', 'rage rap', 'rap']], ['7rU6Iebxzlvqy5t857bKFq', 'Plastic Love', ['Mariya Takeuchi'], 0.0747, 0.648, 0.875, 9.64e-05, -5.141, 0.034, 0.857, 0.196, 2, 102.517, ['classic city pop', 'idol kayo', 'japanese singer-songwriter']], ['7a3LbQFgp7NCuNcGlTgSsN', '忘れられないの', ['sakanaction'], 0.35, 0.642, 0.645, 0.00349, -7.358, 0.0375, 0.917, 0.191, 6, 172.1, ['hokkaido indie', 'j-pop', 'j-rock', 'japanese electropop']], ['4ZwgD4frTwoDdOUsjyoqAJ', 'Dancing In The Courthouse', ['Dominic Fike'], 0.187, 0.621, 0.741, 0, -4.479, 0.0611, 0.691, 0.227, 2, 76.464, ['alternative pop rock', 'pov: indie']], ['6wpDQGn3Gl0j9Wt6D6mYvQ', 'O Descobridor Dos Sete Mares', ['Tim Maia'], 0.0108, 0.643, 0.78, 0.00069, -8.306, 0.0718, 0.963, 0.305, 10, 136.891, ['brazilian boogie', 'brazilian soul', 'mpb']], ['0neC0jkiRxf0UNsb7SkJOR', 'Pyramid', ['ALYSS'], 0.0447, 0.655, 0.758, 0, -9.779, 0.272, 0.414, 0.0971, 3, 114.082, ['uk alternative pop']], ['0wOtc2nY3NOohp4xSwOyTN', 'We Might Even Be Falling In Love (Duet) - Spotify Singles', ['Victoria Monét', 'Bryson Tiller'], 0.473, 0.731, 0.423, 0.000413, -10.147, 0.0784, 0.78, 0.129, 6, 76.964, ['rap', 'kentucky hip hop', 'alternative r&b', 'r&b']], ['7fgmo2cHGzWLexkRhBeECj', 'WINDY SUMMER', ['Anri'], 0.175, 0.653, 0.762, 0.00113, -5.042, 0.0311, 0.838, 0.174, 2, 106.951, ['classic city pop']], ['1YrU8ExqF04ygegVoOOoFU', '中庭の少女たち', ['SHISHAMO'], 0.00227, 0.468, 0.837, 0, -3.841, 0.0358, 0.854, 0.174, 4, 174.848, ['j-pop', 'j-rock']], ['7B4XVwdxXFJ3yzz2BtJPmf', 'Mean It', ['6LACK'], 0.0694, 0.759, 0.66, 0.0539, -7.375, 0.0432, 0.844, 0.0903, 11, 104.039, ['atl hip hop', 'melodic rap', 'r&b', 'rap', 'trap']], ['3VFGA65JnN8EDmnRb5SHW2', 'Zoom (Bonus Track)', ['Lil Uzi Vert'], 0.0784, 0.818, 0.547, 1.3e-06, -5.868, 0.39, 0.432, 0.119, 5, 79.483, ['hip hop', 'melodic rap', 'philly rap', 'rage rap', 'rap']], ['6m07gYVINo4QNYowLB3hUW', 'Searching For Yourself (feat. Raveena)', ['Yeek', 'Raveena'], 0.39, 0.703, 0.528, 0.0304, -8.975, 0.0335, 0.531, 0.142, 9, 89.035, ['indie hip hop', 'alternative r&b', 'hyperpop', 'indie soul']], ['4UeWKazLR1ZwwSVnLw9Ir9', '踊り子', ['Vaundy'], 0.845, 0.718, 0.475, 0.801, -11.469, 0.0611, 0.678, 0.105, 7, 157.032, ['j-pop', 'japanese soul']], ['1dHiSGzb9WFtDKnBFJs4KO', 'Just Say', ['Coco & Breezy', 'Tara Carosielli'], 0.0465, 0.83, 0.419, 0.0145, -9.76, 0.063, 0.511, 0.0815, 5, 119.993, ['indie electropop', 'soulful house']], ['2ydwqnjPOcM24MV0kLThwi', 'Amped', ['Lil Uzi Vert'], 0.00038, 0.534, 0.699, 8.21e-06, -5.078, 0.0326, 0.0555, 0.362, 7, 147.169, ['hip hop', 'melodic rap', 'philly rap', 'rage rap', 'rap']], ['5OCojRyXbEUifhcSkKBuiD', 'Eclipses', ['HIRA'], 0.0503, 0.943, 0.235, 0, -8.462, 0.173, 0.617, 0.159, 10, 125.925, []], ['19mXqYcLHY716cN53T1d1E', 'Wish That You Were Mine', ['The Manhattans'], 0.555, 0.621, 0.534, 1.29e-05, -10.09, 0.036, 0.574, 0.297, 9, 119.615, ['classic soul', 'disco', 'funk', 'motown', 'philly soul', 'quiet storm', 'soul']], ['42VsgItocQwOQC3XWZ8JNA', 'FE!N (feat. Playboi Carti)', ['Travis Scott', 'Playboi Carti'], 0.0316, 0.569, 0.882, 0, -2.777, 0.06, 0.201, 0.142, 3, 148.038, ['atl hip hop', 'slap house', 'pluggnb', 'rap', 'plugg', 'hip hop', 'rage rap']], ['3fDTzkvrOo5xQIO480Qmsb', 'Suicide Doors', ['Lil Uzi Vert'], 0.0113, 0.538, 0.783, 0, -2.565, 0.222, 0.553, 0.281, 11, 75.388, ['hip hop', 'melodic rap', 'philly rap', 'rage rap', 'rap']], ['59MY06cY0nvWUApyWcTYGB', 'supїdo', ['фрози'], 0.00917, 0.702, 0.755, 0.756, -8.435, 0.0436, 0.216, 0.219, 10, 134.962, []], ['2vVUevSgxbWBub5zW7rQPO', 'Homicide (with Jessie Reyez)', ['6LACK', 'Jessie Reyez'], 0.11, 0.765, 0.74, 0, -7.192, 0.178, 0.592, 0.0748, 1, 105.013, ['trap', 'atl hip hop', 'canadian contemporary r&b', 'canadian pop', 'melodic rap', 'r&b', 'rap']], ['5Lv5L45PQmp5CTjs5PlQ6e', 'Silk & Cologne (EI8HT & Offset) - Spider-Verse Remix', ['EI8HT', 'Offset'], 0.268, 0.775, 0.815, 0.00479, -7.607, 0.0974, 0.847, 0.111, 4, 110.007, ['trap', 'atl hip hop', 'pop rap', 'rap', 'hip hop']]]
    user1_medium = to_dataframe(arr2, columns, medium_score)

    arr3 = [['3Xsmpypmc2DcxQBbmnnrB5', '3000 Miles (Baby Baby)', ['Yeek'], 0.00675, 0.643, 0.638, 1.78e-05, -7.753, 0.0424, 0.959, 0.132, 10, 77.505, ['hyperpop', 'indie hip hop']], ['6u3CPnFMKANYgfdiifFOiJ', 'GRAVITY (FEAT. TYLER, THE CREATOR)', ['Brent Faiyaz', 'Dahi', 'Tyler, The Creator'], 0.173, 0.539, 0.615, 0.0056, -8.746, 0.252, 0.493, 0.144, 1, 163.924, ['rap', 'r&b', 'hip hop']], ['0wOtc2nY3NOohp4xSwOyTN', 'We Might Even Be Falling In Love (Duet) - Spotify Singles', ['Victoria Monét', 'Bryson Tiller'], 0.473, 0.731, 0.423, 0.000413, -10.147, 0.0784, 0.78, 0.129, 6, 76.964, ['rap', 'kentucky hip hop', 'alternative r&b', 'r&b']], ['7arX7t70jcRTL4iSYudFJn', 'Give It To Me', ['Pink Sweat$'], 0.109, 0.787, 0.592, 0.000219, -6.01, 0.0453, 0.642, 0.095, 0, 100.02, ['bedroom soul']], ['7eWGnKg4B44sbBPpQp4y2c', 'Dragonball Durag', ['Thundercat'], 0.697, 0.648, 0.59, 0.808, -9.664, 0.0942, 0.401, 0.111, 2, 81.045, ['afrofuturism', 'escape room', 'indie soul']], ['0crw01bqnefvDUDjsuKraD', 'After Last Night (with Thundercat & Bootsy Collins)', ['Bruno Mars', 'Anderson .Paak', 'Silk Sonic', 'Thundercat', 'Bootsy Collins'], 0.0297, 0.651, 0.703, 0, -8.958, 0.0816, 0.647, 0.0608, 0, 140.051, ['neo soul', 'escape room', 'funk', 'soul', 'p funk', 'dance pop', 'afrofuturism', 'pop', 'indie soul', 'hip hop', 'quiet storm']], ['1c0hsvHLELX6y8qymnpLKL', 'Soltera (Remix)', ['Lunay', 'Daddy Yankee', 'Bad Bunny'], 0.361, 0.795, 0.783, 0, -4.271, 0.0432, 0.799, 0.437, 5, 92.01, ['latin hip hop', 'trap latino', 'urbano latino', 'reggaeton', 'reggaeton flow']], ['1NeLwFETswx8Fzxl2AFl91', 'Something About Us', ['Daft Punk'], 0.44, 0.875, 0.475, 0.72, -12.673, 0.0986, 0.384, 0.046, 9, 99.958, ['electro', 'filter house', 'rock']], ['2gq9iG0maBxkuZI7yfGJuv', 'Overtime', ['Bryson Tiller'], 0.254, 0.657, 0.497, 0, -7.689, 0.112, 0.593, 0.367, 1, 106.049, ['kentucky hip hop', 'r&b', 'rap']], ['0bz5Ud9FSTJSm95ZNgvskj', 'Guilty Luv - Kenichiro Nishihara Remix', ['Kenichiro Nishihara'], 0.435, 0.815, 0.666, 0.000551, -6.026, 0.193, 0.842, 0.114, 8, 97.0, ['ambeat']], ['3Q4gttWQ6hxqWOa3tHoTNi', 'Not You Too (feat. Chris Brown)', ['Drake', 'Chris Brown'], 0.342, 0.458, 0.452, 1.94e-05, -9.299, 0.047, 0.316, 0.0703, 9, 86.318, ['canadian hip hop', 'pop rap', 'canadian pop', 'r&b', 'rap', 'hip hop']], ['435yU2MvEGfDdmbH0noWZ0', 'worldstar money (interlude)', ['Joji'], 0.964, 0.577, 0.387, 0.705, -8.607, 0.274, 0.459, 0.208, 7, 146.565, ['viral pop']], ['0tdA3tsJ4n6rJuiId3KrOP', 'cz', ['Mk.gee'], 0.598, 0.734, 0.552, 0.835, -7.595, 0.123, 0.714, 0.107, 2, 91.064, ['experimental r&b']], ['7nc7mlSdWYeFom84zZ8Wr8', 'Tell Em', ['Cochise', '$NOT'], 0.103, 0.672, 0.717, 0, -7.476, 0.226, 0.473, 0.398, 5, 157.905, ['underground hip hop', 'florida rap', 'pluggnb', 'aesthetic rap', 'cloud rap', 'plugg', 'rage rap']], ['6ptijxei8gTrFuIob3LyJW', 'whiskey', ['Julian Skiboat'], 0.576, 0.921, 0.478, 1.72e-05, -7.467, 0.037, 0.692, 0.306, 11, 98.961, []], ['3dPtXHP0oXQ4HCWHsOA9js', '夜に駆ける', ['YOASOBI'], 0.00231, 0.67, 0.874, 1.72e-05, -5.221, 0.0305, 0.789, 0.3, 8, 130.041, ['j-pop', 'japanese teen pop']], ['5IUtvfNvOyVYZUa6AJFrnP', 'Spicy (feat. Post Malone)', ['Ty Dolla $ign', 'Post Malone'], 0.143, 0.782, 0.51, 0, -5.724, 0.0419, 0.118, 0.115, 4, 99.993, ['trap', 'southern hip hop', 'pop rap', 'melodic rap', 'pop', 'dfw rap', 'r&b', 'rap', 'trap soul', 'hip hop']], ['6M4jdLdM7wLGungMV9gsCS', 'Smokin Out The Window', ['Bruno Mars', 'Anderson .Paak', 'Silk Sonic'], 0.0558, 0.627, 0.618, 0, -8.529, 0.0437, 0.848, 0.351, 2, 82.03, ['neo soul', 'escape room', 'dance pop', 'pop', 'indie soul', 'hip hop']], ['3siyfhqP2BSRciLSbwGpzR', 'Whats For Dinner?', ['Dominic Fike'], 0.65, 0.855, 0.469, 2.48e-05, -4.965, 0.0769, 0.305, 0.108, 4, 91.985, ['alternative pop rock', 'pov: indie']], ['3CblJq8QQQ0bb7vwJu8c3v', '4EVA (feat. Pharrell Williams)', ['KAYTRAMINÉ', 'Aminé', 'KAYTRANADA', 'Pharrell Williams'], 0.0248, 0.83, 0.695, 0.0755, -9.445, 0.0716, 0.536, 0.0573, 6, 112.046, ['escape room', 'alternative r&b', 'underground hip hop', 'portland hip hop', 'lgbtq+ hip hop', 'dance pop', 'pop rap', 'indie soul', 'pop', 'rap']], ['6j4j6iRR0Ema531o5Yxr2T', 'Memories', ['Eden Prince', 'Nonô'], 0.161, 0.909, 0.651, 0.0437, -5.672, 0.105, 0.895, 0.0923, 11, 124.026, ['dance pop', 'uk dance', 'deep groove house']], ['4bbJw6usZSqkcmnqjSIOWx', 'Put On A Smile', ['Bruno Mars', 'Anderson .Paak', 'Silk Sonic'], 0.0471, 0.548, 0.627, 0, -8.949, 0.0409, 0.493, 0.285, 9, 143.994, ['neo soul', 'escape room', 'dance pop', 'pop', 'indie soul', 'hip hop']], ['0Yz3F0UGDibDe8uU69zmjn', 'And July', ['HEIZE', 'DEAN', 'dj friz'], 0.0401, 0.734, 0.852, 0, -3.85, 0.0916, 0.787, 0.196, 2, 98.504, ['korean r&b', 'k-pop', 'korean pop']], ['34D6FJysnQioVingDKufuf', 'N 2 Deep', ['Drake', 'Future'], 0.0518, 0.507, 0.429, 0, -7.726, 0.326, 0.0744, 0.109, 2, 91.0, ['trap', 'atl hip hop', 'southern hip hop', 'canadian hip hop', 'pop rap', 'canadian pop', 'rap', 'hip hop']], ['0lTH3Dt0GlBQhxFHSnaZ7g', 'ROCKSTAR P', ['Baby Keem'], 0.152, 0.948, 0.573, 0, -8.162, 0.113, 0.4, 0.0833, 11, 100.017, ['hip hop', 'rap']], ['4NczzeHBQPPDO0B9AAmB8d', 'Assumptions', ['Sam Gellaitry'], 0.123, 0.639, 0.815, 0.00113, -4.718, 0.221, 0.443, 0.193, 11, 126.055, ['future bass', 'scottish electronic', 'vapor twitch']], ['6OTKVgVpVaVjhRLYizPJKA', 'Just for Me', ['PinkPantheress'], 0.653, 0.796, 0.625, 0.00348, -7.218, 0.0645, 0.641, 0.187, 0, 133.035, ['bedroom pop']], ['7ACT6YaXbYvl7hRWEOOEHQ', 'Double Negative (Skeleton Milkshake)', ['Dominic Fike'], 0.00383, 0.689, 0.858, 0, -2.868, 0.161, 0.643, 0.129, 4, 153.977, ['alternative pop rock', 'pov: indie']], ['3a3dQOO19moXPeTt2PomoT', 'What You Heard', ['Sonder'], 0.546, 0.429, 0.371, 5.71e-06, -9.017, 0.0327, 0.135, 0.104, 2, 137.134, ['experimental r&b']], ['1dIWPXMX4kRHj6Dt2DStUQ', 'Chosen (feat. Ty Dolla $ign)', ['Blxst', 'Tyga', 'Ty Dolla $ign'], 0.336, 0.571, 0.767, 0, -5.16, 0.287, 0.605, 0.0809, 2, 93.421, ['trap', 'southern hip hop', 'pop rap', 'r&b', 'rap', 'trap soul', 'hip hop', 'westcoast flow']], ['1S4sxNGmCASkyRI08YhLtT', "Gettin' Old", ['6LACK'], 0.4, 0.743, 0.4, 0.000311, -12.369, 0.0625, 0.229, 0.111, 11, 80.01, ['atl hip hop', 'melodic rap', 'r&b', 'rap', 'trap']], ['6DXRUAHQTwJuCXuhXyRU53', 'peas', ['boylife'], 0.303, 0.249, 0.342, 0.0113, -10.283, 0.0389, 0.126, 0.16, 4, 64.335, ['modern indie pop']], ['6VFKuuqSbA1GpMEosUgTwQ', 'Nobody But You', ['Sonder', 'Jorja Smith'], 0.742, 0.637, 0.299, 0, -11.314, 0.0366, 0.364, 0.359, 7, 83.872, ['uk contemporary r&b', 'r&b', 'experimental r&b']], ['65FftemJ1DbbZ45DUfHJXE', 'OMG', ['NewJeans'], 0.357, 0.804, 0.771, 3.07e-06, -4.067, 0.0433, 0.739, 0.108, 9, 126.956, ['k-pop', 'k-pop girl group']], ['0b6l1obmzq1YrqiKBVLtIo', 'FRONTAL LOBE MUZIK', ['Daniel Caesar', 'Pharrell Williams'], 0.439, 0.681, 0.568, 0.000173, -6.264, 0.136, 0.301, 0.446, 11, 140.004, ['dance pop', 'pop', 'canadian contemporary r&b']], ['4DUmRDbkGK8eSCbnbrcpXo', 'Now I Know', ['Kenichiro Nishihara', 'Pismo'], 0.281, 0.681, 0.731, 0.315, -7.902, 0.0388, 0.898, 0.145, 6, 90.002, ['ambeat']], ['2sXf2JdbB2GlNju00kw9WE', 'Skate', ['Bruno Mars', 'Anderson .Paak', 'Silk Sonic'], 0.037, 0.708, 0.598, 0, -8.365, 0.0291, 0.698, 0.17, 5, 112.027, ['neo soul', 'escape room', 'dance pop', 'pop', 'indie soul', 'hip hop']], ['74fQgHleHZ4V7Gm1XYcGkj', 'One Night Only', ['Sonder'], 0.263, 0.874, 0.402, 0.00094, -5.786, 0.0691, 0.764, 0.147, 4, 119.99, ['experimental r&b']], ['06Xh1KvQofFghlSt33mFjc', 'Wishful Thinking', ['BENEE'], 0.144, 0.644, 0.661, 1.54e-06, -6.161, 0.0332, 0.55, 0.177, 4, 82.022, ['alt z', 'nz pop']], ['65j65yDSE06CNKAPjsHoJt', 'Was It Something I Said', ['MyKey'], 0.618, 0.587, 0.724, 0.000166, -6.775, 0.0321, 0.837, 0.157, 6, 106.994, ['bedroom pop']], ['5bW37bLG2ILFlcLYRK8AU6', 'dio', ['boylife'], 0.0997, 0.814, 0.512, 0.0119, -5.698, 0.034, 0.42, 0.0903, 9, 114.006, ['modern indie pop']], ['7CIERzyqIwLVKGp00YbHRO', 'Pain', ['PinkPantheress'], 0.227, 0.829, 0.617, 0.00306, -8.497, 0.14, 0.81, 0.0618, 1, 125.605, ['bedroom pop']], ['4Tla2jt77nO70DgGwFejbK', 'Run It Up', ['Snakehips', 'EARTHGANG'], 0.00621, 0.838, 0.793, 0, -5.176, 0.145, 0.527, 0.0962, 2, 120.062, ['underground hip hop', 'atl hip hop', 'uk dance', 'indie poptimism', 'electropop', 'rap', 'indie hip hop', 'psychedelic hip hop', 'hip hop']], ['47EiUVwUp4C9fGccaPuUCS', 'DÁKITI', ['Bad Bunny', 'Jhayco'], 0.401, 0.731, 0.573, 5.22e-05, -10.059, 0.0544, 0.145, 0.113, 4, 109.928, ['urbano latino', 'trap latino', 'reggaeton']], ['2bSk87AVkCIIC3Bcligq1z', 'Life Goes On', ['Lil Baby', 'Gunna', 'Lil Uzi Vert'], 0.0021, 0.716, 0.541, 0, -7.909, 0.149, 0.387, 0.115, 1, 139.978, ['trap', 'atl hip hop', 'melodic rap', 'atl trap', 'rap', 'hip hop', 'rage rap', 'philly rap']], ['6IFpwncCkKrXHyP0RuG9r6', 'Bonnie & Clyde', ['DEAN'], 0.426, 0.642, 0.642, 7.74e-06, -5.774, 0.0725, 0.475, 0.672, 3, 94.958, ['k-pop', 'korean r&b']], ['6rTInqW3YECMkQsBEHw4sd', 'Solid (feat. Drake)', ['Young Stoner Life', 'Young Thug', 'Gunna', 'Drake'], 0.0392, 0.887, 0.485, 0, -9.358, 0.2, 0.328, 0.141, 11, 125.987, ['trap', 'atl hip hop', 'canadian hip hop', 'pop rap', 'canadian pop', 'gangster rap', 'melodic rap', 'atl trap', 'rap', 'hip hop']], ['248OFOZef6ShXv6DGgbnxU', 'Saved', ['Khalid'], 0.189, 0.739, 0.448, 0, -10.28, 0.138, 0.553, 0.118, 10, 81.044, ['pop', 'pop r&b']], ['5QDGKcQLqOJvoiu8eScEaM', "Let Em' Know", ['Bryson Tiller'], 0.0336, 0.489, 0.406, 0, -11.44, 0.232, 0.178, 0.344, 9, 110.551, ['kentucky hip hop', 'r&b', 'rap']], ['48WidxP9CqyYtk97pwGZ3c', 'Wasting Time ( feat. Drake & The Neptunes )', ['Brent Faiyaz', 'Drake', 'The Neptunes'], 0.0669, 0.443, 0.72, 0, -3.743, 0.0733, 0.374, 0.0909, 7, 89.378, ['canadian hip hop', 'canadian pop', 'pop rap', 'r&b', 'rap', 'hip hop']]]
    user1_long = to_dataframe(arr3, columns, long_score)

    short_term = []
    items = sp.current_user_top_tracks(limit=50, offset=0, time_range='short_term')['items']
    short_term += items
    final_short = to_array(short_term)
    user2_short = to_dataframe(final_short, columns, short_score)

    medium_term = []
    items = sp.current_user_top_tracks(limit=50, offset=0, time_range='medium_term')['items']
    medium_term += items
    final_medium = to_array(medium_term)
    user2_medium = to_dataframe(final_medium, columns, medium_score)

    long_term = []
    items = sp.current_user_top_tracks(limit=50, offset=0, time_range='long_term')['items']
    long_term += items
    final_long = to_array(long_term)
    user2_long = to_dataframe(final_long, columns, long_score)

    column_names = []
    for col in user1_short.columns:
        column_names.append(col)

    def make_master_row(row):
        song = []
        for x in column_names:
            song.append(row[x])
        return song

    #Function to combine the three dataframes into one master, increasing score of a song if it appears in more than one
    def score_songs(short, medium, long):   
        songs = pd.DataFrame(columns=column_names)
        #repeat_value = how much I increase the score if it appears more than once in short/medium/long term
        repeat_value = 10
        
        for index, row in short.iterrows():
            songs.loc[len(songs.index)] = make_master_row(row)
                
        for index, row in medium.iterrows():
            if len((songs[songs["song_id"] == row["song_id"]]).index) == 0:
                songs.loc[len(songs.index)] = make_master_row(row)
            else:
                current_score = int((songs.loc[songs['song_id'] == row["song_id"]])["score"])
                new_score = int(row["score"])
                index = songs.index[songs["song_id"] == row["song_id"]]
                if current_score < new_score:
                    songs.loc[index, "score"] = new_score + repeat_value
                else:
                    songs.loc[index, "score"] = current_score + repeat_value
                
        for index, row in long.iterrows():
            if len((songs[songs["song_id"] == row["song_id"]]).index) == 0:            
                songs.loc[len(songs.index)] = make_master_row(row)
            else:
                current_score = int((songs.loc[songs['song_id'] == row["song_id"]])["score"])
                new_score = int(row["score"])
                index = songs.index[songs["song_id"] == row["song_id"]]
                if current_score < new_score:
                    songs.loc[index, "score"] = new_score + repeat_value
                else:
                    songs.loc[index, "score"] = current_score + repeat_value
                
        return songs

    user2_master = score_songs(user2_short, user2_medium, user2_long)
    user2_master = user2_master.sort_values(by=['score'], ascending=False)
    user2_master[['artist', 'genres']] = user2_master[['artist', 'genres']].apply(list)
    user2_master['score_rank'] = [x for x in range(1,len(user2_master) + 1)]



    user1_master = score_songs(user1_short, user1_medium, user1_long)
    user1_master = user1_master.sort_values(by=['score'], ascending=False)
    user1_master[['artist', 'genres']] = user1_master[['artist', 'genres']].apply(list)
    user1_master['score_rank'] = [x for x in range(1,len(user1_master) + 1)]


    #replacing spaces in genres to underscores for tfidf reasons
    replace_spaces = lambda lst: [x.replace(' ', '_') for x in lst]
    user1_master['genres'] = user1_master['genres'].apply(replace_spaces)
    user2_master['genres'] = user2_master['genres'].apply(replace_spaces)


    # Songs that both like

    song_matches1 = user1_master[user1_master['song_id'].isin(user2_master['song_id'])].sort_values(by='song_id')
    song_matches2 = user2_master[user2_master['song_id'].isin(user1_master['song_id'])].sort_values(by='song_id')
    concat = song_matches1.merge(song_matches2, how='inner', on='song_id')
    concat['score'] = concat['score_x'] + concat['score_y']
    matches = pd.concat([concat.iloc[:, :3], concat['score']], axis=1)


    # Songs that one listens to that the other would like

    # Song metrics

    num_cols = ['acousticness', 'danceability', 'energy', 'instrumentalness',
        'loudness', 'speechiness', 'valence', 'liveness', 'tempo']
    user1_numbers = user1_master[num_cols]
    user2_numbers = user2_master[num_cols]

    #Combining all songs for umap reduction to better compare distances
    combined = min_max_scaling((pd.concat([user1_numbers,user2_numbers],axis=0)))


    umap_model = umap.UMAP()
    umap_embedding = umap_model.fit_transform(combined)

    umap1 = umap_embedding[:len(user1_master)]
    umap2 = umap_embedding[len(user1_master):]

    #Silhuotette method to determine optimal number of clusters
    ran = [x for x in range(2,11)]
    silhouette_scores = []
    for clusters in ran:
        model = KMeans(n_clusters=clusters, n_init=10)
        cluster_labels = model.fit_predict(umap1)
        silhouette_avg = silhouette_score(umap1, cluster_labels)
        silhouette_scores.append(silhouette_avg * umap1.shape[0])


    user1_cluster_num = np.where(silhouette_scores == max(silhouette_scores))
    user1_cluster_num = ran[user1_cluster_num[0][0]]


    model = KMeans(n_clusters=user1_cluster_num, n_init=10)
    clusters = model.fit_predict(umap1)


    ran = [x for x in range(2,11)]
    silhouette_scores = []
    for clusters in ran:
        model = KMeans(n_clusters=clusters, n_init=10)
        cluster_labels = model.fit_predict(umap2)
        silhouette_avg = silhouette_score(umap2, cluster_labels)
        silhouette_scores.append(silhouette_avg * umap2.shape[0])


    user2_cluster_num = np.where(silhouette_scores == max(silhouette_scores))
    user2_cluster_num = ran[user2_cluster_num[0][0]]


    model = KMeans(n_clusters=user2_cluster_num, n_init=10)
    clusters = model.fit_predict(umap2)

    user1_master, user1_means = cluster_assign(umap1, user1_master, user1_cluster_num)


    user2_master, user2_means = cluster_assign(umap2, user2_master, user2_cluster_num)


    user1_recs = user2_master.copy()
    user2_recs = user1_master.copy()

    #Creating clusters of the users songs and using the cluster centers to represent one type of music they like
    clusters1 = []
    for i in range(len(user1_means.cluster_centers_)):
        cluster_center = i
        cluster_name = "cluster" + str(i) + "_distance"
        clusters1.append(cluster_name)
        user1_recs[cluster_name] = calculate_distance(i,user1_means, umap2)

    #finding the smallest distance of each song to cluster centers of other users songs
    user1_recs['min_value'] = user1_recs[[cluster for cluster in clusters1]].min(axis=1)

    user1_recs = user1_recs.sort_values(by='min_value')
    user1_recs['distance_rank'] = user1_recs['min_value'].rank(ascending=True, method='max').astype(int)

    clusters2 = []
    for i in range(len(user2_means.cluster_centers_)):
        cluster_center = i
        cluster_name = "cluster" + str(i) + "_distance"
        clusters2.append(cluster_name)
        user2_recs[cluster_name] = calculate_distance(i,user2_means, umap1)

    user2_recs['min_value'] = user2_recs[[cluster for cluster in clusters2]].min(axis=1)

    user2_recs = user2_recs.sort_values(by='min_value')
    user2_recs['distance_rank'] = user2_recs['min_value'].rank(ascending=True, method='max').astype(int)

    #Genres


   #each song = seperate document
    def get_genre(df):
        tfidf = TfidfVectorizer(token_pattern=r"\b[\w&+'-]+(?:-[\w&+'-]+)*\b")
        tfidf_matrix =  tfidf.fit_transform(df['genres'].apply(lambda x: " ".join(x)))
        genre_df = pd.DataFrame(tfidf_matrix.toarray())
        genre_df.columns = [i for i in tfidf.get_feature_names_out()]
        genre_df.reset_index(drop = True, inplace=True)
        return genre_df
    
    user1_genres = get_genre(user2_recs)
    user2_genres = get_genre(user1_recs)

    user1_genres, user2_genres = user1_genres.align(user2_genres, fill_value=0.0, axis=1)
    
    def get_taste(genres):
        ls = []
        for x in genres:
            for y in x:
                ls.append(y)
        res = [" ".join([str(item) for item in ls])]
        total = len(res[0].split(' '))
        word_frequencies = {}
        for word in res[0].split(' '):
            if word not in word_frequencies:
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

        for x in word_frequencies:
            word_frequencies[x] = word_frequencies[x] / total
        # Create a dataframe of term frequencies
        return pd.DataFrame(list(word_frequencies.values()), index=list(word_frequencies.keys())).transpose()

    user1_taste = get_taste(user2_recs['genres'])
    user2_taste = get_taste(user1_recs['genres'])
    
    user1_taste, user2_taste = user1_taste.align(user2_taste, fill_value=0.0, axis=1)

    user1_taste, user1_genres = user1_taste.align(user1_genres, fill_value=0.0, axis=1)
    user1_taste, user2_genres = user1_taste.align(user2_genres, fill_value=0.0, axis=1)
    user1_taste, user2_taste = user1_taste.align(user2_taste, fill_value=0.0, axis=1)


    def calculate_distance_genres(taste, library):
        distances = []
        for x in library:
            distance = 1 - dst.cosine(taste, x)
            if distance == 1:
                distances.append(0)
            else:
                distances.append(distance)
        return distances
    
    user1_taste_final = np.array(user1_taste)[0]
    user2_taste_final = np.array(user2_taste)[0]
    user1_genres_final = np.array(user1_genres)
    user2_genres_final = np.array(user2_genres)

    user2_recs['genre_similarity'] = calculate_distance_genres(user2_taste_final, user1_genres_final)
    user1_recs['genre_similarity'] = calculate_distance_genres(user1_taste_final, user2_genres_final)


    user2_recs = user2_recs.sort_values(by='genre_similarity', ascending=False)
    user2_recs['genre_rank'] = user2_recs['genre_similarity'].rank(ascending=False, method='max').astype(int)
    user1_recs = user1_recs.sort_values(by='genre_similarity', ascending=False)
    user1_recs['genre_rank'] = user1_recs['genre_similarity'].rank(ascending=False, method='max').astype(int)


    #Combining ranks


    col_order = user1_recs.columns[:15].tolist()
    col_order1 = col_order + ['cluster'] +  clusters1 + ['min_value', 'genre_similarity', 'score_rank', 'distance_rank', 'genre_rank']
    col_order2 = col_order + ['cluster'] +  clusters2 + ['min_value', 'genre_similarity', 'score_rank', 'distance_rank', 'genre_rank']

    user1_recs = user1_recs.reindex(columns=col_order1)
    user2_recs = user2_recs.reindex(columns=col_order2)

    user2_recs['average_rank'] = combine_ranks(user2_recs)
    user2_final = user2_recs.sort_values(by=['average_rank']).head(20)

    user1_recs['average_rank'] = combine_ranks(user1_recs)
    user1_final =user1_recs.sort_values(by=['average_rank']).head(20)

    tracks = np.concatenate((np.array(user1_final['song_id']), np.array(user2_final['song_id'])), axis=0)

    #Creating actual playlist
    playlist = sp.user_playlist_create(sp.me()['id'], sp.me()['display_name'] + ' x Will joint playlist ', public=True, collaborative=False)


    sp.user_playlist_add_tracks(sp.me()['id'], playlist['id'], tracks=tracks, position=None)


    link = playlist['external_urls']['spotify']

    #Prepping for output
    rel_cols = ['song_id', 'name', 'artist', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'loudness', 'speechiness', 'valence', 'liveness', 'genres', 'score_rank', 'distance_rank', 'genre_rank', 'average_rank']
    nums = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'loudness', 'speechiness', 'valence', 'liveness', 'score_rank', 'distance_rank', 'genre_rank', 'average_rank']
    user1_final[nums] = np.round(np.array(user1_final[nums]), 2)
    user1 = np.array(user1_final[rel_cols])
    user2_final[nums] = np.round(np.array(user2_final[nums]), 2)
    user2 = np.array(user2_final[rel_cols])
    matches = np.array(matches.sort_values(by='score', ascending=False)[matches['score'] > 40])

    return render_template('tracks.html', user1 = user1, user2= user2, matches = matches, link = link, show_loading=True)

@app.route('/loading')
def loading_screen():
    return render_template('loading.html')


def to_array(song_list):
    final = []
    sp = spotipy.Spotify(auth=session.get('token_info').get('access_token'))
    for x in range(len(song_list)):
        song = []
        dict = sp.audio_features(song_list[x]["id"])
        song.append(song_list[x]["id"])
        song.append(song_list[x]["name"])
        artists = []
        if len(song_list[x]["artists"]) > 1:
            for y in song_list[x]["artists"]:
                artists.append(y["name"])
            song.append(artists)
        else:
            artists.append(song_list[x]["artists"][0]["name"])
            song.append(artists)
        song.append(dict[0]["acousticness"])
        song.append(dict[0]["danceability"])
        song.append(dict[0]["energy"])
        song.append(dict[0]["instrumentalness"])
        song.append(dict[0]["loudness"])
        song.append(dict[0]["speechiness"])
        song.append(dict[0]["valence"])
        song.append(dict[0]["liveness"])
        song.append(dict[0]["key"])
        song.append(dict[0]["tempo"])
        if len(song_list[x]["artists"]) > 1:
            genres=[]
            for y in song_list[x]["artists"]:
                genres.append(sp.artist(y['id'])['genres'])
            final_genres = []
            for z in genres:
                for a in z:
                    final_genres.append(a)
            unique = list(set(final_genres))
            song.append(unique)
        else:
            song.append(sp.artist(song_list[x]["artists"][0]["id"])["genres"])
        final.append(song)
    return final

#Function to turn an array into a dataframe with scores
def to_dataframe(array, cols, scores):
    df = pd.DataFrame(array)
    df.columns = cols
    df["score"] = scores
    return df

#Function to create an array of an indexes from a dataframe's index
def new_index(df):
    new_index = []
    for x in range(len(df)):
        new_index.append(x)
    return new_index

#Comparing scores
def compare_scores(song_matches, second_master):
    for x in song_matches['song_id']:
        song_matches = pd.concat([song_matches, second_master[second_master['song_id']==x]], ignore_index=True)
    return song_matches

#normalize variables column-wise from 0-1
def min_max_scaling(df):
    for col in df:
        min_value = df[col].min()
        max_value = df[col].max()
        
        # Perform Min-Max scaling for the entire DataFrame
        df[col] = (df[col] - min_value) / (max_value - min_value)
    
    return df

#Assigning clusters to rows of a dataframe
def cluster_assign(nums, master, clusters):
    #Clustering
    kmeans = KMeans(n_clusters = clusters, n_init=10)
    kmeans.fit(nums)

    #Reassigning cluster values
    master["cluster"] = kmeans.labels_
    
    return master, kmeans

#Function to return cosine similarity for genres
def calculate_genre_distance(taste, library):
    distances = []
    for x in library:
        distance = 1 - dst.cosine(taste, x)
        if distance == 1:
            distances.append(0)
        else:
            distances.append(distance)
    return distances

#Function to return distance
def calculate_distance(cluster_center, kmeans, library):
    distances = []
    for x in library:
        distance = dst.euclidean(kmeans.cluster_centers_[cluster_center], x)
        distances.append(distance)
    return distances

#one doc, basically just term frequency across all of a users songs (used to represent a user's overall taste)
def get_taste(genres):
    ls = []
    for x in genres:
        for y in x:
            ls.append(y)
    res = [" ".join([str(item) for item in ls])]
    tfidf = TfidfVectorizer(token_pattern=r'\b[\w&+]+(?:-[\w&+]+)*\b')
    tfidf_matrix =  tfidf.fit_transform(res)
    genre_df = pd.DataFrame(tfidf_matrix.toarray())
    genre_df.columns = [i for i in tfidf.get_feature_names_out()]
    genre_df.reset_index(drop = True, inplace=True)
    return genre_df

#each song = seperate document, used for individual songs
def get_genre(df):
    tfidf = TfidfVectorizer(token_pattern=r'\b[\w&+]+(?:-[\w&+]+)*\b')
    tfidf_matrix =  tfidf.fit_transform(df['genres'].apply(lambda x: " ".join(x)))
    genre_df = pd.DataFrame(tfidf_matrix.toarray())
    genre_df.columns = [i for i in tfidf.get_feature_names_out()]
    genre_df.reset_index(drop = True, inplace=True)
    return genre_df


#find the columns that are missing
def missing_columns(df1, df2):
    return list(set(df1.columns) - set(df2.columns))



#add the missing columns
def add_missing(df, column_list):
    df = df.reindex(columns=df.columns.tolist() + column_list)
    return df

#Combining the three ranks with how much the original user likes their own song being the half as important as the other two
def combine_ranks(recs):
    return 0.4 * recs['distance_rank'] + 0.2 * recs['score_rank'] + 0.4 * recs['genre_rank']

if __name__ == "__main__":
    app.run(port=5001, debug=True)