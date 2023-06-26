from flask import Flask, request, url_for, session, redirect, render_template, make_response, Response
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
    arr1 = [['3CblJq8QQQ0bb7vwJu8c3v', '4EVA (feat. Pharrell Williams)', ['KAYTRAMINÉ', 'Aminé', 'KAYTRANADA', 'Pharrell Williams'], 0.0248, 0.83, 0.695, 0.0755, -9.445, 0.0716, 0.536, 0.0573, 6, 112.046, ['alternative r&b', 'portland hip hop', 'rap', 'underground hip hop', 'pop rap', 'dance pop', 'lgbtq+ hip hop', 'escape room', 'pop', 'indie soul']], ['19mXqYcLHY716cN53T1d1E', 'Wish That You Were Mine', ['The Manhattans'], 0.555, 0.621, 0.534, 1.29e-05, -10.09, 0.036, 0.574, 0.297, 9, 119.615, ['classic soul', 'disco', 'funk', 'motown', 'philly soul', 'quiet storm', 'soul']], ['3Um9toULmYFGCpvaIPFw7l', "What's Going On", ['Marvin Gaye'], 0.447, 0.283, 0.716, 0, -9.632, 0.0986, 0.828, 0.399, 1, 201.96, ['classic soul', 'motown', 'neo soul', 'northern soul', 'quiet storm', 'soul']], ['1zZdEavQr1Vl769ZMqYUvk', 'Up All Night', ['Kenichiro Nishihara', 'SIRUP'], 0.322, 0.649, 0.8, 0.0147, -5.786, 0.0283, 0.561, 0.126, 4, 111.982, ['japanese soul', 'ambeat', 'japanese r&b']], ['0wOtc2nY3NOohp4xSwOyTN', 'We Might Even Be Falling In Love (Duet) - Spotify Singles', ['Victoria Monét', 'Bryson Tiller'], 0.473, 0.731, 0.423, 0.000413, -10.147, 0.0784, 0.78, 0.129, 6, 76.964, ['rap', 'alternative r&b', 'r&b', 'kentucky hip hop']], ['39sDitIeCMrVX2QyXHY46t', 'Blue Hair', ['TV Girl'], 0.554, 0.751, 0.72, 0.0497, -6.376, 0.0303, 0.884, 0.258, 4, 135.73, []], ['4OYXAD2OSy0RkSsQ0D9BEQ', 'Heartless', ['Kenichiro Nishihara', 'Michael Kaneko'], 0.599, 0.797, 0.79, 0.00271, -5.086, 0.0387, 0.824, 0.114, 5, 103.966, ['ambeat', 'japanese r&b']], ['1dHiSGzb9WFtDKnBFJs4KO', 'Just Say', ['Coco & Breezy', 'Tara Carosielli'], 0.0465, 0.83, 0.419, 0.0145, -9.76, 0.063, 0.511, 0.0815, 5, 119.993, ['soulful house', 'indie electropop']], ['4ZwgD4frTwoDdOUsjyoqAJ', 'Dancing In The Courthouse', ['Dominic Fike'], 0.187, 0.621, 0.741, 0, -4.479, 0.0611, 0.691, 0.227, 2, 76.464, ['alternative pop rock']], ['5cFREA8Fg75ytjbOC1NSOx', 'Pray', ['Kenichiro Nishihara', 'MARTER'], 0.0201, 0.616, 0.782, 0.0227, -6.301, 0.0318, 0.899, 0.0795, 0, 193.992, ['ambeat']], ['4UeWKazLR1ZwwSVnLw9Ir9', '踊り子', ['Vaundy'], 0.845, 0.718, 0.475, 0.801, -11.469, 0.0611, 0.678, 0.105, 7, 157.032, ['j-pop', 'japanese soul']], ['2i2gDpKKWjvnRTOZRhaPh2', 'Moonlight', ['Kali Uchis'], 0.511, 0.639, 0.723, 0, -6.462, 0.0532, 0.878, 0.167, 7, 136.872, ['colombian pop']], ['2aQpISWUBToaF84DDiTeRV', 'Be My Lover (feat. La Bouche) - 2023 Mix', ['Hypaton', 'David Guetta', 'La Bouche'], 0.0347, 0.589, 0.973, 0.000945, -5.317, 0.0426, 0.115, 0.45, 8, 126.003, ['pop dance', 'big room', 'eurodance', 'diva house', 'german techno', 'europop', 'edm', 'dance pop', 'pop']], ['59acp1OhcvxwVBwQJBYKuX', 'Loose', ['Daniel Caesar'], 0.926, 0.198, 0.162, 0.00896, -15.815, 0.0343, 0.0527, 0.135, 1, 70.453, ['canadian contemporary r&b']], ['2lnQcP5hr4RKR63TFvnI4k', 'Overjoyed', ['Stevie Wonder'], 0.541, 0.221, 0.264, 0, -17.533, 0.0332, 0.169, 0.148, 10, 83.128, ['motown', 'soul']], ['4qDpLaFGf5ampf2DXD2TMA', 'Where You Are', ['John Summit', 'Hayla'], 0.00953, 0.56, 0.832, 0.00541, -6.432, 0.0363, 0.0818, 0.546, 9, 126.0, ['house', 'uk dance']], ['6RANU8AS5ICU5PEHh8BYtH', "Isn't She Lovely", ['Stevie Wonder'], 0.182, 0.481, 0.826, 0.00352, -6.974, 0.0851, 0.819, 0.324, 1, 118.679, ['motown', 'soul']], ['0L26wNt3MUtn7BrTaHGtjj', 'Oasis', ['Crush', 'ZICO'], 0.222, 0.46, 0.81, 0, -4.158, 0.254, 0.836, 0.435, 4, 96.207, ['korean r&b', 'k-pop']], ['07gf6qIWA6yt58pR7uBDSw', 'Crave You - Hush Hush Bootleg', ['Flight Facilities', 'Giselle', 'Hush Hush'], 0.141, 0.83, 0.807, 0.00343, -6.899, 0.13, 0.163, 0.0591, 11, 120.012, ['australian indie', 'australian dance', 'nu disco', 'aussietronica', 'indietronica']], ['1qtwebmDBKPQEggSKGoSfy', 'Since I Have A Lover', ['6LACK'], 0.0231, 0.707, 0.73, 0.018, -4.152, 0.0317, 0.11, 0.286, 0, 120.022, ['atl hip hop', 'melodic rap', 'r&b', 'rap', 'trap']], ['3BbD2sqk7P7Rc9V0KF9o4s', 'My Humps', ['Joshwa', 'Lee Foss'], 0.0418, 0.862, 0.79, 0.00615, -8.412, 0.0672, 0.587, 0.0746, 4, 129.005, ['house', 'uk tech house', 'deep disco house']], ['1HA3u5rZF4yvaGNnNyKAMT', "Blowin' In The Wind", ['Stevie Wonder'], 0.361, 0.449, 0.434, 0, -8.486, 0.0356, 0.726, 0.262, 10, 104.674, ['motown', 'soul']], ['3lsiqFV6SKhBgzQCpuM1JR', 'Miracle (with Ellie Goulding) - Mau P Remix', ['Calvin Harris', 'Ellie Goulding', 'Mau P'], 0.00134, 0.642, 0.943, 0.085, -6.87, 0.0519, 0.44, 0.155, 8, 128.005, ['house', 'uk dance', 'indietronica', 'metropopolis', 'electro house', 'progressive house', 'uk pop', 'edm', 'dance pop', 'pop']], ['6iCJCZqDJjmBxt07Oid6FI', 'Buttercup', ['Hippo Campus'], 0.199, 0.623, 0.763, 6.52e-06, -4.412, 0.0719, 0.199, 0.126, 9, 108.871, ['indie pop', 'minneapolis indie', 'modern rock']], ['7E7YqRQZiASXIENrGYlpSU', 'DEUS DA GUERRA', ['$pidxrs?808', 'LEGIXN', 'FXRCE'], 0.0128, 0.646, 0.861, 0.104, -5.721, 0.0314, 0.373, 0.0767, 10, 135.993, ['phonk brasileiro']], ['3SdTKo2uVsxFblQjpScoHy', 'Stand by Me', ['Ben E. King'], 0.57, 0.65, 0.306, 7.07e-06, -9.443, 0.0393, 0.605, 0.0707, 9, 118.068, ['rock-and-roll', 'soul']], ['6Xom58OOXk2SoU711L2IXO', 'Moscow Mule', ['Bad Bunny'], 0.294, 0.804, 0.674, 1.18e-06, -5.453, 0.0333, 0.292, 0.115, 5, 99.968, ['reggaeton', 'trap latino', 'urbano latino']], ['7E6Uy2FLll3gaby9iyCfqz', 'Homiesexual (with Ty Dolla $ign)', ['Daniel Caesar', 'Ty Dolla $ign'], 0.102, 0.364, 0.48, 1.75e-06, -10.807, 0.0885, 0.32, 0.0979, 8, 198.32, ['canadian contemporary r&b', 'trap soul', 'hip hop', 'pop rap', 'trap', 'southern hip hop', 'r&b']], ['40SBS57su9xLiE1WqkXOVr', 'Afraid To Feel', ['LF SYSTEM'], 0.0166, 0.578, 0.912, 0.00362, -3.929, 0.114, 0.68, 0.273, 1, 127.87, ['uk dance']], ['0jHkgTtTaqg5LNCiYDQPUB', "Let's Get It On", ['Marvin Gaye'], 0.0392, 0.55, 0.611, 0, -10.657, 0.0443, 0.626, 0.0631, 3, 168.512, ['classic soul', 'motown', 'neo soul', 'northern soul', 'quiet storm', 'soul']], ['1jQfgl9WRle7D8a3GXLwaD', 'Transform (feat. Charlotte Day Wilson)', ['Daniel Caesar', 'Charlotte Day Wilson'], 0.511, 0.498, 0.292, 1.9e-05, -10.656, 0.031, 0.348, 0.256, 5, 68.963, ['canadian contemporary r&b', 'alternative r&b', 'indie soul']], ['6AQbmUe0Qwf5PZnt4HmTXv', "Boy's a liar Pt. 2", ['PinkPantheress', 'Ice Spice'], 0.252, 0.696, 0.809, 0.000128, -8.254, 0.05, 0.857, 0.248, 5, 132.962, ['bronx drill']], ['0dAfw35k2hBsnbSl74AVJF', 'dashstar*', ['Knock2'], 0.00872, 0.699, 0.949, 0.328, -2.994, 0.0739, 0.186, 0.151, 9, 126.07, []], ['5uakDGEx9HegMZZi840VzH', 'Evian (feat. PinkPantheress, Rizloski & Rax)', ['GoldLink', 'PinkPantheress', 'Rizloski', 'Rax'], 0.101, 0.824, 0.848, 0.0193, -7.051, 0.238, 0.968, 0.1, 8, 134.978, ['alternative r&b', 'dmv rap']], ['08M1S4uwtmPM0jIO1qNyhx', 'NOTHIN LIKE U (feat. Ty Dolla $ign)', ['KAYTRANADA', 'Ty Dolla $ign'], 0.0589, 0.787, 0.805, 1.61e-06, -6.294, 0.409, 0.769, 0.192, 9, 104.862, ['alternative r&b', 'trap soul', 'hip hop', 'pop rap', 'trap', 'southern hip hop', 'r&b', 'lgbtq+ hip hop', 'escape room', 'indie soul']], ['0Y71FEcRkyZOh4hySnEGB5', 'Baggage', ['Breakfast Santana', 'Khaji Beats'], 0.404, 0.629, 0.717, 3.33e-06, -8.505, 0.243, 0.655, 0.238, 0, 117.981, []], ['28JBD8p18xNuOfyV7Cotdn', 'Massive', ['Drake'], 0.114, 0.499, 0.671, 0.0169, -6.774, 0.0561, 0.0557, 0.148, 4, 124.994, ['canadian hip hop', 'canadian pop', 'hip hop', 'rap', 'toronto rap']], ['2C0KFbb4v9CNWR5c9jWcKC', 'Andromeda (feat. DRAM)', ['Gorillaz', 'DRAM'], 0.003, 0.788, 0.472, 0.0322, -9.021, 0.0475, 0.257, 0.59, 0, 130.084, ['alternative hip hop', 'viral trap', 'rock', 'underground hip hop', 'virginia hip hop', 'trap', 'modern rock']], ['2ns1kl3c5NMvGCt2xVMNlI', 'Score (feat. SZA & 6LACK)', ['Isaiah Rashad', 'SZA', '6LACK'], 0.442, 0.5, 0.515, 0.00018, -8.283, 0.31, 0.566, 0.0681, 7, 85.173, ['rap', 'melodic rap', 'underground hip hop', 'hip hop', 'atl hip hop', 'trap', 'tennessee hip hop', 'r&b', 'pop']], ['3Dc86yXq3haXkQ5jwlxLiY', 'Shot My Baby', ['Daniel Caesar'], 0.00481, 0.483, 0.641, 0.0121, -6.945, 0.0412, 0.16, 0.223, 1, 84.96, ['canadian contemporary r&b']], ['47v4uUtj5AukJmCbMq4Kry', 'Freeze Tag (feat. Phoelix)', ['Terrace Martin', 'Robert Glasper', '9th Wonder', 'Kamasi Washington', 'Dinner Party', 'Phoelix'], 0.0596, 0.653, 0.598, 0, -6.928, 0.113, 0.137, 0.253, 11, 82.911, ['alternative hip hop', 'neo soul', 'jazz', 'alternative r&b', 'north carolina hip hop', 'afrofuturism', 'contemporary jazz', 'hip hop', 'jazz saxophone', 'neo r&b', 'indie hip hop', 'modern jazz piano', 'indie jazz', 'indie soul']], ['5aRZk9oWIYUB5alrTs8TTV', 'From Eden', ['Hozier'], 0.584, 0.399, 0.673, 2.44e-05, -5.506, 0.0509, 0.285, 0.118, 0, 142.255, ['irish singer-songwriter', 'modern rock', 'pop']], ['5de9Ho64dovuQI8Uhn5gPD', "I Don't Wanna Do This Anymore", ['XXXTENTACION'], 0.843, 0.433, 0.778, 0, -7.681, 0.041, 0.764, 0.139, 4, 114.208, ['emo rap', 'miami hip hop']], ['7y6c07pgjZvtHI9kuMVqk1', 'Get It Together', ['Drake', 'Black Coffee', 'Jorja Smith'], 0.0945, 0.781, 0.721, 0.391, -9.133, 0.0514, 0.849, 0.104, 5, 123.011, ['uk contemporary r&b', 'canadian pop', 'rap', 'toronto rap', 'hip hop', 'r&b', 'south african pop dance', 'south african house', 'canadian hip hop']], ['5UkoWcFsdWn8NtSAh73Vli', 'Temptation', ['SIDEPIECE'], 0.00665, 0.808, 0.878, 0.0152, -5.282, 0.0948, 0.624, 0.208, 9, 125.011, ['edm']], ['1c0hsvHLELX6y8qymnpLKL', 'Soltera - Remix', ['Lunay', 'Daddy Yankee', 'Bad Bunny'], 0.361, 0.795, 0.783, 0, -4.271, 0.0432, 0.799, 0.437, 5, 92.01, ['trap latino', 'reggaeton flow', 'latin hip hop', 'reggaeton', 'urbano latino']], ['38spM0LahLUfQhOMAqA7AI', 'Be Your Girl (Kaytranada Edition)', ['Teedra Moses', 'KAYTRANADA'], 0.208, 0.864, 0.682, 5.27e-06, -6.73, 0.386, 0.268, 0.1, 6, 115.012, ['neo soul', 'alternative r&b', 'lgbtq+ hip hop', 'escape room', 'indie soul']], ['39JofJHEtg8I4fSyo7Imft', 'B.O.T.A. (Baddest Of Them All) - Edit', ['Eliza Rose', 'Interplanetary Criminal'], 0.164, 0.736, 0.906, 0.585, -7.589, 0.048, 0.698, 0.106, 0, 137.001, ['house', 'breaks', 'experimental house']], ['4AwJSk491AvHk2AAJReGzZ', 'Let Me Go', ['Daniel Caesar'], 0.689, 0.601, 0.481, 0.0033, -8.366, 0.0634, 0.484, 0.109, 7, 152.994, ['canadian contemporary r&b']], ['4NczzeHBQPPDO0B9AAmB8d', 'Assumptions', ['Sam Gellaitry'], 0.123, 0.639, 0.815, 0.00113, -4.718, 0.221, 0.443, 0.193, 11, 126.055, ['future bass', 'scottish electronic', 'vapor twitch']]]
    user1_short = to_dataframe(arr1, columns, short_score)

    arr2 = [['0wOtc2nY3NOohp4xSwOyTN', 'We Might Even Be Falling In Love (Duet) - Spotify Singles', ['Victoria Monét', 'Bryson Tiller'], 0.473, 0.731, 0.423, 0.000413, -10.147, 0.0784, 0.78, 0.129, 6, 76.964, ['rap', 'alternative r&b', 'r&b', 'kentucky hip hop']], ['0Yz3F0UGDibDe8uU69zmjn', 'And July', ['HEIZE', 'DEAN', 'dj friz'], 0.0401, 0.734, 0.852, 0, -3.85, 0.0916, 0.787, 0.196, 2, 98.504, ['korean pop', 'korean r&b', 'k-pop']], ['65FftemJ1DbbZ45DUfHJXE', 'OMG', ['NewJeans'], 0.357, 0.804, 0.771, 3.07e-06, -4.067, 0.0433, 0.739, 0.108, 9, 126.956, ['k-pop', 'k-pop girl group']], ['5uakDGEx9HegMZZi840VzH', 'Evian (feat. PinkPantheress, Rizloski & Rax)', ['GoldLink', 'PinkPantheress', 'Rizloski', 'Rax'], 0.101, 0.824, 0.848, 0.0193, -7.051, 0.238, 0.968, 0.1, 8, 134.978, ['alternative r&b', 'dmv rap']], ['3CblJq8QQQ0bb7vwJu8c3v', '4EVA (feat. Pharrell Williams)', ['KAYTRAMINÉ', 'Aminé', 'KAYTRANADA', 'Pharrell Williams'], 0.0248, 0.83, 0.695, 0.0755, -9.445, 0.0716, 0.536, 0.0573, 6, 112.046, ['alternative r&b', 'portland hip hop', 'rap', 'underground hip hop', 'pop rap', 'dance pop', 'lgbtq+ hip hop', 'escape room', 'pop', 'indie soul']], ['39JofJHEtg8I4fSyo7Imft', 'B.O.T.A. (Baddest Of Them All) - Edit', ['Eliza Rose', 'Interplanetary Criminal'], 0.164, 0.736, 0.906, 0.585, -7.589, 0.048, 0.698, 0.106, 0, 137.001, ['house', 'breaks', 'experimental house']], ['4Tla2jt77nO70DgGwFejbK', 'Run It Up', ['Snakehips', 'EARTHGANG'], 0.00621, 0.838, 0.793, 0, -5.176, 0.145, 0.527, 0.0962, 2, 120.062, ['uk dance', 'tropical house', 'rap', 'underground hip hop', 'hip hop', 'electropop', 'atl hip hop', 'psychedelic hip hop']], ['40SBS57su9xLiE1WqkXOVr', 'Afraid To Feel', ['LF SYSTEM'], 0.0166, 0.578, 0.912, 0.00362, -3.929, 0.114, 0.68, 0.273, 1, 127.87, ['uk dance']], ['4AwJSk491AvHk2AAJReGzZ', 'Let Me Go', ['Daniel Caesar'], 0.689, 0.601, 0.481, 0.0033, -8.366, 0.0634, 0.484, 0.109, 7, 152.994, ['canadian contemporary r&b']], ['08M1S4uwtmPM0jIO1qNyhx', 'NOTHIN LIKE U (feat. Ty Dolla $ign)', ['KAYTRANADA', 'Ty Dolla $ign'], 0.0589, 0.787, 0.805, 1.61e-06, -6.294, 0.409, 0.769, 0.192, 9, 104.862, ['alternative r&b', 'trap soul', 'hip hop', 'pop rap', 'trap', 'southern hip hop', 'r&b', 'lgbtq+ hip hop', 'escape room', 'indie soul']], ['4qDpLaFGf5ampf2DXD2TMA', 'Where You Are', ['John Summit', 'Hayla'], 0.00953, 0.56, 0.832, 0.00541, -6.432, 0.0363, 0.0818, 0.546, 9, 126.0, ['house', 'uk dance']], ['6AQbmUe0Qwf5PZnt4HmTXv', "Boy's a liar Pt. 2", ['PinkPantheress', 'Ice Spice'], 0.252, 0.696, 0.809, 0.000128, -8.254, 0.05, 0.857, 0.248, 5, 132.962, ['bronx drill']], ['0aR99yr1CNoxlZxm2veB65', 'At All', ['KAYTRANADA'], 0.0399, 0.898, 0.471, 0.698, -10.13, 0.519, 0.41, 0.0694, 2, 120.099, ['alternative r&b', 'escape room', 'indie soul', 'lgbtq+ hip hop']], ['1ZGyyYFVOIIZtzsoC1A2mX', 'TIL I DIE', ['Space Rangers', 'Disco Lines'], 0.00045, 0.806, 0.931, 0.303, -3.35, 0.0579, 0.829, 0.0893, 11, 123.985, []], ['1dHiSGzb9WFtDKnBFJs4KO', 'Just Say', ['Coco & Breezy', 'Tara Carosielli'], 0.0465, 0.83, 0.419, 0.0145, -9.76, 0.063, 0.511, 0.0815, 5, 119.993, ['soulful house', 'indie electropop']], ['4UeWKazLR1ZwwSVnLw9Ir9', '踊り子', ['Vaundy'], 0.845, 0.718, 0.475, 0.801, -11.469, 0.0611, 0.678, 0.105, 7, 157.032, ['j-pop', 'japanese soul']], ['0dAfw35k2hBsnbSl74AVJF', 'dashstar*', ['Knock2'], 0.00872, 0.699, 0.949, 0.328, -2.994, 0.0739, 0.186, 0.151, 9, 126.07, []], ['7pARbCUoO1CTzU5ucMEaVF', 'Summer 91', ['Noizu'], 0.00598, 0.5, 0.953, 0.000882, -5.606, 0.0441, 0.327, 0.26, 2, 124.992, ['bass house', 'house']], ['75rGONmoi48LLYBFaGiYsv', 'Both Of Us - Edit', ['Jayda G'], 0.00851, 0.735, 0.71, 0.762, -10.222, 0.0766, 0.507, 0.0684, 11, 123.962, ['canadian electronic']], ['3Wnba0vkxR4nmXbaRF9foE', '化け猫 - Leaks From His Laptop', ['キタニタツヤ'], 0.108, 0.867, 0.55, 3.94e-05, -5.614, 0.105, 0.967, 0.179, 8, 119.888, ['japanese electropop']], ['01krBM9VOLv2CSgo1wPSut', 'Funk Soda', ['J Funk Boy'], 0.686, 0.738, 0.879, 0, -4.441, 0.341, 0.825, 0.381, 1, 123.222, []], ['19mXqYcLHY716cN53T1d1E', 'Wish That You Were Mine', ['The Manhattans'], 0.555, 0.621, 0.534, 1.29e-05, -10.09, 0.036, 0.574, 0.297, 9, 119.615, ['classic soul', 'disco', 'funk', 'motown', 'philly soul', 'quiet storm', 'soul']], ['69IKYOFqrwpldC0ue4kGPk', 'bbycakes (with Lil Uzi Vert, PinkPantheress & Shygirl)', ['Mura Masa', 'Lil Uzi Vert', 'PinkPantheress', 'Shygirl'], 0.161, 0.729, 0.813, 4.48e-05, -4.661, 0.0429, 0.777, 0.0364, 11, 148.028, ['art pop', 'escape room', 'alternative r&b', 'electra', 'channel islands indie', 'melodic rap', 'rap', 'hyperpop', 'grimewave', 'vapor soul', 'philly rap', 'indie soul']], ['38spM0LahLUfQhOMAqA7AI', 'Be Your Girl (Kaytranada Edition)', ['Teedra Moses', 'KAYTRANADA'], 0.208, 0.864, 0.682, 5.27e-06, -6.73, 0.386, 0.268, 0.1, 6, 115.012, ['neo soul', 'alternative r&b', 'lgbtq+ hip hop', 'escape room', 'indie soul']], ['4kdnniHBTCnzaYXrnsweSf', 'Poison', ['Klaus Veen'], 0.0428, 0.826, 0.925, 0.00621, -6.441, 0.254, 0.726, 0.0689, 1, 126.505, ['dutch tech house']], ['5ooCfBqZyTB5CTdu1x7S77', 'Everything You Have Done - Meduza Edit', ['GENESI', 'MEDUZA'], 0.0459, 0.678, 0.935, 0.202, -6.382, 0.0592, 0.485, 0.198, 2, 124.989, ['pop dance', 'uk dance', 'pop house']], ['1qtwebmDBKPQEggSKGoSfy', 'Since I Have A Lover', ['6LACK'], 0.0231, 0.707, 0.73, 0.018, -4.152, 0.0317, 0.11, 0.286, 0, 120.022, ['atl hip hop', 'melodic rap', 'r&b', 'rap', 'trap']], ['07gf6qIWA6yt58pR7uBDSw', 'Crave You - Hush Hush Bootleg', ['Flight Facilities', 'Giselle', 'Hush Hush'], 0.141, 0.83, 0.807, 0.00343, -6.899, 0.13, 0.163, 0.0591, 11, 120.012, ['australian indie', 'australian dance', 'nu disco', 'aussietronica', 'indietronica']], ['2qOm7ukLyHUXWyR4ZWLwxA', 'It Was A Good Day', ['Ice Cube'], 0.33, 0.798, 0.744, 0.000106, -5.328, 0.136, 0.794, 0.292, 7, 82.356, ['conscious hip hop', 'g funk', 'gangster rap', 'hip hop', 'rap', 'west coast rap']], ['1bA2ZK7CFxEMnyn1dWP2jp', 'Baby', ['Bakermat'], 0.135, 0.708, 0.922, 0.0513, -2.715, 0.116, 0.637, 0.225, 1, 125.961, ['minimal tech house', 'tropical house']], ['3Um9toULmYFGCpvaIPFw7l', "What's Going On", ['Marvin Gaye'], 0.447, 0.283, 0.716, 0, -9.632, 0.0986, 0.828, 0.399, 1, 201.96, ['classic soul', 'motown', 'neo soul', 'northern soul', 'quiet storm', 'soul']], ['423g0qldElg2Ge2UwLyCjt', '夜行', ['indigo la End'], 0.00334, 0.731, 0.744, 0.000106, -4.439, 0.0323, 0.678, 0.0492, 2, 116.982, ['j-rock', 'japanese indie pop']], ['41SwdQIX8Hy2u6fuEDgvWr', '10%', ['KAYTRANADA', 'Kali Uchis'], 0.0267, 0.794, 0.757, 0.000306, -6.644, 0.123, 0.615, 0.0621, 6, 107.99, ['alternative r&b', 'colombian pop', 'lgbtq+ hip hop', 'escape room', 'indie soul']], ['31B7wLv4yvtjDoTTmbnxeE', 'Jungle', ['Fred again..'], 0.00984, 0.665, 0.727, 0.394, -8.073, 0.0391, 0.397, 0.172, 1, 133.922, ['edm', 'house']], ['6gdDu39yYqPcaTgCwYEW8i', 'Turn On The Lights again.. (feat. Future)', ['Fred again..', 'Swedish House Mafia', 'Future'], 0.0125, 0.683, 0.887, 0.446, -4.944, 0.0497, 0.551, 0.318, 6, 132.007, ['house', 'pop dance', 'rap', 'hip hop', 'atl hip hop', 'trap', 'progressive electro house', 'southern hip hop', 'edm']], ['2i2gDpKKWjvnRTOZRhaPh2', 'Moonlight', ['Kali Uchis'], 0.511, 0.639, 0.723, 0, -6.462, 0.0532, 0.878, 0.167, 7, 136.872, ['colombian pop']], ['3T00mhdOYUuM5yiuPs3xhs', 'San Frandisco', ['Dom Dolla'], 0.0443, 0.779, 0.937, 0.101, -6.269, 0.0496, 0.49, 0.0912, 9, 125.025, ['australian house', 'deep groove house', 'house']], ['0GGfGINoVYiSFXPOjg3RHj', 'Found (feat. Brent Faiyaz)', ['Tems', 'Brent Faiyaz'], 0.625, 0.821, 0.412, 1.64e-05, -7.795, 0.0341, 0.333, 0.0985, 7, 109.977, ['dmv rap', 'rap', 'alte', 'afro r&b', 'r&b', 'nigerian pop']], ['7snnTlaWaN39nfN1PhUaT8', 'Se Voce Nao Quer Passa a Vez', ['Mc Delux', 'DJ Guih Da ZO'], 0.136, 0.956, 0.827, 0.000259, 1.295, 0.222, 0.614, 0.423, 9, 129.949, ['funk mtg', 'rave funk', 'phonk brasileiro', 'funk mandelao']], ['3BbD2sqk7P7Rc9V0KF9o4s', 'My Humps', ['Joshwa', 'Lee Foss'], 0.0418, 0.862, 0.79, 0.00615, -8.412, 0.0672, 0.587, 0.0746, 4, 129.005, ['house', 'uk tech house', 'deep disco house']], ['1zZdEavQr1Vl769ZMqYUvk', 'Up All Night', ['Kenichiro Nishihara', 'SIRUP'], 0.322, 0.649, 0.8, 0.0147, -5.786, 0.0283, 0.561, 0.126, 4, 111.982, ['japanese soul', 'ambeat', 'japanese r&b']], ['5UkoWcFsdWn8NtSAh73Vli', 'Temptation', ['SIDEPIECE'], 0.00665, 0.808, 0.878, 0.0152, -5.282, 0.0948, 0.624, 0.208, 9, 125.011, ['edm']], ['5vOKolqRUv1StcnuMm9wi9', 'I Love You', ['T3'], 0.567, 0.771, 0.331, 2.31e-05, -14.604, 0.04, 0.111, 0.106, 2, 119.957, []], ['56aAUmUVoxpGxaSuvAsXlo', 'Love Shy', ['Kristine Blond', 'James Hype'], 0.0105, 0.809, 0.875, 2.05e-05, -4.084, 0.0663, 0.665, 0.123, 2, 125.024, ['deep groove house', 'pop dance', 'uk dance', 'uk garage', 'speed garage', 'bassline']], ['6Xom58OOXk2SoU711L2IXO', 'Moscow Mule', ['Bad Bunny'], 0.294, 0.804, 0.674, 1.18e-06, -5.453, 0.0333, 0.292, 0.115, 5, 99.968, ['reggaeton', 'trap latino', 'urbano latino']], ['59nOXPmaKlBfGMDeOVGrIK', 'WAIT FOR U (feat. Drake & Tems)', ['Future', 'Drake', 'Tems'], 0.314, 0.463, 0.642, 0, -4.474, 0.34, 0.339, 0.0686, 1, 83.389, ['canadian pop', 'rap', 'toronto rap', 'hip hop', 'atl hip hop', 'trap', 'southern hip hop', 'afro r&b', 'nigerian pop', 'alte', 'canadian hip hop']], ['39sDitIeCMrVX2QyXHY46t', 'Blue Hair', ['TV Girl'], 0.554, 0.751, 0.72, 0.0497, -6.376, 0.0303, 0.884, 0.258, 4, 135.73, []], ['5xd26YqKxCdZzOTiTlOWxg', 'On My Mind', ['Diplo', 'SIDEPIECE'], 0.00157, 0.77, 0.73, 0.674, -6.577, 0.0406, 0.632, 0.217, 7, 123.025, ['pop dance', 'tropical house', 'moombahton', 'electro house', 'edm']], ['5wG3HvLhF6Y5KTGlK0IW3J', 'Trance (with Travis Scott & Young Thug)', ['Metro Boomin', 'Travis Scott', 'Young Thug'], 0.18, 0.571, 0.549, 0, -7.38, 0.404, 0.447, 0.168, 1, 119.497, ['rap', 'melodic rap', 'gangster rap', 'hip hop', 'atl hip hop', 'trap', 'slap house', 'atl trap']], ['4gZYqEH51kKkRV1HmmpqFQ', "Woman's Touch", ['Bushbaby'], 0.000795, 0.738, 0.795, 0.295, -6.315, 0.0494, 0.755, 0.153, 0, 130.003, ['bass house']]]
    user1_medium = to_dataframe(arr2, columns, medium_score)

    arr3 = [['3Xsmpypmc2DcxQBbmnnrB5', '3000 Miles (Baby Baby)', ['Yeek'], 0.00675, 0.643, 0.638, 1.78e-05, -7.753, 0.0424, 0.959, 0.132, 10, 77.505, ['hyperpop', 'indie hip hop']], ['7arX7t70jcRTL4iSYudFJn', 'Give It To Me', ['Pink Sweat$'], 0.109, 0.787, 0.592, 0.000219, -6.01, 0.0453, 0.642, 0.095, 0, 100.02, ['bedroom soul']], ['0lTH3Dt0GlBQhxFHSnaZ7g', 'ROCKSTAR P', ['Baby Keem'], 0.152, 0.948, 0.573, 0, -8.162, 0.113, 0.4, 0.0833, 11, 100.017, ['rap']], ['6u3CPnFMKANYgfdiifFOiJ', 'Gravity (feat. Tyler, The Creator)', ['Brent Faiyaz', 'DJ Dahi', 'Tyler, The Creator'], 0.173, 0.539, 0.615, 0.0056, -8.746, 0.252, 0.493, 0.144, 1, 163.924, ['rap', 'hip hop', 'r&b', 'dmv rap']], ['3yR2neLio3L6Jt6xYP0dVd', 'Please', ['jagger finn'], 0.00863, 0.536, 0.65, 0.85, -8.625, 0.027, 0.367, 0.249, 0, 112.985, ['bedroom pop']], ['435yU2MvEGfDdmbH0noWZ0', 'worldstar money (interlude)', ['Joji'], 0.964, 0.577, 0.387, 0.705, -8.607, 0.274, 0.459, 0.208, 7, 146.565, ['viral pop']], ['0crw01bqnefvDUDjsuKraD', 'After Last Night (with Thundercat & Bootsy Collins)', ['Bruno Mars', 'Anderson .Paak', 'Silk Sonic', 'Thundercat', 'Bootsy Collins'], 0.0297, 0.651, 0.703, 0, -8.958, 0.0816, 0.647, 0.0608, 0, 140.051, ['funk', 'escape room', 'neo soul', 'afrofuturism', 'hip hop', 'soul', 'p funk', 'dance pop', 'pop', 'indie soul']], ['7eWGnKg4B44sbBPpQp4y2c', 'Dragonball Durag', ['Thundercat'], 0.697, 0.648, 0.59, 0.808, -9.664, 0.0942, 0.401, 0.111, 2, 81.045, ['afrofuturism', 'indie soul']], ['1NeLwFETswx8Fzxl2AFl91', 'Something About Us', ['Daft Punk'], 0.44, 0.875, 0.475, 0.72, -12.673, 0.0986, 0.384, 0.046, 9, 99.958, ['electro', 'filter house', 'rock']], ['1c0hsvHLELX6y8qymnpLKL', 'Soltera - Remix', ['Lunay', 'Daddy Yankee', 'Bad Bunny'], 0.361, 0.795, 0.783, 0, -4.271, 0.0432, 0.799, 0.437, 5, 92.01, ['trap latino', 'reggaeton flow', 'latin hip hop', 'reggaeton', 'urbano latino']], ['2gq9iG0maBxkuZI7yfGJuv', 'Overtime', ['Bryson Tiller'], 0.254, 0.657, 0.497, 0, -7.689, 0.112, 0.593, 0.367, 1, 106.049, ['kentucky hip hop', 'r&b', 'rap']], ['0wOtc2nY3NOohp4xSwOyTN', 'We Might Even Be Falling In Love (Duet) - Spotify Singles', ['Victoria Monét', 'Bryson Tiller'], 0.473, 0.731, 0.423, 0.000413, -10.147, 0.0784, 0.78, 0.129, 6, 76.964, ['rap', 'alternative r&b', 'r&b', 'kentucky hip hop']], ['58ge6dfP91o9oXMzq3XkIS', '505', ['Arctic Monkeys'], 0.00237, 0.52, 0.852, 5.79e-05, -5.866, 0.0543, 0.234, 0.0733, 0, 140.267, ['garage rock', 'modern rock', 'permanent wave', 'rock', 'sheffield indie']], ['3Q4gttWQ6hxqWOa3tHoTNi', 'Not You Too (feat. Chris Brown)', ['Drake', 'Chris Brown'], 0.342, 0.458, 0.452, 1.94e-05, -9.299, 0.047, 0.316, 0.0703, 9, 86.318, ['canadian pop', 'rap', 'toronto rap', 'hip hop', 'r&b', 'canadian hip hop']], ['1DwabqQdr9uWTx40ByKH3g', 'All Mine', ['jagger finn'], 0.253, 0.681, 0.472, 0.756, -8.348, 0.0463, 0.388, 0.169, 7, 81.972, ['bedroom pop']], ['2xB46Bj9HZ4cr058yN4Pla', 'Secrets', ['A Boogie Wit da Hoodie'], 0.118, 0.539, 0.648, 0, -6.008, 0.0619, 0.403, 0.362, 4, 91.066, ['melodic rap', 'rap', 'trap']], ['0PZRlp2wrQDausamlangtw', 'Under', ['Mac Ayres', 'Jordan Robertson'], 0.568, 0.732, 0.366, 0.000593, -8.601, 0.0568, 0.544, 0.194, 3, 85.909, ['chill r&b']], ['1xLs8Mu1QEVbGCpyHQ2r2U', 'Why', ['Dominic Fike'], 0.342, 0.684, 0.856, 0.00728, -2.676, 0.0411, 0.343, 0.109, 6, 117.027, ['alternative pop rock']], ['3siyfhqP2BSRciLSbwGpzR', 'Whats For Dinner?', ['Dominic Fike'], 0.65, 0.855, 0.469, 2.48e-05, -4.965, 0.0769, 0.305, 0.108, 4, 91.985, ['alternative pop rock']], ['7nc7mlSdWYeFom84zZ8Wr8', 'Tell Em', ['Cochise', '$NOT'], 0.103, 0.672, 0.717, 0, -7.476, 0.226, 0.473, 0.398, 5, 157.905, ['cloud rap', 'underground hip hop', 'florida rap', 'plugg', 'pluggnb', 'aesthetic rap']], ['0tdA3tsJ4n6rJuiId3KrOP', 'cz', ['Mk.gee'], 0.598, 0.734, 0.552, 0.835, -7.595, 0.123, 0.714, 0.107, 2, 91.064, []], ['5IUtvfNvOyVYZUa6AJFrnP', 'Spicy (feat. Post Malone)', ['Ty Dolla $ign', 'Post Malone'], 0.143, 0.782, 0.51, 0, -5.724, 0.0419, 0.118, 0.115, 4, 99.993, ['dfw rap', 'trap soul', 'rap', 'melodic rap', 'hip hop', 'pop rap', 'trap', 'southern hip hop', 'r&b', 'pop']], ['6ptijxei8gTrFuIob3LyJW', 'whiskey', ['Julian Skiboat'], 0.576, 0.921, 0.478, 1.72e-05, -7.467, 0.037, 0.692, 0.306, 11, 98.961, []], ['6M4jdLdM7wLGungMV9gsCS', 'Smokin Out The Window', ['Bruno Mars', 'Anderson .Paak', 'Silk Sonic'], 0.0558, 0.627, 0.618, 0, -8.529, 0.0437, 0.848, 0.351, 2, 82.03, ['escape room', 'neo soul', 'hip hop', 'dance pop', 'pop', 'indie soul']], ['6j4j6iRR0Ema531o5Yxr2T', 'Memories', ['Eden Prince', 'Nonô'], 0.161, 0.909, 0.651, 0.0437, -5.672, 0.105, 0.895, 0.0923, 11, 124.026, ['uk dance', 'dance pop']], ['4bbJw6usZSqkcmnqjSIOWx', 'Put On A Smile', ['Bruno Mars', 'Anderson .Paak', 'Silk Sonic'], 0.0471, 0.548, 0.627, 0, -8.949, 0.0409, 0.493, 0.285, 9, 143.994, ['escape room', 'neo soul', 'hip hop', 'dance pop', 'pop', 'indie soul']], ['5PvVkf1Yuq3XyMqqjPiKPd', 'Been Away', ['Brent Faiyaz'], 0.477, 0.602, 0.5, 5.69e-05, -9.377, 0.298, 0.395, 0.295, 8, 132.798, ['dmv rap', 'r&b', 'rap']], ['7ACT6YaXbYvl7hRWEOOEHQ', 'Double Negative (Skeleton Milkshake)', ['Dominic Fike'], 0.00383, 0.689, 0.858, 0, -2.868, 0.161, 0.643, 0.129, 4, 153.977, ['alternative pop rock']], ['3dPtXHP0oXQ4HCWHsOA9js', '夜に駆ける', ['YOASOBI'], 0.00231, 0.67, 0.874, 1.72e-05, -5.221, 0.0305, 0.789, 0.3, 8, 130.041, ['j-pop', 'japanese teen pop']], ['6cLxofxCjrFpQYtifjK5Vf', 'Only in the West', ['Yeek'], 0.0723, 0.954, 0.537, 0.0569, -6.871, 0.0901, 0.525, 0.218, 7, 116.964, ['hyperpop', 'indie hip hop']], ['6OTKVgVpVaVjhRLYizPJKA', 'Just for me', ['PinkPantheress'], 0.653, 0.796, 0.625, 0.00348, -7.218, 0.0645, 0.641, 0.187, 0, 133.035, []], ['17q2kvipvoCK4lX9n21zht', 'SLICE INTERLUDE', ['Baby Keem'], 0.108, 0.659, 0.479, 0, -7.162, 0.138, 0.472, 0.311, 0, 142.028, ['rap']], ['34D6FJysnQioVingDKufuf', 'N 2 Deep', ['Drake', 'Future'], 0.0518, 0.507, 0.429, 0, -7.726, 0.326, 0.0744, 0.109, 2, 91.0, ['canadian pop', 'rap', 'toronto rap', 'hip hop', 'atl hip hop', 'trap', 'southern hip hop', 'canadian hip hop']], ['0Yz3F0UGDibDe8uU69zmjn', 'And July', ['HEIZE', 'DEAN', 'dj friz'], 0.0401, 0.734, 0.852, 0, -3.85, 0.0916, 0.787, 0.196, 2, 98.504, ['korean pop', 'korean r&b', 'k-pop']], ['6FBzhcfgGacfXF3AmtfEaX', 'C U Girl', ['Steve Lacy'], 0.663, 0.414, 0.473, 0.0523, -8.911, 0.116, 0.409, 0.128, 11, 100.0, ['afrofuturism']], ['4NczzeHBQPPDO0B9AAmB8d', 'Assumptions', ['Sam Gellaitry'], 0.123, 0.639, 0.815, 0.00113, -4.718, 0.221, 0.443, 0.193, 11, 126.055, ['future bass', 'scottish electronic', 'vapor twitch']], ['7qwt4xUIqQWCu1DJf96g2k', 'Hello?', ['Clairo', 'Rejjie Snow'], 0.641, 0.839, 0.473, 0.000459, -7.835, 0.0607, 0.0644, 0.0939, 10, 98.732, ['indie pop', 'bedroom pop', 'irish hip hop', 'pop', 'indie soul']], ['6Fc9IYSkbmVwv4dlzn8hJb', 'Overrated', ['Blxst'], 0.625, 0.842, 0.545, 0.00164, -5.993, 0.172, 0.466, 0.103, 1, 97.897, ['pop rap', 'westcoast flow']], ['6mtjo5kIHKlboGE84lf1FJ', 'Need It', ['Migos', 'YoungBoy Never Broke Again'], 0.0313, 0.852, 0.745, 0, -3.667, 0.171, 0.897, 0.0824, 11, 148.007, ['baton rouge rap', 'rap', 'hip hop', 'atl hip hop', 'trap']], ['3a3dQOO19moXPeTt2PomoT', 'What You Heard', ['Sonder'], 0.546, 0.429, 0.371, 5.71e-06, -9.017, 0.0327, 0.135, 0.104, 2, 137.134, []], ['248OFOZef6ShXv6DGgbnxU', 'Saved', ['Khalid'], 0.189, 0.739, 0.448, 0, -10.28, 0.138, 0.553, 0.118, 10, 81.044, ['pop', 'pop r&b']], ['6DXRUAHQTwJuCXuhXyRU53', 'peas', ['boylife'], 0.303, 0.249, 0.342, 0.0113, -10.283, 0.0389, 0.126, 0.16, 4, 64.335, ['modern indie pop']], ['1dIWPXMX4kRHj6Dt2DStUQ', 'Chosen (feat. Ty Dolla $ign)', ['Blxst', 'Tyga', 'Ty Dolla $ign'], 0.336, 0.571, 0.767, 0, -5.16, 0.287, 0.605, 0.0809, 2, 93.421, ['trap soul', 'rap', 'hip hop', 'pop rap', 'trap', 'southern hip hop', 'westcoast flow', 'r&b']], ['6VFKuuqSbA1GpMEosUgTwQ', 'Nobody But You', ['Sonder', 'Jorja Smith'], 0.742, 0.637, 0.299, 0, -11.314, 0.0366, 0.364, 0.359, 7, 83.872, ['uk contemporary r&b', 'r&b']], ['7rDvD5WGtchCGkfQOpIb8C', "I'm Trying", ['Yeek', 'Dominic Fike'], 0.541, 0.804, 0.429, 0.00106, -6.712, 0.0714, 0.768, 0.103, 1, 96.957, ['indie hip hop', 'hyperpop']], ['1X8frTk9CAPyDiJOOwMo2z', 'チューリップ', ['indigo la End'], 0.0451, 0.657, 0.761, 0, -3.857, 0.0304, 0.556, 0.321, 6, 125.97, ['j-rock', 'japanese indie pop']], ['0trHOzAhNpGCsGBEu7dOJo', 'N.Y. State of Mind', ['Nas'], 0.0394, 0.665, 0.91, 0, -4.682, 0.223, 0.887, 0.227, 6, 84.099, ['conscious hip hop', 'east coast hip hop', 'gangster rap', 'hardcore hip hop', 'hip hop', 'queens hip hop', 'rap']], ['1S4sxNGmCASkyRI08YhLtT', "Gettin' Old", ['6LACK'], 0.4, 0.743, 0.4, 0.000311, -12.369, 0.0625, 0.229, 0.111, 11, 80.01, ['atl hip hop', 'melodic rap', 'r&b', 'rap', 'trap']], ['0b6l1obmzq1YrqiKBVLtIo', 'FRONTAL LOBE MUZIK', ['Daniel Caesar', 'Pharrell Williams'], 0.439, 0.681, 0.568, 0.000173, -6.264, 0.136, 0.301, 0.446, 11, 140.004, ['canadian contemporary r&b', 'dance pop', 'pop']], ['2sXf2JdbB2GlNju00kw9WE', 'Skate', ['Bruno Mars', 'Anderson .Paak', 'Silk Sonic'], 0.037, 0.708, 0.598, 0, -8.365, 0.0291, 0.698, 0.17, 5, 112.027, ['escape room', 'neo soul', 'hip hop', 'dance pop', 'pop', 'indie soul']]]
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


    user1_numbers = user1_master.iloc[:, 3:13]
    user2_numbers = user2_master.iloc[:, 3:13]

    #Combining all songs for umap reduction to better compare distances
    combined = scale((pd.concat([user1_numbers,user2_numbers],axis=0)))


    umap_model = umap.UMAP()
    umap_embedding = umap_model.fit_transform(combined)

    umap1 = umap_embedding[:len(user1_master)]
    umap2 = umap_embedding[len(user1_master):]

    #Silhuotette method to determine optimal number of clusters
    ran = [x for x in range(2,11)]
    silhouette_scores = []
    for clusters in ran:
        model = KMeans(n_clusters=clusters)
        cluster_labels = model.fit_predict(umap1)
        silhouette_avg = silhouette_score(umap1, cluster_labels)
        silhouette_scores.append(silhouette_avg * umap1.shape[0])


    user1_cluster_num = np.where(silhouette_scores == max(silhouette_scores))
    user1_cluster_num = ran[user1_cluster_num[0][0]]


    model = KMeans(n_clusters=user1_cluster_num)
    clusters = model.fit_predict(umap1)


    ran = [x for x in range(2,11)]
    silhouette_scores = []
    for clusters in ran:
        model = KMeans(n_clusters=clusters)
        cluster_labels = model.fit_predict(umap2)
        silhouette_avg = silhouette_score(umap2, cluster_labels)
        silhouette_scores.append(silhouette_avg * umap2.shape[0])


    user2_cluster_num = np.where(silhouette_scores == max(silhouette_scores))
    user2_cluster_num = ran[user2_cluster_num[0][0]]


    model = KMeans(n_clusters=user2_cluster_num)
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
    user1_recs['distance_rank'] = [x for x in range(1,len(user1_recs) + 1)]


    clusters2 = []
    for i in range(len(user2_means.cluster_centers_)):
        cluster_center = i
        cluster_name = "cluster" + str(i) + "_distance"
        clusters2.append(cluster_name)
        user2_recs[cluster_name] = calculate_distance(i,user2_means, umap1)

    user2_recs['min_value'] = user2_recs[[cluster for cluster in clusters2]].min(axis=1)

    user2_recs = user2_recs.sort_values(by='min_value')
    user2_recs['distance_rank'] = [x for x in range(1,len(user2_recs) + 1)]


    #Genres


    user1_genres = get_genre(user2_recs)
    user2_genres = get_genre(user1_recs)

    user1_genres, user2_genres = user1_genres.align(user2_genres, fill_value=0.0, axis=1)


    #Getting term frequency of genres across all songs to get overall genres for a user
    user1_taste = get_taste(user2_recs['genres'])
    user2_taste = get_taste(user1_recs['genres'])

    user1_taste, user2_taste = user1_taste.align(user2_taste, fill_value=0.0, axis=1)


    user1_taste = np.array(user1_taste)[0]
    user2_taste = np.array(user2_taste)[0]


    user1_genres = np.array(user1_genres)
    user2_genres = np.array(user2_genres)

    #Using cosine similarity to determine how closely each song matches the other user's overall genre taste
    user2_recs['genre_similarity'] = calculate_genre_distance(user2_taste, user1_genres)
    user1_recs['genre_similarity'] = calculate_genre_distance(user1_taste, user2_genres)


    user2_recs = user2_recs.sort_values(by='genre_similarity', ascending=False)
    user2_recs['genre_rank'] = get_ranks(user2_recs)
    user1_recs = user1_recs.sort_values(by='genre_similarity', ascending=False)
    user1_recs['genre_rank'] = get_ranks(user1_recs)


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

#Using StandardScaler for a dataframe
def scale(nums):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(nums)
    return scaled

#Assigning clusters to rows of a dataframe
def cluster_assign(nums, master, clusters):
    #Clustering
    kmeans = KMeans(n_clusters = clusters)
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

def get_ranks(df):
    rows = len(df)
    ranks = [x for x in range(1, rows + 1)]
    return ranks

#Combining the three ranks with how much the original user likes their own song being the half as important as the other two
def combine_ranks(recs):
    return 0.4 * recs['distance_rank'] + 0.2 * recs['score_rank'] + 0.4 * recs['genre_rank']

if __name__ == "__main__":
    app.run(port=5001, debug=True)
