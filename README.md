# Spotify Joint Playlist Project

Created an app that will create a joint playlist between two users on Spotify, creating a playlist with songs that both people should enjoy.

Link to a walkthrough of the website: https://youtu.be/MVTnEEGDoyo

## Prerequisites

- Python (version 3.6 or above) installed on your system
- pip package manager

## Configuration

<u>Install Dependencies<u>
- pip install flask spotipy pandas numpy umap scikit-learn

Setup variables
- Setup SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET in a config.py file with your own Spotify client credentials

## Project Overview

This project uses the Spotify API to retrieve user data on short, medium, and long-term songs that they listen to. The goal of the project is to recommend the most fitting songs from one user's library to the other user. The recommendation process involves comparing song metrics obtained from the Spotify API, as well as comparing the genres associated with each song to a user's overall genres.

The main objectives of the project are as follows:

- Retrieve user data on short, medium, and long term songs from the Spotify API.
- Compare song metrics, such as acousticness, energy, etc.
- Compare the genres associated with each song to the user's overall genres.
- Generate recommendations based on the comparison results.
- Creates a playlist for the user on Spotify and gives some insight into how the songs were chosen.

## Data Collection and Preproccessing

- Setup Spotify API
- Extract neccesary data from dicitionaries
  - Example dictionary for a song: {
    "album": {
        "album_type": "ALBUM",
        "artists": [{
            "external_urls": {
                "spotify": "https://open.spotify.com/artist/ID"
            },
            "name": "Slum Village",
            ...
        }],
        "available_markets": ["AD", "AE", ...],
        "external_urls": {
            "spotify": "https://open.spotify.com/album/ID"
        },
        "images": [{
            "url": "https://i.scdn.co/image/IMAGE_ID",
            ...
        }],
        "name": "Detroit Deli (A Taste Of Detroit)",
        ...
    },
    "artists": [{
        "external_urls": {
            "spotify": "https://open.spotify.com/artist/ID"
        },
        "name": "Slum Village",
        ...
    }, ...],
    "available_markets": ["AD", "AE", ...],
    "duration_ms": 225400,
    "explicit": true,
    "external_ids": {
        "isrc": "USCA20400008"
    },
    "external_urls": {
        "spotify": "https://open.spotify.com/track/ID"
    },
    ...
}

- Final dataframe, added score to signify how much the user likes the song

![Master DataFrame](imgs/master_df.png)


## Songs that one user likes that the other would like

### Song Metrics
- Scaled song metrics
- UMAP dimensionality reduction on features

![Combined Features](imgs/together.png)
- KMeans clustering and Silhoutte Method to determine optimal number of clusters

![User1 Silhouette](imgs/silhouette1.png)
![User1 Kmeans](imgs/kmeans1.png)

![User2 Silhouette](imgs/silhouette2.png)
![User2 Kmeans](imgs/kmeans2.png)

- Comapre cluster centers from a user to each song from the other user. Euclidean distance to determine most audibly similar

### Genres
- Term Frequency of all genres among all a users songs to represent their taste

![Combined Features](imgs/user1_taste.png)

Learned that escape room is a genre that I really enjoy!

![Combined Features](imgs/user2_taste.png)

- Example genre values of a song after applying TF-IDF vectorizer

![Combined Features](imgs/song.png)

Really enjoy this song, makes sense

- Compare user's taste to each songs genres. Use cosine similarity to determine similarity to a users genre prefernces

## Generating recommendations

- Create a holistic score representing how good of a recommendation it is for each song based on three things.
  - Personal score
    - How much the original user likes their own song
  - Metric score
    - How audibly similar a song is to other songs that the user likes
  - Genre score
    - How similar are the genres of a song to a users overall genre preferences


