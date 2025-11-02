# app.py
from dotenv import load_dotenv
load_dotenv()
import os
import io
import csv
import pickle
from pathlib import Path
import random

import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

st.set_page_config(page_title="Music Recommender", page_icon="🎧", layout="wide")
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {padding-top: 2rem; padding-bottom: 2.5rem;}
    div[data-testid="stMetricValue"] { font-size: 1.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")

if not CLIENT_ID or not CLIENT_SECRET:
    st.warning(
        "Set **SPOTIPY_CLIENT_ID** and **SPOTIPY_CLIENT_SECRET** as environment variables "
        "(don’t hardcode secrets in code or git)."
    )

@st.cache_resource(show_spinner=False)
def get_spotify():
    auth = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    return spotipy.Spotify(client_credentials_manager=auth)

sp = get_spotify()

BASE_DIR = Path(__file__).resolve().parent

def _load_pickle(fname):
    path = BASE_DIR / fname
    if not path.exists():
        st.error(f"Missing file: `{fname}` in `{BASE_DIR}`")
        st.stop()
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data(show_spinner=True)
def load_data():
    df = _load_pickle("df.pkl")
    sim = _load_pickle("similarity.pkl")
    if "song" not in df.columns:
        st.error("`df.pkl` must include a 'song' column.")
        st.stop()
    if "artist" not in df.columns:
        df["artist"] = ""
    df["song"] = df["song"].astype(str)
    df["artist"] = df["artist"].astype(str)
    df["display"] = (df["song"] + " — " + df["artist"]).str.strip(" —")
    return df, sim

music, similarity = load_data()

@st.cache_data(show_spinner=False)
def search_track(song, artist="", limit=1):
    try:
        q = f"track:{song}"
        if artist.strip():
            q += f" artist:{artist}"
        res = sp.search(q=q, type="track", limit=limit)
        items = res.get("tracks", {}).get("items", [])
        return items[0] if items else None
    except Exception as e:
        st.sidebar.info(f"Spotify search issue for '{song}': {e}")
        return None

@st.cache_data(show_spinner=False)
def album_image(track):
    if not track:
        return "https://i.postimg.cc/0QNxYz4V/social.png"
    imgs = track.get("album", {}).get("images", [])
    return imgs[0]["url"] if imgs else "https://i.postimg.cc/0QNxYz4V/social.png"

@st.cache_data(show_spinner=False)
def audio_features(track_id):
    try:
        feats = sp.audio_features([track_id])
        if feats and feats[0]:
            wanted = [
                "danceability", "energy", "valence", "tempo",
                "acousticness", "instrumentalness"
            ]
            return {k: feats[0].get(k) for k in wanted}
    except Exception:
        pass
    return None

def _index_of_song(title):
    matches = music.index[music["song"] == str(title)]
    return int(matches[0]) if len(matches) else None

def recommend(seed_title, k=5):
    idx = _index_of_song(seed_title)
    if idx is None:
        return []
    ranked = sorted(enumerate(similarity[idx]), key=lambda x: x[1], reverse=True)
    return [i for i, _ in ranked[1 : k + 1]]

with st.sidebar:
    st.title("Options")
    top_k = st.slider("How many recommendations?", 3, 10, 6, 1)
    st.divider()
    st.subheader("Filter by Artist (optional)")
    artists = sorted({a for a in music["artist"] if a.strip()})
    artist_filter = st.selectbox("Only show this artist", ["(no filter)"] + artists)
    st.divider()
    st.subheader("Extras")
    explain_model = st.checkbox("Show how this model works", value=True)

left, _ = st.columns([0.6, 0.05])
with left:
    st.markdown("## 🎵 Music Recommender — *Content-based*")
    st.caption("Pick a song to get similar tracks.")

if "favorites" not in st.session_state:
    st.session_state.favorites = []

st.divider()

options = music.index.to_list()
def fmt(i): return music.at[i, "display"]
sel_idx = st.selectbox("Search or select a seed song:", options, format_func=fmt, index=0)
seed_song = music.at[sel_idx, "song"]
seed_artist = music.at[sel_idx, "artist"]

c1, c2 = st.columns([0.18, 0.18])
with c1:
    run_btn = st.button("🔍 Recommend")
with c2:
    st.caption("Tip: Click album covers for Spotify links when available.")

st.subheader("Track")
seed_track = search_track(seed_song, seed_artist)
seed_img = album_image(seed_track)

s1, s2 = st.columns([0.15, 0.5])
with s1:
    st.image(seed_img, use_container_width=True)
with s2:
    st.markdown(f"**{seed_song}**  \n{seed_artist or ''}")
    if seed_track:
        ext = seed_track.get("external_urls", {}).get("spotify")
        if ext:
            st.markdown(f"[Open in Spotify]({ext})")
    if seed_track:
        feats = audio_features(seed_track["id"])
        if feats:
            a, b, c = st.columns(3)
            with a:
                st.metric("Danceability", f"{feats['danceability']:.2f}")
                st.metric("Energy", f"{feats['energy']:.2f}")
            with b:
                st.metric("Valence", f"{feats['valence']:.2f}")
                st.metric("Tempo (BPM)", f"{feats['tempo']:.0f}")
            with c:
                st.metric("Acousticness", f"{feats['acousticness']:.2f}")
                st.metric("Instrumentalness", f"{feats['instrumentalness']:.2f}")

st.divider()

def passes_artist(i):
    if artist_filter == "(no filter)":
        return True
    return music.at[i, "artist"].strip() == artist_filter

def render_card(i):
    title = music.at[i, "song"]
    artist = music.at[i, "artist"]
    track = search_track(title, artist)
    img = album_image(track)
    link = track.get("external_urls", {}).get("spotify") if track else None
    preview = track.get("preview_url") if track else None
    popularity = track.get("popularity") if track else None
    with st.container():
        c1, c2 = st.columns([0.18, 0.82])
        with c1:
            if link:
                st.markdown(f"[![cover]({img})]({link})")
            else:
                st.image(img, use_container_width=True)
        with c2:
            st.markdown(f"### {title}")
            st.markdown(artist or "")
            meta_bits = []
            if popularity is not None:
                meta_bits.append(f"Popularity: **{popularity}**")
            if track and track.get("explicit"):
                meta_bits.append("Explicit")
            if meta_bits:
                st.caption(" • ".join(meta_bits))
            if preview:
                st.audio(preview)
            if track:
                feats = audio_features(track["id"])
                if feats:
                    A, B, C = st.columns(3)
                    with A:
                        st.metric("Danceability", f"{feats['danceability']:.2f}")
                        st.metric("Energy", f"{feats['energy']:.2f}")
                    with B:
                        st.metric("Valence", f"{feats['valence']:.2f}")
                        st.metric("Tempo", f"{feats['tempo']:.0f} BPM")
                    with C:
                        st.metric("Acoustic", f"{feats['acousticness']:.2f}")
                        st.metric("Instrumental", f"{feats['instrumentalness']:.2f}")
            if link:
                st.caption(f"[Open in Spotify]({link})")

should_run = run_btn
if should_run:
    recs = recommend(seed_song, k=top_k)
    if artist_filter != "(no filter)":
        recs = [i for i in recs if passes_artist(i)]
    if not recs:
        st.info("No recommendations found with current filters.")
    else:
        st.subheader("Recommended for you")
        for i in recs:
            render_card(i)
            st.divider()

if explain_model:
    with st.expander("How this works (short & clear)"):
        st.markdown(
            """
**Engine:** Content-based recommender using a precomputed **similarity matrix** between songs.  
Pick a seed track, then we return the most similar rows (skipping the seed).

**Spotify data:** Album art, preview audio, popularity, and audio features are fetched to enhance the UI.  
They **do not** change the precomputed ranking.
"""
        )