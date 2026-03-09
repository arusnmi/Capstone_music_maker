import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="AI Music Composer & Listener Insight Platform", layout="wide")

# Title
st.title("🎵 AI Music Composer & Listener Insight Platform")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "AI Composer Studio", "Recommendations", "Mood & Instrument Analyzer", "Remix & Mashup Studio", "Feedback", "Analytics Dashboard"])

if page == "Home":
    st.header("Welcome to the AI Music Composer & Listener Insight Platform")
    st.write("""
    This platform offers:
    - AI Music Composer Studio: Generate music based on your preferences
    - Personalized Music Recommendations: Get song suggestions based on your listening history
    - Mood & Instrument Analyzer: Analyze the mood and instruments in tracks
    - Interactive Feedback & Adaptive Composition: Rate tracks and improve the system
    - Remix & Mashup Studio: Create blends of multiple songs
    """)
    st.image("https://via.placeholder.com/800x400.png?text=Music+Platform+Banner", use_column_width=True)

elif page == "AI Composer Studio":
    st.header("🎼 AI Composer Studio")
    st.write("Generate music based on your preferences")

    col1, col2 = st.columns(2)
    with col1:
        genre = st.selectbox("Select Genre", ["Pop", "Rock", "Jazz", "Classical", "Electronic"])
        mood = st.selectbox("Select Mood", ["Happy", "Sad", "Energetic", "Calm"])
    with col2:
        tempo = st.slider("Tempo (BPM)", 60, 200, 120)
        duration = st.slider("Duration (seconds)", 30, 300, 120)

    instruments = st.multiselect("Select Instruments", ["Piano", "Guitar", "Drums", "Bass", "Violin", "Synthesizer"])

    if st.button("Generate Music"):
        st.success("Music generated! (Placeholder - actual generation would happen here)")
        st.audio("https://www.soundjay.com/misc/sounds/bell-ringing-05.wav")  # Placeholder audio

elif page == "Recommendations":
    st.header("🎧 Personalized Music Recommendations")
    st.write("Get song suggestions based on your listening history")

    user_id = st.text_input("Enter User ID", "user123")

    if st.button("Get Recommendations"):
        # Placeholder recommendations
        recommendations = [
            {"title": "Song A", "artist": "Artist A", "genre": "Pop"},
            {"title": "Song B", "artist": "Artist B", "genre": "Rock"},
            {"title": "Song C", "artist": "Artist C", "genre": "Jazz"}
        ]
        for rec in recommendations:
            st.write(f"**{rec['title']}** by {rec['artist']} ({rec['genre']})")

elif page == "Mood & Instrument Analyzer":
    st.header("🎭 Mood & Instrument Analyzer")
    st.write("Analyze the mood and instruments in tracks")

    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file)
        if st.button("Analyze"):
            # Placeholder analysis
            st.write("**Detected Mood:** Happy")
            st.write("**Detected Instruments:** Piano, Guitar")
            st.write("**Tempo:** 120 BPM")
            st.write("**Energy:** High")

elif page == "Remix & Mashup Studio":
    st.header("🔀 Remix & Mashup Studio")
    st.write("Create blends of multiple songs")

    col1, col2 = st.columns(2)
    with col1:
        track1 = st.selectbox("Select Track 1", ["Song A", "Song B", "Song C"])
        blend_ratio = st.slider("Blend Ratio", 0.0, 1.0, 0.5)
    with col2:
        track2 = st.selectbox("Select Track 2", ["Song D", "Song E", "Song F"])
        tempo_adjust = st.slider("Tempo Adjustment", 0.5, 2.0, 1.0)

    if st.button("Create Mashup"):
        st.success("Mashup created! (Placeholder)")
        st.audio("https://www.soundjay.com/misc/sounds/bell-ringing-05.wav")  # Placeholder

elif page == "Feedback":
    st.header("📝 Interactive Feedback")
    st.write("Rate tracks and provide feedback to improve the system")

    track_to_rate = st.selectbox("Select Track to Rate", ["Generated Track 1", "Generated Track 2", "Mashup 1"])

    rating = st.slider("Rating (1-5)", 1, 5, 3)
    comments = st.text_area("Comments (optional)")

    if st.button("Submit Feedback"):
        st.success("Feedback submitted! Thank you for your input.")

elif page == "Analytics Dashboard":
    st.header("📊 Analytics Dashboard")
    st.write("Visualize user engagement, popular moods, instruments, and track trends")

    # Placeholder charts
    fig, ax = plt.subplots()
    genres = ['Pop', 'Rock', 'Jazz', 'Classical', 'Electronic']
    counts = [20, 15, 10, 8, 12]
    ax.bar(genres, counts)
    ax.set_title("Genre Distribution")
    st.pyplot(fig)

    st.write("**Top Moods:** Happy (40%), Energetic (30%), Calm (20%), Sad (10%)")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("© 2024 AI Music Composer Platform")
