# AI music maker
This project is a music maker that has support with AI

### Project scope

* AI music Composer studio: based on inputs such as artist, genera, or mood, it would create a new track for you to listen to. it also has intractive fetures for you to change the tempo, mood, instruments and other musical parematers of th song. You could also combine multple songs toghter to create a new song. it will be downloadable in a WAV format 
* Music insight: Using your music lisining histroy it would  act as a listener insights engine, analyzing user listening history to provide personalized song recommendations, playlist generation, and era-specific suggestions. Recommendations will consider genre, mood, tempo, instruments, artist preferences, and listening patterns.
* Mood and instrument analyzer: using an inputted song, it will automatically detect song mood and instruments used in both dataset and AI-generated tracks. Mood classifications include happy, sad, chill, energetic, while instruments include piano, guitar, drums, synthesizer, and more. This information will support music recommendations, playlist curation, and remixing functions.
* Adaptive feedback: Using user feedback on ai genarated musinc, it would dynamically adapt outputs. Users can rate tracks, provide textual feedback, and adjust musical elements such as tempo, instruments, beats, or mood. The system will use this data to refine music generation models over time, improving personalization and AI creativity.
* Intregrated remix and mashup studio: the AI music maker would allow users to merge multiple songs based on artist, genre, mood, or era, producing harmonically coherent remixes and mashups. Users can add instruments, adjust tempo, and combine melodies, with all outputs available for download in WAV format. This feature will also support playlist generation based on mood or listening trends, allowing users to explore creative combinations of AI-generated and dataset tracks.


### coloums used for each feture

AI music maker: from the music info dataset: name, artist, genera, dancablity, enegery, key, loudness, mode, instruemntials, valence. tempo

Music insight: from user histroy, track id, times played

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
