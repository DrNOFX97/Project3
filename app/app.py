import os
import re
import time
import wave
import json
from uuid import uuid4
from tqdm.auto import tqdm
import streamlit as st
import pinecone
import yt_dlp
from vosk import Model, KaldiRecognizer
from flask import Flask, request, jsonify, render_template

# Environment setup
os.environ['PATH'] += os.pathsep + '/usr/local/bin'  # Adjust FFmpeg path as necessary

# Access secrets (assuming these are set in Streamlit secrets)
openai_api_key = st.secrets["OPENAI_API_KEY"]
langchain_api_key = st.secrets["LANGCHAIN_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]

# Initialize Pinecone client
pc = pinecone.Pinecone(api_key=pinecone_api_key, environment="us-east1-gcp")

# Initialize Flask app
app = Flask(__name__)
app.debug = False  # Disable debug mode

# Function to check if input URL is a valid YouTube URL
def is_valid_youtube_url(url):
    pattern = r'^(https?://)?(www\.)?(youtube\.com|youtu\.?be)/.+$'
    return bool(re.match(pattern, url))

# Function to download and transcribe audio from YouTube video
def transcribe_youtube_video(url):
    try:
        # Download video and extract audio
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'outtmpl': 'audio.wav',
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        audio_file = 'audio.wav'

        # Transcribe audio using Vosk
        model = Model("model")
        rec = KaldiRecognizer(model, 16000)

        wf = wave.open(audio_file, "rb")
        results = []
        total_frames = wf.getnframes()

        with tqdm(total=total_frames, desc="Transcribing") as pbar:
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    part_result = json.loads(rec.Result())
                    results.append(part_result['text'])
                pbar.update(4000)

        part_result = json.loads(rec.FinalResult())
        results.append(part_result['text'])
        transcription = " ".join(results)
        return transcription

    except yt_dlp.DownloadError as e:
        st.error(f"Error downloading video: {e}")
        return None

    except Exception as e:
        st.error(f"Error processing video: {e}")
        return None

# Route to render the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle user queries
@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['query']
    response = agent(user_input)  # Define 'agent' function to handle user queries
    conversational_memory.add_message(user_input, response['output'])  # Store conversation history
    return jsonify({'response': response['output']})

# Main Streamlit app
def main():
    st.title("YouTube Video Transcription and Similarity Search")

    # User input for YouTube video URL
    video_url = st.text_input("Enter the YouTube video URL:", key="video_url")

    if st.button("Transcribe and Index"):
        st.info("Downloading video and transcribing audio... This may take some time.")

        # Download and transcribe YouTube video
        transcription = transcribe_youtube_video(video_url)

        if transcription:
            st.success("Transcription complete.")
            st.text_area("Transcription", value=transcription, height=200)

            # Process and index transcription
            chunks = text_splitter.split_text(transcription)[:3]  # Example chunking

            if chunks:
                for chunk in chunks:
                    # Generate a unique ID for each chunk, embed the document, and add metadata
                    index.upsert(vectors=[(str(uuid4()), embed.embed_document(chunk), {'text': chunk})])
                st.success("Text chunks indexed successfully.")
            else:
                st.error("Transcription failed. Please check the YouTube URL and try again.")
        else:
            st.warning("Please enter a valid YouTube video URL.")

    # User input for similarity search query
    query = st.text_input("Enter your query:", key="query")

    if st.button("Search"):
        if query:
            # Perform similarity search
            results = vectorstore.similarity_search(query, k=3)

            # Display search results
            if results:
                st.subheader("Top 3 Most Relevant Documents:")
                for result in results:
                    st.write(f"- Document ID: {result.id}, Score: {result.score}")
            else:
                st.warning("No results found.")
        else:
            st.warning("Please enter a query.")

if __name__ == '__main__':
    main()
