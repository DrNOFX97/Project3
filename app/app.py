import os
import re
import time
import subprocess
import wave
import json
from uuid import uuid4
from tqdm.auto import tqdm
import streamlit as st
import pinecone
import yt_dlp
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from vosk import Model, KaldiRecognizer
from flask import Flask, request, jsonify, render_template

# Environment setup
os.environ['PATH'] += os.pathsep + '/usr/local/bin'

# Access secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]
langchain_api_key = st.secrets["LANGCHAIN_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]

# Initialize Pinecone client
pc = pinecone.Pinecone(api_key=pinecone_api_key, environment="us-east1-gcp")

# Initialize OpenAI embeddings
model_name = 'text-embedding-ada-002'
embed = OpenAIEmbeddings(model=model_name, openai_api_key=openai_api_key)

# Define index name and specifications
index_name = 'langchain-retrieval-augmentation'
spec = pinecone.ServerlessSpec(cloud="aws", region="us-east-1")

# Check if index exists, create if not
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(index_name, dimension=1536, metric='dotproduct', spec=spec)
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# Connect to Pinecone index
index = pc.Index(index_name)

# Initialize Pinecone vector store
text_field = 'text'
vectorstore = PineconeVectorStore(index, embed, text_field)

# Initialize Flask app
app = Flask(__name__)
app.debug = False  # Disable debug mode

# Function to check if input URL is a valid YouTube URL
def is_valid_youtube_url(url):
    pattern = r'^(https?://)?(www\.)?(youtube\.com|youtu\.?be)/.+$'
    return bool(re.match(pattern, url))

# Function to download and transcribe audio from YouTube video
def transcribe_youtube_video(url):
    # Download video
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
            'nopostoverwrites': False  # Ensure postprocessor options are set correctly
        }],
        'outtmpl': 'audio.wav',
        'ffmpeg_location': '/usr/local/bin/ffmpeg'  # Adjust path based on your installation
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)  # Download video based on URL
            if 'entries' in info_dict:  # Check if multiple videos were returned (playlist)
                info_dict = info_dict['entries'][0]

        audio_file = 'audio.wav'
    except Exception as e:
        st.error(f"Error downloading video: {e}")
        return None

    # Transcribe audio
    model = Model("model")
    rec = KaldiRecognizer(model, 16000)

    try:
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
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return None

# Route to render the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle user queries
@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['query']
    response = agent(user_input)
    conversational_memory.add_message(user_input, response['output'])  # Store conversation history
    return jsonify({'response': response['output']})

# Main Streamlit app
def main():
    st.title("YouTube Video Transcription and Similarity Search")

    # User input for YouTube video URL
    video_url = st.text_input("Enter the YouTube video URL:", key="video_url")

    if st.button("Transcribe and Index"):
        st.info("Downloading video and transcribing audio... This may take some time.")

        if is_valid_youtube_url(video_url):
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
                st.warning("Error: Unable to transcribe audio from the provided YouTube URL.")
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
