import os
import re
import time
import json
import wave
from uuid import uuid4
from tqdm.auto import tqdm
import streamlit as st
import pinecone
import yt_dlp
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from vosk import Model, KaldiRecognizer
from flask import Flask, request, jsonify, render_template

# Environment setup
os.environ['PATH'] += os.pathsep + '/usr/local/bin'

# Access secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]

# Initialize Pinecone client
pinecone.init(api_key=pinecone_api_key, environment="us-east1-gcp")

# Initialize OpenAI embeddings
model_name = 'text-embedding-ada-002'
embed = OpenAIEmbeddings(model=model_name, openai_api_key=openai_api_key)

# Define index name and specifications
index_name = 'langchain-retrieval-augmentation'
spec = pinecone.ServerlessSpec(cloud="aws", region="us-east-1")

# Check if index exists, create if not
existing_indexes = [index_info["name"] for index_info in pinecone.list_indexes()]
if index_name not in existing_indexes:
    pinecone.create_index(index_name, dimension=1536, metric='dotproduct', spec=spec)
    while not pinecone.describe_index(index_name).status['ready']:
        time.sleep(1)

# Connect to Pinecone index
index = pinecone.Index(index_name)

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

def transcribe_youtube_video(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
            'nopostoverwrites': False
        }],
        'outtmpl': 'audio.wav',
        'ffmpeg_location': '/usr/local/bin/ffmpeg'
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info without downloading
            info_dict = ydl.extract_info(url, download=False)
            
            if isinstance(info_dict, str):
                raise ValueError(f"yt-dlp returned an unexpected string: {info_dict}")
            elif not isinstance(info_dict, dict):
                raise ValueError(f"yt-dlp returned an unexpected type: {type(info_dict)}")

            # Proceed with download
            video_title = info_dict.get('title', 'Unknown Title')
            st.info(f"Downloading: {video_title}")
            ydl.download([url])

        return 'audio.wav'

    except yt_dlp.utils.DownloadError as e:
        st.error(f"Error downloading video: {e}")
    except yt_dlp.utils.ExtractorError as e:
        st.error(f"Error extracting video info: {e}")
    except ValueError as e:
        st.error(f"Unexpected data from yt-dlp: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {type(e).__name__}, {str(e)}")
    
    st.error("Unable to transcribe audio from the provided YouTube URL.")
    return None

# Main Streamlit app function
def main():
    st.title("YouTube Video Transcription and Similarity Search")

    # User input for YouTube video URL
    video_url = st.text_input("Enter the YouTube video URL:", key="video_url")

    if st.button("Transcribe and Index"):
        st.info("Downloading video and transcribing audio... This may take some time.")

        if is_valid_youtube_url(video_url):
            audio_file = transcribe_youtube_video(video_url)
            if audio_file:
                # Transcribe audio
                model = Model("model")
                rec = KaldiRecognizer(model, 16000)

                results = []

                try:
                    with wave.open(audio_file, "rb") as wf:
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

                        # Load the final result from 'rec.FinalResult()'
                        part_result = json.loads(rec.FinalResult())
                        results.append(part_result['text'])

                        # Join all texts in results into a single transcription string
                        transcription = " ".join(results)

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
                            st.error("No text chunks generated. Check the transcription process.")
                except Exception as e:
                    st.error(f"Error transcribing audio: {e}")
            else:
                st.error("Failed to retrieve audio file.")
        else:
            st.warning("Please enter a valid YouTube video URL.")

    # User input for similarity search query
    query = st.text_input("Enter your query:", key="query")

    if st.button("Search"):
        if query:
            try:
                # Perform similarity search
                results = vectorstore.similarity_search(query, k=3)

                # Display search results
                if results:
                    st.subheader("Top 3 Most Relevant Documents:")
                    for result in results:
                        st.write(f"- Document ID: {result.id}, Score: {result.score}")
                else:
                    st.warning("No results found.")
            except Exception as e:
                st.error(f"An error occurred during the search: {e}")
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()
