import os
import time
import streamlit as st
import yt_dlp
from vosk import Model, KaldiRecognizer
import wave
import json
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm
from uuid import uuid4
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Add the path to ffmpeg to the PATH environment variable
ffmpeg_path = '/usr/local/bin/ffmpeg'
ffprobe_path = '/usr/local/bin/ffprobe'
os.environ['PATH'] += os.pathsep + os.path.dirname(ffmpeg_path)

# Access secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key, environment="us-east1-gcp")

# Initialize OpenAI embeddings
model_name = 'text-embedding-ada-002'
embed = OpenAIEmbeddings(model=model_name, openai_api_key=openai_api_key)

# Define index name and specifications
index_name = 'langchain-retrieval-augmentation'
spec = ServerlessSpec(cloud="aws", region="us-east-1")

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

# Function to transcribe YouTube video
def transcribe_youtube_video(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192'
        }],
        'outtmpl': 'audio.wav',
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        audio_file = 'audio.wav'
    except Exception as e:
        st.error(f"Error downloading video: {e}")
        return None

    # Transcribe audio using Vosk
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

            for chunk in chunks:
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
