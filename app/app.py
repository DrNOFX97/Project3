import os
import streamlit as st
import yt_dlp
from vosk import Model, KaldiRecognizer
from tqdm.auto import tqdm
import wave
import json

# Environment setup
ffmpeg_path = '/usr/local/bin/ffmpeg'  # Example path, adjust as necessary
ffprobe_path = '/usr/local/bin/ffprobe'  # Example path, adjust as necessary
os.environ['PATH'] += os.pathsep + os.path.dirname(ffmpeg_path) + os.pathsep + os.path.dirname(ffprobe_path)

# Initialize Streamlit app
st.title("YouTube Video Transcription")

# Function to check if input URL is a valid YouTube URL
def is_valid_youtube_url(url):
    pattern = r'^(https?://)?(www\.)?(youtube\.com|youtu\.?be)/.+$'
    return bool(re.match(pattern, url))

# Function to transcribe YouTube video
def transcribe_youtube_video(url):
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'ffmpeg_location': ffmpeg_path,
            'ffprobe_location': ffprobe_path,
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
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return None

def main():
    # User input for YouTube video URL
    video_url = st.text_input("Enter the YouTube video URL:")

    if st.button("Transcribe"):
        if is_valid_youtube_url(video_url):
            st.info("Downloading video and transcribing audio... This may take some time.")

            # Download and transcribe YouTube video
            transcription = transcribe_youtube_video(video_url)

            if transcription:
                st.success("Transcription complete.")
                st.text_area("Transcription", value=transcription, height=200)
            else:
                st.warning("Transcription failed. Please check the YouTube URL and try again.")
        else:
            st.warning("Please enter a valid YouTube video URL.")

if __name__ == '__main__':
    main()
