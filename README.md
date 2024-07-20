```markdown
# Speech-to-Text System for YouTube Videos

## Overview

This project aims to develop a robust speech-to-text system capable of transcribing YouTube video content, processing the transcriptions, and providing an interactive querying interface. The system leverages advanced speech recognition models, text embedding techniques, and a scalable indexing solution to deliver accurate and accessible transcriptions of video content.

## Features

- **Video Downloading**: Extracts audio from YouTube videos.
- **Audio Processing**: Converts audio to a suitable format for transcription.
- **Speech Recognition**: Utilizes the Vosk model for accurate transcription of audio.
- **Text Processing**: Splits transcriptions into manageable chunks, embeds them using OpenAI’s models, and indexes them for efficient retrieval.
- **Web Interface**: User-friendly interface built with Flask for uploading videos, viewing transcriptions, and querying content.
- **Metadata Extraction**: Captures relevant metadata (title, uploader, duration, etc.) for each video.
- **Multi-Query Retrieval**: Enhances the accuracy of information retrieval by generating and processing multiple query variations.

## Installation

### Prerequisites

- Python 3.7 or higher
- Pip (Python package installer)

### Step-by-Step Installation

1. **Clone the Repository**

    ```bash
    git clone https://github.com/yourusername/speech-to-text-youtube.git
    cd speech-to-text-youtube
    ```

2. **Install Required Packages**

    ```python
    import subprocess
    import sys

    # Function to install packages using pip
    def install_packages(packages):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])

    # List of packages to install
    packages = [
        'vosk', 'yt-dlp', 'tqdm', 'datasets', 'openai', 'pinecone-client', 'tiktoken',
        'pyarrow==11.0.0', 'flask',
        'langchain', 'langchainhub', 'langchain-openai', 'langchain_community',
        'langchain-pinecone', 'langchain_anthropic'
    ]

    # Install the packages
    install_packages(packages)

    # Additional setup steps (uncomment if needed for your environment)
    # subprocess.check_call(["wget", "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"])
    # subprocess.check_call(["unzip", "vosk-model-small-en-us-0.15.zip"])
    # subprocess.check_call(["mv", "vosk-model-small-en-us-0.15", "model"])
    # subprocess.check_call(["rm", "vosk-model-small-en-us-0.15.zip"])
    # subprocess.check_call(["rm", "-rf", "audio.wav"])
    # subprocess.check_call(["rm", "-rf", "transcription.txt"])
    # subprocess.check_call(["apt-get", "update"])
    # subprocess.check_call(["apt-get", "install", "-y", "ffmpeg"])
    ```

## Usage

### Running the Web Interface

To start the Flask web interface, navigate to the project directory and run:

```bash
export FLASK_APP=app.py
flask run
```

This will start the web server on `http://127.0.0.1:5000/`, where you can upload YouTube video URLs, view transcriptions, and query the content.

### Example Code Snippets

#### Video Downloading and Transcription

```python
import subprocess
from pytube import YouTube
import speech_recognition as sr

def download_audio(video_url, output_path):
    subprocess.check_call(["yt-dlp", "-x", "--audio-format", "wav", video_url, "-o", f"{output_path}/%(title)s.%(ext)s"])
    return f"{output_path}/{video_url.split('=')[-1]}.wav"

def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
    return recognizer.recognize_google(audio_data)

# Example usage
audio_file = download_audio('https://www.youtube.com/watch?v=dQw4w9WgXcQ', './audio')
transcription = transcribe_audio(audio_file)
```

#### Text Embedding and Indexing

```python
import openai
import pinecone

# Initialize Pinecone and OpenAI
pinecone.init(api_key="YOUR_PINECONE_API_KEY")
openai.api_key = "YOUR_OPENAI_API_KEY"

def embed_text(text):
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    return response['data'][0]['embedding']

def index_text(embedding, vector_id):
    index = pinecone.Index("your-index")
    index.upsert(vectors=[(vector_id, embedding)])

# Process and index text chunks
text_chunks = ["Quantum computing uses qubits...", "Michio Kaku is a theoretical physicist..."]
for i, chunk in enumerate(text_chunks):
    embedding = embed_text(chunk)
    index_text(embedding, f"chunk_{i}")
```

## Project Structure

```
speech-to-text-youtube/
│
├── app.py                     # Flask application
├── requirements.txt           # List of dependencies
├── README.md                  # Project documentation
├── audio/                     # Directory for downloaded audio files
├── transcriptions/            # Directory for transcriptions
├── static/                    # Static files (CSS, JS)
└── templates/                 # HTML templates for Flask
```

## Contributions

Contributions are welcome! Please open an issue or submit a pull request for any changes or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## References

- **Google Speech-to-Text API Documentation**: [https://cloud.google.com/speech-to-text](https://cloud.google.com/speech-to-text)
- **Vosk Speech Recognition Model**: [https://alphacephei.com/vosk/](https://alphacephei.com/vosk/)
- **OpenAI Embeddings**: [https://beta.openai.com/docs/guides/embeddings](https://beta.openai.com/docs/guides/embeddings)
- **Pinecone Documentation**: [https://www.pinecone.io/docs/](https://www.pinecone.io/docs/)

For further details, please refer to the [comprehensive project report](COMPREHENSIVE_REPORT.md).

---

Happy coding!
```

This `README.md` provides an overview of the project, installation instructions, usage examples, project structure, and contribution guidelines. Make sure to replace `"YOUR_PINECONE_API_KEY"` and `"YOUR_OPENAI_API_KEY"` with your actual API keys in the example code.
