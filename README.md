---
title: SyncDub
app_file: gradio_app.py
sdk: gradio
sdk_version: 5.29.0
---
# SyncDub

SyncDub is a Python application designed to automatically translate and dub videos into various languages. It leverages speech recognition, speaker diarization, machine translation, and text-to-speech (TTS) technologies to create dubbed versions of input videos while attempting to preserve background audio and optionally clone speaker voices.

## Features

*   **Video Input:** Accepts YouTube URLs or local video file uploads.
*   **Audio Extraction & Separation:** Extracts audio from video and separates speech from background noise using Demucs.
*   **Speech Recognition:** Transcribes the spoken content using Whisper.
*   **Speaker Diarization:** Identifies different speakers in the audio using `pyannote.audio`.
*   **Machine Translation:** Translates the transcribed text into multiple target languages using `deep-translator` (with options for batch, iterative, or Groq API methods).
*   **Text-to-Speech (TTS):**
    *   **Simple Dubbing:** Generates dubbed audio using Microsoft Edge TTS, allowing gender selection per speaker.
    *   **Voice Cloning:** Uses Coqui XTTS to clone the original speakers' voices for a more natural dub (requires reference audio extraction).
*   **Audio Mixing:** Combines the generated dubbed speech with the original background audio.
*   **Video Reassembly:** Creates the final dubbed video file.
*   **Subtitle Generation:** Outputs translated subtitles in `.srt` format.
*   **Web Interface:** Provides an easy-to-use Gradio interface for processing videos.
*   **Command-Line Interface:** Includes a `demo.py` script for terminal-based usage.

## Requirements

*   Python 3.8+
*   FFmpeg (must be installed and accessible in your system's PATH)
*   Key Python packages (see `requirements.txt`):
    *   `yt-dlp`
    *   `moviepy`
    *   `pyannote.audio`
    *   `transformers`
    *   `torch` & `torchaudio` (often dependencies of the above)
    *   `deep-translator`
    *   `TTS` (Coqui TTS for voice cloning)
    *   `edge-tts`
    *   `gradio`
    *   `python-dotenv`
    *   `demucs`
*   **API Keys:**
    *   `HUGGINGFACE_TOKEN`: Required for `pyannote.audio` speaker diarization models. Obtain from [Hugging Face](https://huggingface.co/settings/tokens).
    *   `GROQ_API_KEY` (Optional): Required if using the "groq" translation method. Obtain from [GroqCloud](https://console.groq.com/keys).

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/PranavInani/SyncDub
    cd SyncDub
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install FFmpeg:** Follow instructions for your operating system ([https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)). Ensure it's added to your system's PATH.
4.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    # Depending on your system, you might need to install PyTorch separately
    # See https://pytorch.org/get-started/locally/
    ```
5.  **Create a `.env` file:** In the root directory of the project, create a file named `.env` and add your Hugging Face token:
    ```dotenv
    # filepath: .env
    HUGGINGFACE_TOKEN=hf_YOUR_HUGGINGFACE_TOKEN_HERE
    # Add GROQ_API_KEY if you plan to use the Groq translation method
    # GROQ_API_KEY=gsk_YOUR_GROQ_API_KEY_HERE
    ```
    Replace `hf_YOUR_HUGGINGFACE_TOKEN_HERE` with your actual token.

## Usage

### Gradio Web Interface

Launch the web application:

```bash
python gradio_app.py
```

This will start a local web server. Open the provided URL (usually `http://127.0.0.1:7860`) in your browser.

1.  **Input:** Enter a video URL (e.g., YouTube) or upload a local video file.
2.  **Target Language:** Select the desired output language.
3.  **TTS Method:** Choose between "Simple dubbing (Edge TTS)" or "Voice cloning (XTTS)".
4.  **Translation Method:** Select "batch", "iterative", or "groq".
5.  **Maximum Speakers:** Optionally specify the number of speakers to detect. Click "Update Speaker Options" to configure genders if needed (especially for Edge TTS).
6.  **Process:** Click "Process Video".
7.  **Output:** Download links for the dubbed video and subtitle file will appear upon completion. Use the "Reset Everything" button to clear temporary files before processing a new video.

### Command-Line Demo

Run the demo script:

```bash
python demo.py
```

The script will prompt you for:

1.  Video URL or local file path.
2.  Target language code (e.g., `en`, `es`, `hi`).
3.  TTS engine choice (1 for Edge TTS, 2 for XTTS).
4.  Maximum number of speakers (optional).
5.  Speaker genders (if using Edge TTS or as fallback for XTTS).

The processed files will be saved in the `temp` directory, with the final video typically named `output_video.mp4`.

## Configuration

*   **API Keys:** Configure `HUGGINGFACE_TOKEN` and optionally `GROQ_API_KEY` in the `.env` file.
*   **Models:** Model sizes and specific checkpoints can be adjusted within the Python scripts (`speech_recognition.py`, `speech_diarization.py`, etc.) if needed.

## Directory Structure

*   `temp/`: Stores intermediate files like downloaded video, extracted audio, separated sources, final output video.
*   `audio/`: Often used for initial audio extraction outputs.
*   `audio2/`: Stores the generated TTS audio segments and the final mixed dubbed audio.
*   `reference_audio/`: Stores extracted audio snippets for each speaker when using XTTS voice cloning.
*   `outputs/`: Stores the final video and subtitle files made available for download in the Gradio interface.

## Troubleshooting

*   **Errors during processing:**
    *   Ensure `ffmpeg` is installed correctly and accessible in your PATH.
    *   Verify that the `HUGGINGFACE_TOKEN` in your `.env` file is correct and valid.
    *   Check if you have sufficient disk space and memory, especially for large videos or voice cloning.
    *   Ensure all dependencies from `requirements.txt` are installed correctly in your virtual environment.
*   **Voice cloning issues:** XTTS quality depends heavily on the quality and duration of the extracted reference audio for each speaker. If diarization is poor or speakers have very little unique speech, cloning may fail or produce poor results. Consider using "Simple dubbing" as an alternative.
*   **Slow processing:** Video processing, especially diarization, translation, and TTS (XTTS), can be computationally intensive and time-consuming.
*   **Reset:** Use the "Reset Everything" button in the Gradio app to clear temporary directories if you encounter persistent issues or before starting a new video.


## Contributing

Everyone is encouraged to contribute.
