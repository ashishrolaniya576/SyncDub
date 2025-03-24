import os
import sys
import logging
from dotenv import load_dotenv
import re

# Add the current directory to path to help with imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import the required modules
from media_ingestion import MediaIngester
from speech_recognition import SpeechRecognizer
from speech_diarization import SpeakerDiarizer
from translate import translate_text, generate_srt_subtitles
from text_to_speech import generate_edge_tts

def create_directories(dirs):
    """Create necessary directories"""
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)

def main():
    # Load environment variables
    load_dotenv()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Create necessary directories
    create_directories(["temp", "audio", "audio2"])
    
    # Get API tokens
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    
    if not hf_token:
        logger.error("Error: HUGGINGFACE_TOKEN not found in .env file")
        return
    
    # Get input from user
    media_source = input("Enter video URL or local file path: ")
    target_language = input("Enter target language code (e.g., en, es, fr, de): ")
    
    # Initialize components
    logger.info("Initializing pipeline components...")
    ingester = MediaIngester(output_dir="temp")
    recognizer = SpeechRecognizer(model_size="base")
    diarizer = SpeakerDiarizer(hf_token=hf_token)
    
    # Step 1: Process input and extract audio
    logger.info("Processing media source...")
    video_path = ingester.process_input(media_source)
    audio_path = ingester.extract_audio(video_path)
    
    # Step 2: Perform speech recognition
    logger.info("Transcribing audio...")
    segments = recognizer.transcribe(audio_path)
    
    # Step 3: Perform speaker diarization
    logger.info("Identifying speakers...")
    
    # Add user input for max speakers
    max_speakers_str = input("Maximum number of speakers to detect (leave blank for auto): ")
    max_speakers = int(max_speakers_str) if max_speakers_str.strip() else None

    # Then call diarize with this parameter
    speakers = diarizer.diarize(audio_path, max_speakers=max_speakers)
    
    # Step 4: Assign speakers to segments
    logger.info("Assigning speakers to segments...")
    final_segments = diarizer.assign_speakers_to_segments(segments, speakers)
    
    # Step 5: Translate the segments
    logger.info(f"Translating to {target_language}...")
    translated_segments = translate_text(
        final_segments, 
        target_lang=target_language,
        translation_method="groq"  # Can be "batch" or "iterative" or "groq"
    )


    # Print translated segments for debugging
    subtitle_file = f"temp/{os.path.basename(video_path).split('.')[0]}_{target_language}.srt"
    generate_srt_subtitles(translated_segments, output_file=subtitle_file)
    logger.info(f"Generated subtitle file: {subtitle_file}")
    # Step 6: Configure voice characteristics for speakers
    voice_config = {}  # Map of speaker_id to gender

    # Detect number of unique speakers
    unique_speakers = set()
    for segment in translated_segments:
        if 'speaker' in segment:
            unique_speakers.add(segment['speaker'])
    print(unique_speakers)
    if len(unique_speakers) > 1:
        logger.info(f"Detected {len(unique_speakers)} speakers")
        for speaker in sorted(list(unique_speakers)):  
            match = re.search(r'SPEAKER_(\d+)', speaker)
            if match:
                speaker_id = int(match.group(1))
                gender = input(f"Select voice gender for Speaker {speaker_id+1} (m/f): ").lower()
                voice_config[speaker_id] = "female" if gender.startswith("f") else "male"
    
    
    # # Step 7: Generate speech in target language
    logger.info("Generating speech...")
    generate_edge_tts(translated_segments, target_language, voice_config, output_dir="audio2")
    
    

if __name__ == "__main__":
    main()