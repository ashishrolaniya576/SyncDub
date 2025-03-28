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
from text_to_speech import generate_tts  # Import both TTS functions
from audio_to_video import create_video_with_mixed_audio

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
    create_directories(["temp", "audio", "audio2", "reference_audio"])
    
    # Get API tokens
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    
    if not hf_token:
        logger.error("Error: HUGGINGFACE_TOKEN not found in .env file")
        return
    
    # Get input from user
    media_source = input("Enter video URL or local file path: ")
    target_language = input("Enter target language code (e.g., en, es, fr, de): ")
    
    # Choose TTS engine
    print("\nSelect TTS engine:")
    print("1. Simple dubbing (Edge TTS)")
    print("2. Voice cloning (XTTS)")
    tts_choice = input("Enter choice (1/2): ").strip()
    use_voice_cloning = tts_choice == "2"
    
    # Initialize components
    logger.info("Initializing pipeline components...")
    ingester = MediaIngester(output_dir="temp")
    recognizer = SpeechRecognizer(model_size="base")
    diarizer = SpeakerDiarizer(hf_token=hf_token)
    
    # Step 1: Process input and extract audio
    logger.info("Processing media source...")
    video_path = ingester.process_input(media_source)
    audio_path = ingester.extract_audio(video_path)
    clean_audio_path, bg_audio_path = ingester.separate_audio_sources(audio_path)
    logger.info("Extracted audio: %s", audio_path)
    logger.info("Cleaned audio: %s", clean_audio_path)
    logger.info("Background audio: %s", bg_audio_path)
    logger.info("Audio processing completed.")
    
    # Step 2: Perform speech recognition
    logger.info("Transcribing audio...")
    segments = recognizer.transcribe(clean_audio_path)
    
    # Step 3: Perform speaker diarization
    logger.info("Identifying speakers...")
    
    # Add user input for max speakers
    max_speakers_str = input("Maximum number of speakers to detect (leave blank for auto): ")
    max_speakers = int(max_speakers_str) if max_speakers_str.strip() else None

    # Then call diarize with this parameter
    speakers = diarizer.diarize(clean_audio_path, max_speakers=max_speakers)
    
    # Step 4: Assign speakers to segments
    logger.info("Assigning speakers to segments...")
    final_segments = diarizer.assign_speakers_to_segments(segments, speakers)
    
    # Step 5: Translate the segments
    logger.info(f"Translating to {target_language}...")
    translated_segments = translate_text(
        final_segments, 
        target_lang=target_language,
        translation_method="batch"  # Can be "batch" or "iterative" or "groq"
    )

    # Print translated segments for debugging
    subtitle_file = f"temp/{os.path.basename(video_path).split('.')[0]}_{target_language}.srt"
    generate_srt_subtitles(translated_segments, output_file=subtitle_file)
    logger.info(f"Generated subtitle file: {subtitle_file}")
    
    # Step 6: Configure voice characteristics for speakers
    voice_config = {}  # Map of speaker_id to gender or voice config

    # Detect number of unique speakers
    unique_speakers = set()
    for segment in translated_segments:
        if 'speaker' in segment:
            unique_speakers.add(segment['speaker'])
    
    logger.info(f"Detected {len(unique_speakers)} speakers")
    
    if use_voice_cloning:
        # Extract reference audio for voice cloning
        logger.info("Extracting speaker reference audio for voice cloning...")
        reference_files = diarizer.extract_speaker_references(
            clean_audio_path, 
            speakers, 
            output_dir="reference_audio"
        )
        
        # Create voice config for XTTS
        for speaker in sorted(list(unique_speakers)):
            match = re.search(r'SPEAKER_(\d+)', speaker)
            if match:
                speaker_id = int(match.group(1))
                if speaker in reference_files:
                    voice_config[speaker_id] = {
                        'engine': 'xtts',
                        'reference_audio': reference_files[speaker],
                        'language': target_language
                    }
                    logger.info(f"Using voice cloning for Speaker {speaker_id+1} with reference file: {os.path.basename(reference_files[speaker])}")
                else:
                    # Fallback to Edge TTS if no reference audio
                    logger.warning(f"No reference audio found for Speaker {speaker_id+1}, falling back to Edge TTS")
                    gender = input(f"Select voice gender for Speaker {speaker_id+1} (m/f): ").lower()
                    voice_config[speaker_id] = {
                        'engine': 'edge_tts',
                        'gender': "female" if gender.startswith("f") else "male"
                    }
    else:
        # Standard Edge TTS configuration - keeping your current approach
        if len(unique_speakers) > 0:
            for speaker in sorted(list(unique_speakers)):
                match = re.search(r'SPEAKER_(\d+)', speaker)
                if match:
                    speaker_id = int(match.group(1))
                    gender = input(f"Select voice gender for Speaker {speaker_id+1} (m/f): ").lower()
                    voice_config[speaker_id] = {
                        'engine': 'edge_tts',
                        'gender': "female" if gender.startswith("f") else "male"
                    }
    
    # Step 7: Generate speech in target language
    logger.info("Generating speech...")
    
    dubbed_audio_path = generate_tts(translated_segments, target_language, voice_config, output_dir="audio2")

    # Step 8: Create video with mixed audio
    logger.info("Creating video with translated audio...")
    create_video_with_mixed_audio(video_path, bg_audio_path, dubbed_audio_path)
    
    logger.info("Process completed successfully!")

if __name__ == "__main__":
    main()