import os
import logging
from dotenv import load_dotenv
from media_ingestion import MediaIngester
from speech_recognition import SpeechRecognizer
from speech_diarization import SpeakerDiarizer
from translate import translate_text
from text_to_speech import audio_segmentation_to_voice, accelerate_segments, toneconverter
from utils import create_directories

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
    audio_path = ingester.extract_audio(video_path, output_path="audio.wav")
    
    # Step 2: Perform speech recognition
    logger.info("Transcribing audio...")
    segments = recognizer.transcribe(audio_path)
    
    # Step 3: Perform speaker diarization
    logger.info("Identifying speakers...")
    speakers = diarizer.diarize(audio_path)
    
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
    
    # Prepare result structure for TTS
    result_diarize = {"segments": translated_segments}
    
    # Step 6: Generate speech in target language
    logger.info("Generating speech...")
    # You can select different voices for different speakers
    # Default voice selections based on available voices in your system
    valid_speakers = audio_segmentation_to_voice(
        result_diarize,
        TRANSLATE_AUDIO_TO=target_language,
        is_gui=False,
        # Select appropriate voices for your target language
        tts_voice00="en-US-AriaNeural-Female",  # Default voice for SPEAKER_00
        tts_voice01="en-US-GuyNeural-Male",     # Default voice for SPEAKER_01
        tts_voice02="en-GB-SoniaNeural-Female", # Default voice for SPEAKER_02
        # Add more voices as needed for additional speakers
    )
    
    # Step 7: Adjust speech timing to match original
    logger.info("Adjusting speech timing...")
    audio_files, speakers_list = accelerate_segments(
        result_diarize,
        max_accelerate_audio=2.0,
        valid_speakers=valid_speakers
    )
    
    # Optional: Apply voice conversion to make synthetic voices sound more natural
    use_voice_conversion = input("Apply voice conversion to make TTS voices sound more natural? (y/n): ").lower() == 'y'
    if use_voice_conversion:
        logger.info("Applying voice conversion...")
        toneconverter(
            result_diarize,
            preprocessor_max_segments=3,
            method_vc="freevc"  # Can be "freevc" or "openvoice"
        )
    
    # Step 8: Create final video with translated speech
    logger.info("Generating final video...")
    # Add code to merge the translated audio with the original video
    # This could use ffmpeg to replace the audio track in the original video
    
    # You might need to implement a function like:
    # merge_audio_with_video(video_path, "audio2/audio", output_path="output.mp4")
    
    logger.info("Translation and dubbing completed successfully!")
    logger.info(f"Processed audio files: {len(audio_files)}")

if __name__ == "__main__":
    main()