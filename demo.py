import os
from dotenv import load_dotenv
from media_ingestion import MediaIngester
from speech_recognition import SpeechRecognizer
from speech_diarization import SpeakerDiarizer
from translate import translate_text
from text_to_speech import audio_segmentation_to_voice, accelerate_segments, toneconverter

def main():
    # Load environment variables
    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    
    if not hf_token:
        print("Error: HUGGINGFACE_TOKEN not found in .env file")
        return
    
    # Get input from user
    media_source = input("Enter video URL or local file path: ")
    target_language = input("Enter target language code (e.g., en, es, fr, de): ")
    
    # Initialize components
    print("Initializing pipeline components...")
    ingester = MediaIngester(output_dir="temp")
    recognizer = SpeechRecognizer(model_size="base")
    diarizer = SpeakerDiarizer(hf_token=hf_token)
    
    # Step 1: Process input and extract audio
    print("Processing media source...")
    video_path = ingester.process_input(media_source)
    audio_path = ingester.extract_audio(video_path)
    
    # Step 2: Perform speech recognition
    print("Transcribing audio...")
    segments = recognizer.transcribe(audio_path)
    
    # Step 3: Perform speaker diarization
    print("Identifying speakers...")
    speakers = diarizer.diarize(audio_path)
    
    # Step 4: Assign speakers to transcription segments
    print("Assigning speakers to segments...")
    final_segments = diarizer.assign_speakers_to_segments(segments, speakers)
    
    # Step 5: Translate the segments
    print(f"Translating to {target_language}...")
    translated_segments = translate_text(
        final_segments, 
        target_lang=target_language,
        translation_method="groq"  # You can change to "batch" or "iterative" if preferred
    )
    
    # Prepare result structure for TTS
    result_diarize = {"segments": translated_segments}
    
    # Step 6: Generate speech in target language
    print("Generating speech...")
    # Default voice selections - modify as needed
    valid_speakers = audio_segmentation_to_voice(
        result_diarize,
        TRANSLATE_AUDIO_TO=target_language,
        is_gui=False,
        tts_voice00="en-US-AriaNeural-Female",  # Default voice for SPEAKER_00
        tts_voice01="en-US-GuyNeural-Male",     # Default voice for SPEAKER_01
        tts_voice02="en-GB-SoniaNeural-Female", # Default voice for SPEAKER_02
    )
    
    # Step 7: Adjust speech timing to match original
    print("Adjusting speech timing...")
    audio_files, speakers_list = accelerate_segments(
        result_diarize,
        max_accelerate_audio=2.0,
        valid_speakers=valid_speakers
    )
    
    # Optional: Apply voice conversion to make synthetic voices sound more natural
    use_voice_conversion = input("Apply voice conversion to make TTS voices sound more natural? (y/n): ").lower() == 'y'
    if use_voice_conversion:
        print("Applying voice conversion...")
        toneconverter(
            result_diarize,
            preprocessor_max_segments=3,
            method_vc="freevc"  # Can be "freevc" or "openvoice"
        )
    
    # Step 8: Create final video with translated speech
    print("\nGenerating final video...")
    # Here you would add code to merge the translated audio with the original video
    
    print("\nTranslation and dubbing completed successfully!")

if __name__ == "__main__":
    main()