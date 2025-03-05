import os
from dotenv import load_dotenv
from media_ingestion import MediaIngester
from speech_recognition import SpeechRecognizer
from speech_diarization import SpeakerDiarizer

def main():
    # Load environment variables
    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    
    if not hf_token:
        print("Error: HUGGINGFACE_TOKEN not found in .env file")
        return
    
    # Get input from user
    media_source = input("Enter video URL or local file path: ")
    
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
    
    # Print results
    print("\nTranscription with speaker identification:")
    for i, segment in enumerate(final_segments):
        print(f"[{segment['speaker']}] ({segment['start']:.1f}s - {segment['end']:.1f}s): {segment['text']}")

if __name__ == "__main__":
    main()