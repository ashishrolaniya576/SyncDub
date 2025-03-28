import os
import sys
import logging
import re
import gradio as gr
from dotenv import load_dotenv

# Add the current directory to path to help with imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if (current_dir not in sys.path):
    sys.path.append(current_dir)

# Import the required modules
from media_ingestion import MediaIngester
from speech_recognition import SpeechRecognizer
from speech_diarization import SpeakerDiarizer
from translate import translate_text, generate_srt_subtitles
from text_to_speech import generate_tts
from audio_to_video import create_video_with_mixed_audio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create necessary directories
def create_directories(dirs):
    """Create necessary directories"""
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)

create_directories(["temp", "audio", "audio2", "reference_audio", "output"])

# Load environment variables
load_dotenv()

# Get API tokens
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if not hf_token:
    logger.error("Error: HUGGINGFACE_TOKEN not found in .env file")

# Language options
LANGUAGE_OPTIONS = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese": "zh",
    "Hindi": "hi",
    "Arabic": "ar"
}

# Global variable to store speaker information for dynamic UI generation
speaker_info = {}

def process_video(video_input, youtube_url, target_language, tts_choice, max_speakers, progress=gr.Progress()):
    """Process video through the SyncDub pipeline and return intermediate outputs for UI"""
    # Initialize progress and status
    progress(0, desc="Initializing")
    status_updates = []
    
    def update_status(message):
        status_updates.append(message)
        return "\n".join(status_updates)
    
    try:
        # Initialize components
        update_status("Initializing pipeline components...")
        ingester = MediaIngester(output_dir="temp")
        recognizer = SpeechRecognizer(model_size="base")
        diarizer = SpeakerDiarizer(hf_token=hf_token)
        
        # Determine input source
        media_source = youtube_url if youtube_url else video_input
        if not media_source:
            return None, "Error: No input provided. Please upload a video or enter a YouTube URL."
        
        # Step 1: Process input and extract audio
        update_status("Processing media source...")
        progress(0.1, desc="Processing media")
        video_path = ingester.process_input(media_source)
        audio_path = ingester.extract_audio(video_path)
        
        update_status("Separating audio sources...")
        progress(0.2, desc="Separating audio")
        clean_audio_path, bg_audio_path = ingester.separate_audio_sources(audio_path)
        update_status(f"Audio processing completed. Extracted clean speech and background audio.")
        
        # Step 2: Perform speech recognition
        update_status("Transcribing audio...")
        progress(0.3, desc="Transcribing audio")
        segments = recognizer.transcribe(clean_audio_path)
        
        # Step 3: Perform speaker diarization
        update_status("Identifying speakers...")
        progress(0.4, desc="Identifying speakers")
        max_speakers_value = None if not max_speakers else int(max_speakers)
        speakers = diarizer.diarize(clean_audio_path, max_speakers=max_speakers_value)
        
        # Step 4: Assign speakers to segments
        update_status("Assigning speakers to segments...")
        progress(0.5, desc="Assigning speakers")
        final_segments = diarizer.assign_speakers_to_segments(segments, speakers)
        
        # Gather unique speakers for the next UI step
        unique_speakers = set()
        for segment in final_segments:
            if 'speaker' in segment:
                unique_speakers.add(segment['speaker'])
        
        # Extract reference audio for voice cloning if needed
        reference_files = {}
        if tts_choice == "Voice cloning (XTTS)":
            update_status(f"Extracting reference audio for {len(unique_speakers)} speakers...")
            progress(0.6, desc="Extracting reference audio")
            reference_files = diarizer.extract_speaker_references(
                clean_audio_path, 
                speakers, 
                output_dir="reference_audio"
            )
        
        # Step 5: Translate the segments
        update_status(f"Translating to {target_language}...")
        progress(0.7, desc="Translating")
        target_lang_code = LANGUAGE_OPTIONS.get(target_language, "en")
        translated_segments = translate_text(
            final_segments, 
            target_lang=target_lang_code,
            translation_method="batch"
        )
        
        # Create subtitles for preview
        subtitle_file = f"temp/{os.path.basename(video_path).split('.')[0]}_{target_lang_code}.srt"
        generate_srt_subtitles(translated_segments, output_file=subtitle_file)
        update_status(f"Generated subtitle file: {subtitle_file}")
        
        # Prepare speaker info for the next UI step
        global speaker_info
        speaker_info = {
            "video_path": video_path,
            "bg_audio_path": bg_audio_path,
            "translated_segments": translated_segments,
            "target_language": target_lang_code,
            "use_voice_cloning": tts_choice == "Voice cloning (XTTS)",
            "reference_files": reference_files,
            "unique_speakers": sorted(list(unique_speakers)),
            "subtitle_file": subtitle_file
        }
        
        # Get sample transcript for preview
        sample_transcript = ""
        for i, segment in enumerate(translated_segments[:5]):  # Show first 5 segments
            sample_transcript += f"Speaker {segment.get('speaker', 'Unknown').replace('SPEAKER_', '')}: {segment.get('text', '')}\n"
            if i >= 4 and len(translated_segments) > 5:
                sample_transcript += f"\n... and {len(translated_segments) - 5} more segments"
        
        progress(1.0, desc="Initial processing complete")
        return sample_transcript, update_status("Initial processing complete. Please configure speaker voices below.")
    
    except Exception as e:
        logger.exception("Error processing video")
        return None, update_status(f"Error: {str(e)}")

def create_speaker_ui():
    """Dynamically create UI elements for speaker configuration based on detected speakers"""
    global speaker_info
    use_voice_cloning = speaker_info.get("use_voice_cloning", False)
    reference_files = speaker_info.get("reference_files", {})
    unique_speakers = speaker_info.get("unique_speakers", [])
    
    if not unique_speakers:
        return [gr.Markdown("No speakers detected.")]
    
    # Create a list of components to return
    components = []
    components.append(gr.Markdown(f"### Configure voices for {len(unique_speakers)} detected speakers"))
    components.append(gr.Markdown("Add labels to help identify which speaker is which person in your video."))
    speaker_configs = {}
    
    for speaker in unique_speakers:
        match = re.search(r'SPEAKER_(\d+)', speaker)
        if not match:
            continue
            
        speaker_id = int(match.group(1))
        speaker_label = f"Speaker {speaker_id+1}"
        
        # Create speaker section with divider for visual separation
        components.append(gr.Markdown(f"---"))
        components.append(gr.Markdown(f"#### {speaker_label}"))
        
        # Add a text field for speaker identification/labeling
        speaker_name = gr.Textbox(
            label="Label (optional, e.g. 'John', 'Interviewer', 'Child')", 
            placeholder="Enter a name or role to identify this speaker",
            value=""
        )
        components.append(speaker_name)
        
        # Add gender selection for all speakers
        gender_radio = gr.Radio(
            choices=["Male", "Female"], 
            value="Male",
            label=f"Voice Gender for {speaker_label}"
        )
        components.append(gender_radio)
        
        # Handle voice cloning option
        if use_voice_cloning and speaker in reference_files:
            # Reference audio exists for this speaker
            ref_audio = reference_files[speaker]
            # Ensure the file exists
            if os.path.exists(ref_audio):
                components.append(gr.Audio(value=ref_audio, label="Reference Audio Sample"))
                voice_option = gr.Radio(
                    choices=["Use voice cloning", "Use Edge TTS"],
                    value="Use voice cloning",
                    label=f"Voice Option for {speaker_label}"
                )
                components.append(voice_option)
            else:
                # No valid reference audio, fallback to Edge TTS
                components.append(gr.Markdown("*No valid reference audio available for this speaker, falling back to Edge TTS*"))
                voice_option = None
        else:
            # No reference audio or not using voice cloning
            if use_voice_cloning:
                components.append(gr.Markdown("*No reference audio available for this speaker*"))
            voice_option = None
        
        # Store configuration references
        speaker_configs[speaker_id] = {
            "speaker": speaker,
            "use_cloning": use_voice_cloning and speaker in reference_files,
            "ref_audio": reference_files.get(speaker, None),
            "name_input": speaker_name,
            "gender_input": gender_radio,
            "voice_option_input": voice_option
        }
    
    speaker_info["speaker_configs"] = speaker_configs
    
    return components

def finalize_video(progress=gr.Progress()):
    """Generate final dubbed video with configured voices"""
    global speaker_info
    progress(0, desc="Starting final processing")
    status_updates = []
    
    def update_status(message):
        status_updates.append(message)
        return "\n".join(status_updates)
    
    try:
        # Extract needed info from global state
        video_path = speaker_info["video_path"]
        bg_audio_path = speaker_info["bg_audio_path"]
        translated_segments = speaker_info["translated_segments"]
        target_language = speaker_info["target_language"]
        use_voice_cloning = speaker_info["use_voice_cloning"]
        speaker_configs = speaker_info.get("speaker_configs", {})
        
        # Create voice configuration based on UI selections
        update_status("Configuring voices for speakers...")
        progress(0.1, desc="Configuring voices")
        
        # Process speaker configurations from UI inputs
        voice_config = {}
        
        if use_voice_cloning:
            for speaker_id, config in speaker_configs.items():
                name = config["name_input"].value
                name_display = f" ({name})" if name else ""
                gender = config["gender_input"].value.lower() if hasattr(config["gender_input"], "value") else "male"
                
                if config["use_cloning"] and config["ref_audio"] and os.path.exists(config["ref_audio"]):
                    # Check if user wants to use cloning or fallback
                    use_cloning_option = True
                    if config["voice_option_input"] and hasattr(config["voice_option_input"], "value"):
                        use_cloning_option = config["voice_option_input"].value == "Use voice cloning"
                    
                    if use_cloning_option:
                        # XTTS voice cloning configuration
                        voice_config[speaker_id] = {
                            'engine': 'xtts',
                            'reference_audio': config["ref_audio"],
                            'language': target_language
                        }
                        update_status(f"Speaker {speaker_id+1}{name_display}: Using voice cloning with reference audio")
                    else:
                        # Fallback to Edge TTS by user choice
                        voice_config[speaker_id] = {
                            'engine': 'edge_tts',
                            'gender': "female" if gender == "female" else "male"
                        }
                        update_status(f"Speaker {speaker_id+1}{name_display}: Using Edge TTS ({gender}) - user choice")
                else:
                    # Fallback to Edge TTS if no reference audio
                    voice_config[speaker_id] = {
                        'engine': 'edge_tts',
                        'gender': "female" if gender == "female" else "male"
                    }
                    update_status(f"Speaker {speaker_id+1}{name_display}: Using Edge TTS ({gender}) - no reference audio")
        else:
            # Standard Edge TTS configuration
            for speaker_id, config in speaker_configs.items():
                name = config["name_input"].value if hasattr(config["name_input"], "value") else ""
                name_display = f" ({name})" if name else ""
                gender = config["gender_input"].value.lower() if hasattr(config["gender_input"], "value") else "male"
                
                # Simple string format
                voice_config[speaker_id] = "female" if gender == "female" else "male"
                update_status(f"Speaker {speaker_id+1}{name_display}: Using Edge TTS ({gender})")
        
        # Generate speech in target language
        update_status("Generating speech audio...")
        progress(0.3, desc="Generating speech")
        dubbed_audio_path = generate_tts(translated_segments, target_language, voice_config, output_dir="audio2")
        
        # Create video with mixed audio
        update_status("Creating final video with translated audio...")
        progress(0.7, desc="Creating video")
        output_video_path = create_video_with_mixed_audio(video_path, bg_audio_path, dubbed_audio_path)
        
        progress(1.0, desc="Complete")
        return output_video_path, speaker_info["subtitle_file"], update_status("Video dubbing completed successfully!")
    
    except Exception as e:
        logger.exception("Error generating dubbed video")
        return None, None, update_status(f"Error: {str(e)}")

# Function to show the generate button after processing
def show_generate_button():
    return gr.update(visible=True)

# Function to update message after processing
def update_after_processing():
    return "Processing complete! Please go to the 'Configure Voices' tab to continue."

# Main Gradio App
with gr.Blocks(title="SyncDub - AI Video Dubbing") as app:
    gr.Markdown("# SyncDub - AI Video Dubbing")
    gr.Markdown("Translate and dub videos with AI-powered voice cloning")
    
    with gr.Tabs() as tabs:
        with gr.Tab("1. Upload & Process"):
            with gr.Row():
                with gr.Column():
                    # Input section
                    video_input = gr.Video(label="Upload Video")
                    youtube_url = gr.Textbox(label="Or enter YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
                    
                    with gr.Row():
                        target_language = gr.Dropdown(
                            choices=list(LANGUAGE_OPTIONS.keys()),
                            value="English",
                            label="Target Language"
                        )
                        
                        tts_choice = gr.Radio(
                            choices=["Standard TTS", "Voice cloning (XTTS)"],
                            value="Voice cloning (XTTS)",
                            label="Voice Generation Method"
                        )
                    
                    max_speakers = gr.Number(
                        label="Max Speakers (Optional)", 
                        value=None, 
                        precision=0,
                        placeholder="Leave empty for auto-detection"
                    )
                    
                    process_btn = gr.Button("Process Video", variant="primary")
                
                with gr.Column():
                    # Output/status section
                    status_output = gr.Textbox(
                        label="Status", 
                        placeholder="Processing status will appear here...",
                        lines=10,
                        interactive=False
                    )
                    
                    transcript_output = gr.Textbox(
                        label="Sample Transcript", 
                        placeholder="Transcript will appear here after processing...",
                        lines=8,
                        interactive=False
                    )
                    
                    after_process_msg = gr.Textbox(
                        label="Next Steps", 
                        visible=False,
                        interactive=False
                    )
        
        with gr.Tab("2. Configure Voices"):
            # Container for dynamic speaker UI elements
            with gr.Column() as voice_config_container:
                gr.Markdown("Please process a video in the Upload & Process tab first.")
            
            # Generate button (initially hidden)
            generate_btn = gr.Button("Generate Dubbed Video", visible=False, variant="primary")
        
        with gr.Tab("3. Results"):
            with gr.Column():
                output_video = gr.Video(label="Dubbed Video", interactive=False)
                with gr.Row():
                    subtitle_download = gr.File(label="Download Subtitles")
                    # A button to download video could be added here
                final_status = gr.Textbox(
                    label="Generation Status", 
                    placeholder="Status will appear here during generation...",
                    lines=5, 
                    interactive=False
                )
    
    # Connect events
    process_chain = process_btn.click(
        fn=process_video,
        inputs=[video_input, youtube_url, target_language, tts_choice, max_speakers],
        outputs=[transcript_output, status_output]
    )
    
    # Update the UI in the second tab
    process_chain.success(
        fn=create_speaker_ui,
        inputs=[],
        outputs=voice_config_container
    )
    
    # Show the generate button
    process_chain.success(
        fn=show_generate_button,
        inputs=[],
        outputs=generate_btn
    )
    
    # Show instructions for next steps
    process_chain.success(
        fn=update_after_processing,
        inputs=[],
        outputs=after_process_msg
    )
    
    # Set visibility for instruction message
    process_chain.success(
        fn=lambda: gr.update(visible=True),
        inputs=[],
        outputs=after_process_msg
    )
    
    # Handle the generate button click
    generate_chain = generate_btn.click(
        fn=finalize_video,
        inputs=[],
        outputs=[output_video, subtitle_download, final_status]
    )

# Launch the app
if __name__ == "__main__":
    app.launch()