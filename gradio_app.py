import gradio as gr
import os
import re
from generate_tts import generate_tts  # Ensure these modules exist
from video_processing import create_video_with_mixed_audio

# Global speaker info dictionary
speaker_info = {
    "use_voice_cloning": False,
    "reference_files": {},
    "unique_speakers": []
}

def create_speaker_ui():
    """Dynamically create UI elements for speaker configuration based on detected speakers"""
    global speaker_info
    use_voice_cloning = speaker_info.get("use_voice_cloning", False)
    reference_files = speaker_info.get("reference_files", {})
    unique_speakers = speaker_info.get("unique_speakers", [])

    if not unique_speakers:
        return gr.Markdown("No speakers detected.")

    with gr.Column():
        gr.Markdown(f"### Configure voices for {len(unique_speakers)} detected speakers")
        gr.Markdown("Add labels to help identify which speaker is which person in your video.")

        speaker_configs = {}

        for speaker in unique_speakers:
            match = re.search(r'SPEAKER_(\d+)', speaker)
            if not match:
                continue

            speaker_id = int(match.group(1))
            speaker_label = f"Speaker {speaker_id+1}"

            with gr.Row():
                gr.Markdown(f"#### {speaker_label}")
                
                speaker_name = gr.Textbox(
                    label="Label (optional, e.g., 'John', 'Interviewer', 'Child')",
                    placeholder="Enter a name or role",
                    value=""
                )
                
                gender_radio = gr.Radio(
                    choices=["Male", "Female"],
                    value="Male",
                    label=f"Voice Gender for {speaker_label}"
                )
                
                if use_voice_cloning and speaker in reference_files:
                    ref_audio = reference_files[speaker]
                    if os.path.exists(ref_audio):
                        gr.Audio(value=ref_audio, label="Reference Audio Sample")
                        voice_option = gr.Radio(
                            choices=["Use voice cloning", "Use Edge TTS"],
                            value="Use voice cloning",
                            label=f"Voice Option for {speaker_label}"
                        )
                    else:
                        gr.Markdown("*No valid reference audio available for this speaker*")
                        voice_option = None
                else:
                    voice_option = None

            speaker_configs[speaker_id] = {
                "speaker": speaker,
                "use_cloning": use_voice_cloning and speaker in reference_files,
                "ref_audio": reference_files.get(speaker, None),
                "name_input": speaker_name,
                "gender_input": gender_radio,
                "voice_option_input": voice_option
            }

    speaker_info["speaker_configs"] = speaker_configs
    return gr.Column()

# Gradio UI setup
def main_ui():
    with gr.Blocks() as ui:
        gr.Markdown("## AI Dubbing System")
        video_input = gr.File(label="Upload MP4 Video")
        transcript_input = gr.File(label="Upload Transcript (.srt or .txt)")
        output_video = gr.File(label="Processed Video with Dubbed Audio")
        
        with gr.Row():
            gr.Button("Process Video", variant="primary")
            
        with gr.Column():
            create_speaker_ui()
        
        video_input.change(fn=create_speaker_ui, inputs=[], outputs=[])

    return ui

# Run the UI
if __name__ == "__main__":
    ui = main_ui()
    ui.launch()
