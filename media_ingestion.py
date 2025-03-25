import yt_dlp
import os
from moviepy.editor import VideoFileClip
import subprocess
import shutil

class MediaIngester:
    def __init__(self, output_dir="temp"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def process_input(self, source):
        """Process various input types and extract audio"""
        if source.startswith(("http://", "https://")):
            # Handle URL (including YouTube)
            return self.download_from_url(source)
        elif os.path.isfile(source):
            # Handle local file
            return self.process_local_file(source)
        else:
            raise ValueError("Input source not recognized")
    
    def download_from_url(self, url):
        """Download media from URL (including YouTube)"""
        output_path = os.path.join(self.output_dir, "downloaded_video.mp4")
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
            'outtmpl': output_path,
            # Add additional options to make downloads more robust
            'retries': 10,
            'fragment_retries': 10,
            'ignoreerrors': False,
            'no_warnings': False,
            'geo_bypass': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return output_path
    
    def process_local_file(self, file_path):
        """Process local video or audio file"""
        # For simplicity, just return the path if it's a valid file
        return file_path
        
    def extract_audio(self, video_path):
        """Extract audio from video file"""
        audio_path = os.path.join(self.output_dir, "extracted_audio.wav")
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path)
        return audio_path
    


def separate_audio_sources(self, audio_path):
    """
    Separate voice and background music from an audio file using Demucs
    
    Parameters:
        audio_path (str): Path to the input audio file
        
    Returns:
        tuple: (voice_audio_path, background_music_path)
    """
    # Create output directory for separated audio
    separation_dir = os.path.join(self.output_dir, "separated")
    os.makedirs(separation_dir, exist_ok=True)
    
    # Final output paths
    voice_path = os.path.join(separation_dir, "voice.wav")
    music_path = os.path.join(separation_dir, "music.wav")
    
    try:
        # Method 1: Using Demucs as a command-line tool
        cmd = [
            "demucs", "--two-stems=vocals",
            "-o", separation_dir,
            audio_path
        ]
        
        print(f"Separating audio sources from {os.path.basename(audio_path)}...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Separation complete.")
        
        # Demucs creates a subdirectory with model name and then the base name
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        model_name = "htdemucs"  # default model
        demucs_output_dir = os.path.join(separation_dir, model_name, base_name)
        
        # Get the paths to the separated files
        actual_voice_path = os.path.join(demucs_output_dir, "vocals.wav") 
        actual_music_path = os.path.join(demucs_output_dir, "no_vocals.wav")
        
        # Move files to their final locations
        shutil.copy2(actual_voice_path, voice_path)
        shutil.copy2(actual_music_path, music_path)
        
        # Clean up if needed
        shutil.rmtree(os.path.join(separation_dir, model_name))
        
        return voice_path, music_path
        
    except Exception as e:
        print(f"Error during audio separation: {e}")
        
        # Method 2: Fall back to Python API
        try:
            print("Attempting separation using Python API...")
            import torch
            from demucs.pretrained import get_model
            from demucs.apply import apply_model
            import torchaudio
            
            # Load audio
            audio, sr = torchaudio.load(audio_path)
            
            # Convert to mono if needed
            if audio.shape[0] > 1:
                audio = audio.mean(0, keepdim=True)
            
            # Load model
            model = get_model('htdemucs')
            model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            
            # Apply separation
            sources = apply_model(model, audio, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            
            # Sources is a dictionary with keys "vocals" and "no_vocals"
            torchaudio.save(voice_path, sources[0].cpu(), sr)
            torchaudio.save(music_path, sources[1].cpu(), sr)
            
            return voice_path, music_path
        
        except Exception as e2:
            print(f"Python API separation also failed: {e2}")
            return None, None