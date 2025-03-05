import yt_dlp
import os
from moviepy.editor import VideoFileClip

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