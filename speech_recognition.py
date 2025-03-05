import whisper
import os

class SpeechRecognizer:
    def __init__(self, model_size="base"):
        self.model = whisper.load_model(model_size)
        
    def transcribe(self, audio_path, language=None):
        """Transcribe audio file with timestamps"""
        # Get results with word timestamps
        result = self.model.transcribe(
            audio_path,
            language=language,
            word_timestamps=True,
            verbose=False
        )
        
        # Return segments with timestamps
        return result["segments"]