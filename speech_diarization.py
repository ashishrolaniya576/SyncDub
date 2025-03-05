from pyannote.audio import Pipeline
import torch
import os

class SpeakerDiarizer:
    def __init__(self, hf_token):
        """Initialize speaker diarization with HuggingFace token"""
        self.diarization_pipeline = None
        try:
            print("Loading diarization pipeline...")
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization@2.1",
                use_auth_token=hf_token
            )
            print("Diarization model loaded successfully!")
        except Exception as e:
            print(f"Error loading diarization model: {e}")
    
    def diarize(self, audio_path, min_speakers=1, max_speakers=None):
        """Identify speakers in audio file"""
        if not self.diarization_pipeline:
            print("Diarization pipeline not available")
            return []
        
        try:
            # Set parameters for diarization
            params = {}
            if min_speakers is not None:
                params["min_speakers"] = min_speakers
            if max_speakers is not None:
                params["max_speakers"] = max_speakers
                
            # Use the diarization pipeline
            diarization = self.diarization_pipeline(audio_path, **params)
            
            speakers = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speakers.append({'start': turn.start, 'end': turn.end, 'speaker': speaker})
            return speakers
        except Exception as e:
            print(f"Error during diarization: {e}")
            return []
    
    def assign_speakers_to_segments(self, segments, speakers):
        """Match transcription segments with speaker information"""
        for segment in segments:
            segment_mid = (segment["start"] + segment["end"]) / 2
            # Find speaker active at this time
            for speaker_turn in speakers:
                if speaker_turn["start"] <= segment_mid <= speaker_turn["end"]:
                    segment["speaker"] = speaker_turn["speaker"]
                    break
            else:
                segment["speaker"] = "unknown"
                
        return segments