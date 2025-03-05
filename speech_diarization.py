from pyannote.audio import Pipeline
import torch

class SpeakerDiarizer:
    def __init__(self, hf_token):
        """Initialize speaker diarization with HuggingFace token"""
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.0",
            use_auth_token=hf_token
        )
        
    def diarize(self, audio_path, min_speakers=1, max_speakers=None):
        """Identify speakers in audio file"""
        # Set diarization parameters
        params = {"min_speakers": min_speakers}
        if max_speakers:
            params["max_speakers"] = max_speakers
            
        # Run diarization
        try:
            print(f"Running diarization on {audio_path} ...")
            # Use self.pipeline instead of self.diarization_pipeline and pass the params
            diarization = self.pipeline(audio_path, **params)
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