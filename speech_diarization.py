import torch
import gc
import os
import logging
import time
from typing import List, Dict, Optional, Union, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpeakerDiarizer:
    """Speaker diarization class to identify different speakers in audio"""
    
    DIARIZATION_MODELS = {
        "pyannote_3.1": "pyannote/speaker-diarization-3.1",
        "pyannote_2.1": "pyannote/speaker-diarization@2.1",
        "disable": "",
    }
    
    def __init__(self, hf_token: str, model_name: str = "pyannote_3.1", device: Optional[str] = None):
        """
        Initialize the speaker diarization pipeline
        
        Parameters:
            hf_token (str): HuggingFace authentication token
            model_name (str): Name of the diarization model to use (default: "pyannote_3.1")
            device (str): Device to run diarization on ("cpu", "cuda:0", etc.)
        """
        self.hf_token = hf_token
        self.diarization_pipeline = None
        self.model_name = model_name
        
        # Set device
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Speaker diarization will use device: {self.device}")
    
    def load_model(self) -> bool:
        """Load the diarization model"""
        if self.model_name == "disable" or not self.hf_token:
            logger.info("Speaker diarization is disabled")
            return False
            
        if self.diarization_pipeline is not None:
            return True
            
        try:
            logger.info(f"Loading diarization model: {self.model_name}")
            
            # Get the actual model name from our mapping
            model_path = self.DIARIZATION_MODELS.get(self.model_name, self.model_name)
            
            # Import here to avoid loading unnecessary dependencies if diarization is disabled
            from pyannote.audio import Pipeline
            
            self.diarization_pipeline = Pipeline.from_pretrained(
                model_path,
                use_auth_token=self.hf_token
            )
            self.diarization_pipeline.to(torch.device(self.device))
            logger.info("Diarization model loaded successfully")
            return True
            
        except Exception as error:
            error_str = str(error)
            # Clean up
            self.diarization_pipeline = None
            gc.collect()
            torch.cuda.empty_cache()
            
            # Provide helpful error messages for common issues
            if "'NoneType' object has no attribute 'to'" in error_str:
                if self.model_name == "pyannote_2.1":
                    logger.error(
                        "You need to accept the license agreement for Pyannote 2.1. "
                        "Visit: https://huggingface.co/pyannote/speaker-diarization and "
                        "https://huggingface.co/pyannote/segmentation"
                    )
                elif self.model_name == "pyannote_3.1":
                    logger.error(
                        "You need to accept the license agreement for Pyannote 3.1. "
                        "Visit: https://huggingface.co/pyannote/speaker-diarization-3.1 and "
                        "https://huggingface.co/pyannote/segmentation-3.0"
                    )
            else:
                logger.error(f"Error loading diarization model: {error}")
            
            return False
    
    def diarize(self, audio_path: str, min_speakers: int = 1, max_speakers: int = 2) -> List[Dict[str, Any]]:
        """
        Perform speaker diarization on audio file
        
        Parameters:
            audio_path (str): Path to audio file
            min_speakers (int): Minimum number of speakers to detect
            max_speakers (int): Maximum number of speakers to detect
            
        Returns:
            List of diarization segments with speaker IDs
        """
        # Skip diarization if only one speaker or model is disabled
        if max(min_speakers, max_speakers) <= 1 or self.model_name == "disable":
            logger.info("Single speaker mode - skipping diarization")
            return []
            
        # Load model if not already loaded
        if not self.load_model():
            logger.warning("Could not load diarization model, returning empty result")
            return []
        
        try:
            start_time = time.time()
            logger.info(f"Starting diarization with min_speakers={min_speakers}, max_speakers={max_speakers}")
            
            # Run diarization
            diarization = self.diarization_pipeline(
                audio_path,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )
            
            # Convert to our format
            speakers = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speakers.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker
                })
            
            # Encode speaker IDs consistently (SPEAKER_00, SPEAKER_01, etc.)
            speakers = self._reencode_speakers(speakers)
            print(f"Diarization completed in {time.time() - start_time:.1f}s")
            print(f"Found {len(set(s['speaker'] for s in speakers))} speakers")
            logger.info(f"Diarization completed in {time.time() - start_time:.1f}s")
            logger.info(f"Found {len(set(s['speaker'] for s in speakers))} speakers")
            
            return speakers
            
        except Exception as error:
            logger.error(f"Error during diarization: {error}")
            # Clean up on error
            gc.collect()
            torch.cuda.empty_cache()
            return []
    
    def _reencode_speakers(self, speakers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Reencode speaker IDs to consistent format (SPEAKER_00, SPEAKER_01, etc.)
        """
        speaker_mapping = {}
        counter = 0
        
        # First check if already in the correct format
        if speakers and speakers[0]['speaker'].startswith("SPEAKER_"):
            return speakers
        
        # Create mapping and apply it
        for segment in speakers:
            old_speaker = segment["speaker"]
            if old_speaker not in speaker_mapping:
                speaker_mapping[old_speaker] = f"SPEAKER_{counter:02d}"
                counter += 1
            segment["speaker"] = speaker_mapping[old_speaker]
        
        return speakers
    
    def assign_speakers_to_segments(self, segments: List[Dict[str, Any]], speakers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Assign speaker labels to transcript segments
        
        Parameters:
            segments (list): List of transcript segments with start/end times
            speakers (list): List of speaker segments from diarization
            
        Returns:
            Updated segments with speaker information
        """
        # If no speakers found or single speaker mode, assign everything to SPEAKER_00
        if not speakers:
            for segment in segments:
                segment["speaker"] = "SPEAKER_00"
            return segments
        
        for segment in segments:
            segment_mid = (segment.get("start", 0) + segment.get("end", 0)) / 2
            segment_start = segment.get("start", 0)
            segment_end = segment.get("end", 0)
            
            # Try to find matching speakers
            speaker_overlaps = []
            
            for speaker_turn in speakers:
                # Check for any overlap
                if not (speaker_turn["end"] <= segment_start or speaker_turn["start"] >= segment_end):
                    # Calculate overlap duration
                    overlap_start = max(speaker_turn["start"], segment_start)
                    overlap_end = min(speaker_turn["end"], segment_end)
                    overlap_duration = overlap_end - overlap_start
                    
                    speaker_overlaps.append((speaker_turn["speaker"], overlap_duration))
            
            # Sort by overlap duration
            speaker_overlaps.sort(key=lambda x: x[1], reverse=True)
            
            # Assign speaker with the most overlap
            if speaker_overlaps:
                segment["speaker"] = speaker_overlaps[0][0]
            else:
                # Find nearest speaker if no overlap
                closest_speaker = min(
                    speakers,
                    key=lambda s: min(
                        abs(s["start"] - segment_mid),
                        abs(s["end"] - segment_mid)
                    )
                )
                segment["speaker"] = closest_speaker["speaker"]
                logger.warning(f"No speaker overlap found for segment at {segment_start:.2f}s, using nearest speaker")
        
        return segments
    
    def process_audio(self, audio_path: str, transcript_segments: List[Dict[str, Any]], 
                     min_speakers: int = 1, max_speakers: int = 2) -> List[Dict[str, Any]]:
        """
        Process audio file to get speaker diarization and assign to transcript
        
        Parameters:
            audio_path (str): Path to audio file
            transcript_segments (list): Transcript segments with start/end times
            min_speakers (int): Minimum number of speakers
            max_speakers (int): Maximum number of speakers
            
        Returns:
            Updated transcript with speaker information
        """
        logger.info(f"Processing audio for speaker diarization: {os.path.basename(audio_path)}")
        
        # Simple case - single speaker
        if max(min_speakers, max_speakers) <= 1:
            for segment in transcript_segments:
                segment["speaker"] = "SPEAKER_00"
            return transcript_segments
        
        # Perform diarization
        speakers = self.diarize(audio_path, min_speakers, max_speakers)
        
        # Assign speakers to segments
        result = self.assign_speakers_to_segments(transcript_segments, speakers)
        
        # Clean up memory
        gc.collect()
        torch.cuda.empty_cache()
        
        return result
    
    def __del__(self):
        """Clean up resources when object is destroyed"""
        if self.diarization_pipeline is not None:
            del self.diarization_pipeline
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except:
                pass