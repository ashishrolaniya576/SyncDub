from pyannote.audio import Pipeline
import torch
import os
import time

class SpeakerDiarizer:
    def __init__(self, hf_token, device=None):
        """Initialize speaker diarization with HuggingFace token"""
        self.diarization_pipeline = None
        try:
            print("Loading diarization pipeline...")
            # Check available devices
            if device is None:
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
            
            # Use the newer version that's compatible with your libraries
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            self.diarization_pipeline.to(torch.device(device))
            print("Diarization model loaded successfully!")
        except Exception as e:
            print(f"Error loading diarization model: {e}")
    
    def diarize(self, audio_path, min_speakers=1, max_speakers=None, device=None):
        """Identify speakers in audio file"""
        if not self.diarization_pipeline:
            print("Diarization pipeline not available")
            return []
        
        try:
            print("Starting speaker diarization (this may take several minutes for longer files)...")
            start_time = time.time()
            
            # Set parameters for diarization
            params = {}
            if min_speakers is not None:
                params["min_speakers"] = min_speakers
            if max_speakers is not None:
                params["max_speakers"] = max_speakers
            
            # Set device if specified (cuda:0, cpu, etc.)
            if device:
                print(f"Using device: {device}")
                self.diarization_pipeline.to(torch.device(device))
            
            # Add progress updates
            print("Running diarization model...")
            print("This process may take several minutes with no visible progress...")
            print("Consider using a smaller audio segment for testing")
            
            # Use the diarization pipeline
            diarization = self.diarization_pipeline(audio_path, **params)
            
            print("Processing diarization results...")
            speakers = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speakers.append({'start': turn.start, 'end': turn.end, 'speaker': speaker})
            
            duration = time.time() - start_time
            print(f"Diarization completed in {duration:.1f} seconds")
            print(f"Detected {len(set(s['speaker'] for s in speakers))} unique speakers")
            
            return speakers
        except Exception as e:
            print(f"Error during diarization: {e}")
            return []
    
    def assign_speakers_to_segments(self, segments, speakers):
        """
        Assign speaker labels to transcript segments based on timing overlap
        
        Args:
            segments: List of transcript segments with start/end times
            speakers: List of speaker segments from diarization
            
        Returns:
            Updated segments with speaker information
        """
        # If no speakers found, assign everything to SPEAKER_0
        if not speakers:
            for segment in segments:
                segment["speaker"] = "SPEAKER_0"
            return segments
        
        # For single speaker, optimize by assigning all to same speaker
        if len(set(s["speaker"] for s in speakers)) == 1:
            speaker_id = speakers[0]["speaker"]
            for segment in segments:
                segment["speaker"] = speaker_id
            return segments
        
        # Process each segment
        for segment in segments:
            segment_start = segment.get("start", 0)
            segment_end = segment.get("end", 0)
            segment_duration = segment_end - segment_start
            
            # Find overlapping speakers
            speaker_overlaps = []
            
            for speaker_turn in speakers:
                # Fast check for any overlap
                if not (speaker_turn["end"] <= segment_start or speaker_turn["start"] >= segment_end):
                    # Calculate overlap duration
                    overlap_start = max(speaker_turn["start"], segment_start)
                    overlap_end = min(speaker_turn["end"], segment_end)
                    overlap_duration = overlap_end - overlap_start
                    
                    # Calculate overlap percentage relative to segment duration
                    overlap_percentage = overlap_duration / segment_duration if segment_duration > 0 else 0
                    
                    speaker_overlaps.append((speaker_turn["speaker"], overlap_duration, overlap_percentage))
            
            # Assign speaker with the most overlap
            if speaker_overlaps:
                # Sort by overlap duration (descending)
                speaker_overlaps.sort(key=lambda x: x[1], reverse=True)
                segment["speaker"] = speaker_overlaps[0][0]
                
                # Add confidence score if desired
                # segment["speaker_confidence"] = speaker_overlaps[0][2]
            else:
                # Find nearest speaker if no overlap
                segment_mid = (segment_start + segment_end) / 2
                
                closest_speaker = min(
                    speakers,
                    key=lambda s: min(
                        abs(s["start"] - segment_mid),
                        abs(s["end"] - segment_mid)
                    )
                )
                segment["speaker"] = closest_speaker["speaker"]
                
                # You can log this if logging is set up
                # print(f"No speaker overlap found for segment at {segment_start:.2f}s, using nearest speaker")
        
        return segments
    
    def extract_speaker_references(self, audio_path, speakers, output_dir="reference_audio", min_duration=3.0, max_duration=10.0):
        """
        Extract reference audio clips for each unique speaker.
        
        Args:
            audio_path: Path to the original audio file
            speakers: List of speaker segments from diarization
            output_dir: Directory to save reference audio clips
            min_duration: Minimum duration for a reference clip (seconds)
            max_duration: Maximum duration for a reference clip (seconds)
            
        Returns:
            Dictionary mapping speaker IDs to reference audio file paths
        """
        import os
        from pydub import AudioSegment
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the original audio file
        try:
            full_audio = AudioSegment.from_file(audio_path)
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return {}
        
        # Get unique speaker IDs
        unique_speakers = set(segment["speaker"] for segment in speakers)
        reference_files = {}
        
        print(f"Extracting reference audio for {len(unique_speakers)} speakers...")
        
        for speaker in unique_speakers:
            # Find all segments for this speaker
            speaker_segments = [s for s in speakers if s["speaker"] == speaker]
            
            # Sort segments by duration (descending)
            speaker_segments.sort(key=lambda s: s["end"] - s["start"], reverse=True)
            
            # Find a segment with suitable duration
            selected_segment = None
            for segment in speaker_segments:
                duration = segment["end"] - segment["start"]
                if duration >= min_duration:
                    # If longer than max_duration, trim it
                    if duration > max_duration:
                        mid_point = (segment["start"] + segment["end"]) / 2
                        half_max = max_duration / 2
                        segment = {
                            "start": mid_point - half_max,
                            "end": mid_point + half_max,
                            "speaker": speaker
                        }
                    selected_segment = segment
                    break
            
            # If no segment is long enough, take the longest one
            if selected_segment is None and speaker_segments:
                selected_segment = speaker_segments[0]
                
            # Extract the audio segment
            if selected_segment:
                start_ms = int(selected_segment["start"] * 1000)
                end_ms = int(selected_segment["end"] * 1000)
                
                # Extract audio segment
                speaker_audio = full_audio[start_ms:end_ms]
                
                # Save to file
                speaker_id = speaker.replace("SPEAKER_", "")
                output_path = os.path.join(output_dir, f"speaker_{speaker_id}_reference.wav")
                speaker_audio.export(output_path, format="wav")
                
                reference_files[speaker] = output_path
                
                print(f"  Extracted {selected_segment['end'] - selected_segment['start']:.2f}s reference audio for {speaker}")
            else:
                print(f"  No suitable audio segment found for {speaker}")
        
        return reference_files
