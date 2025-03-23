import numpy as np
import os
import re
import tempfile
import logging
from gtts import gTTS
from pydub import AudioSegment
from pathlib import Path


# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create directory structure
def ensure_directories():
    """Ensure the required directories exist"""
    directories = ["audio", "audio2"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
ensure_directories()  # Call immediately to ensure directories exist

# Setup audio effects for pydub
def setup_audio_effects():
    """Setup custom audio effects"""
    from pydub import effects
    
    # Add speedup if it's missing
    if not hasattr(AudioSegment, "speedup"):
        def speedup(audio_segment, playback_speed=1.5):
            if playback_speed <= 0 or playback_speed == 1.0:
                return audio_segment
            new_frame_rate = int(audio_segment.frame_rate * playback_speed)
            adjusted = audio_segment._spawn(audio_segment.raw_data, 
                                          overrides={'frame_rate': new_frame_rate})
            return adjusted.set_frame_rate(audio_segment.frame_rate)
        AudioSegment.speedup = speedup
    
    # Add time_stretch if it's missing
    if not hasattr(effects, "time_stretch"):
        def time_stretch(audio_segment, stretch_factor):
            if stretch_factor <= 0 or stretch_factor == 1.0:
                return audio_segment
            original_frame_rate = audio_segment.frame_rate
            new_frame_rate = int(original_frame_rate / stretch_factor)
            stretched = audio_segment._spawn(
                audio_segment.raw_data,
                overrides={'frame_rate': new_frame_rate}
            )
            return stretched.set_frame_rate(original_frame_rate)
        effects.time_stretch = time_stretch
    
    return effects

effects = setup_audio_effects()

def estimate_speech_rate(text, target_duration):
    """Estimate speech rate factor based on text length and target duration"""
    syllable_count = len(text.split()) * 3  # Rough approximation
    natural_duration = syllable_count / 4.5  # Using 4.5 syllables/sec
    
    if target_duration <= 0:
        return 1.0
    
    rate_factor = natural_duration / target_duration
    return max(0.7, min(1.8, rate_factor))  # Limit to reasonable range

def adjust_audio_duration(audio_segment, target_duration):
    """Adjust audio to target duration by adding silence or trimming"""
    current_duration = len(audio_segment) / 1000  # ms to seconds
    
    if current_duration < target_duration:
        silence_duration_ms = int((target_duration - current_duration) * 1000)
        silence = AudioSegment.silent(duration=silence_duration_ms)
        return audio_segment + silence
    else:
        return audio_segment[:int(target_duration * 1000)]

def create_voice_clone(text, gender, output_path, target_duration=None, language="hi"):
    """Create voice clone with specific characteristics and timing"""
    # Create base TTS
    tts = gTTS(text=text, lang=language, slow=False)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    tts.save(temp_file.name)
    temp_file.close()
    
    # Load audio
    audio = AudioSegment.from_file(temp_file.name)
    
    # Apply gender-specific voice characteristics
    if gender.lower() == 'male':
        # Male voice characteristics
        modified_audio = audio._spawn(audio.raw_data, overrides={
            "frame_rate": int(audio.frame_rate * 0.89)
        }).set_frame_rate(audio.frame_rate)
        modified_audio = modified_audio.low_pass_filter(2000)
    else:
        # Female voice characteristics
        modified_audio = audio._spawn(audio.raw_data, overrides={
            "frame_rate": int(audio.frame_rate * 1.12)
        }).set_frame_rate(audio.frame_rate)
        modified_audio = modified_audio.high_pass_filter(300)
    
    # Time constraint adjustment
    if target_duration is not None:
        current_duration = len(modified_audio) / 1000  # ms to seconds
        
        if abs(current_duration - target_duration) > 0.1:  # 100ms threshold
            speed_factor = current_duration / target_duration
            speed_factor = min(max(speed_factor, 0.7), 2.0)  # Keep within bounds
            
            logger.info(f"  Adjusting timing: {current_duration:.2f}s → {target_duration:.2f}s (factor: {speed_factor:.2f})")
            
            # Apply time adjustment
            if speed_factor > 1:
                modified_audio = modified_audio.speedup(playback_speed=speed_factor)
            else:
                modified_audio = effects.time_stretch(modified_audio, 1/speed_factor)
            
            # Fine-tune if needed
            new_duration = len(modified_audio) / 1000
            if abs(new_duration - target_duration) > 0.1:
                modified_audio = adjust_audio_duration(modified_audio, target_duration)
    
    # Save the modified audio
    modified_audio.export(output_path, format="wav")
    
    # Clean up temporary file
    os.unlink(temp_file.name)
    
    # Log final duration
    final_audio = AudioSegment.from_file(output_path)
    final_duration = len(final_audio) / 1000
    logger.info(f"  Final duration: {final_duration:.2f}s (target: {target_duration if target_duration else 'None'}s)")
    
    return output_path

def generate_speech(segments, target_language, voice_config=None,output_dir="audio2"):
    """
    Generate speech for all segments
    
    Args:
        segments: List of segments with text, speaker, start and end times
        target_language: Language code for TTS
        voice_config: Dictionary mapping speaker IDs to genders ('male'/'female')
        
    Returns:
        List of created audio files
    """
    # Ensure output directory exists
    os.makedirs(f"{output_dir}", exist_ok=True)
    # Generate the full audio
    output_path = os.path.join(output_dir, "dubbed_conversation.wav")
    max_end_time = max(segment['end'] for segment in segments)
    
    # Create a silent audio of the total duration
    combined = AudioSegment.silent(duration=int(max_end_time * 1000) + 100) 
    ensure_directories()
    audio_files = []
    
    # Default voice configuration if none provided
    if voice_config is None:
        voice_config = {}
    
    # Process each segment
    for i, segment in enumerate(segments):
        # Extract speaker ID
        speaker = segment.get('speaker', 'SPEAKER_00')
        match = re.search(r'SPEAKER_(\d+)', speaker)
        speaker_id = int(match.group(1)) if match else 0
        
        # Determine gender (default to alternating)
        gender = voice_config.get(speaker_id, 'female' if speaker_id % 2 else 'male')
        
        # Get text and timing information
        text = segment['text']
        start = segment['start']
        end = segment['end']
        duration = end - start
        
        # Create output filename
        output_file = f"audio/{start}.wav"
        
        logger.info(f"Processing segment {i+1} (Speaker {speaker_id}, {gender}):")
        logger.info(f"  Text: {text[:50]}{'...' if len(text) > 50 else ''}")
        logger.info(f"  Duration: {duration:.2f}s")
        
        # Generate the voice
        create_voice_clone(
            text=text,
            gender=gender,
            output_path=output_file,
            target_duration=duration,
            language=target_language
        )
        
        audio_files.append(output_file)

        # Add segment to combined audio at the exact timestamp
        segment_audio = AudioSegment.from_file(output_file)
        # Position in ms
        position_ms = int(segment['start'] * 1000)
        # Add to combined audio
        combined = combined.overlay(segment_audio, position=position_ms)
        # Export the final combined audio
    combined.export(output_path, format="wav")
    logger.info(f"  Final combined duration: {len(combined_audio) / 1000:.2f}s")
    
        # Clean up segment files
    for file in audio_files:
        try:
            os.remove(file)
        except:
            pass
    
    # Verify the final duration
    final_audio = AudioSegment.from_file(output_path)
    final_duration_sec = len(final_audio) / 1000
    
    print(f"\nTarget duration: {max_end_time:.2f} seconds")
    print(f"Actual duration: {final_duration_sec:.2f} seconds")
    
    # If the final audio is still too long, trim it
    if final_duration_sec > max_end_time + 0.1:  # Allow 100ms grace
        trimmed = final_audio[:int(max_end_time * 1000)]
        trimmed.export(output_path, format="wav")
        print(f"Trimmed to exactly {max_end_time:.2f} seconds")
            

    return output_path

def adjust_speech_timing(segments, max_speed=2.0, output_dir="audio2"):
    """
    Adjust the timing of speech segments to match original
    
    Args:
        segments: List of segments with start and end times
        max_speed: Maximum acceleration factor
        output_dir: Output directory
        
    Returns:
        tuple: (list of output files, list of speaker names)
    """
    # Create output directory
    os.makedirs(f"{output_dir}/audio", exist_ok=True)
    
    output_files = []
    speaker_names = []
    
    # Process each segment
    for segment in segments:
        start = segment["start"]
        end = segment["end"] 
        speaker = segment["speaker"]
        
        # Calculate target duration
        target_duration = end - start
        
        # Source and destination paths
        source_file = f"audio/{start}.wav"
        target_file = f"{output_dir}/audio/{start}.ogg"
        
        if os.path.exists(source_file):
            # Load audio
            audio = AudioSegment.from_file(source_file)
            actual_duration = len(audio) / 1000  # ms to seconds
            
            # Calculate speed adjustment
            speed_factor = actual_duration / target_duration if target_duration > 0 else 1.0
            
            # Apply reasonable limits
            if speed_factor > max_speed:
                speed_factor = max_speed
            elif 0.8 <= speed_factor <= 1.15:
                speed_factor = 1.0  # No adjustment needed for small differences
                
            # Round for simplicity
            speed_factor = round(speed_factor, 1)
            
            logger.info(f"Segment {start}-{end}: Adjusting with factor {speed_factor}")
            
            # Apply the adjustment
            if speed_factor == 1.0:
                adjusted_audio = audio
            elif speed_factor > 1.0:
                adjusted_audio = audio.speedup(playback_speed=speed_factor)
            else:
                adjusted_audio = effects.time_stretch(audio, 1/speed_factor)
            
            # Save the result
            adjusted_audio.export(target_file, format="ogg")
            output_files.append(target_file)
            
            # Format speaker name
            match = re.search(r'SPEAKER_(\d+)', speaker)
            speaker_num = int(match.group(1)) + 1 if match else 1
            speaker_names.append(f"Speaker {speaker_num}")
        else:
            logger.warning(f"Audio file not found: {source_file}")
    
    return output_files, speaker_names

def apply_voice_effects(segments, method="simple"):
    """
    Apply voice effects to make speakers more distinctive
    
    Args:
        segments: List of segments with speaker identifiers
        method: Effect method to use
    """
    logger.info(f"Applying voice effects using method: {method}")
    
    # Get unique speakers
    speakers_processed = set()
    
    # Process each segment
    for segment in segments:
        start = segment["start"]
        speaker = segment["speaker"]
        
        if speaker in speakers_processed:
            continue
            
        speakers_processed.add(speaker)
        
        # Extract speaker ID
        match = re.search(r'SPEAKER_(\d+)', speaker)
        speaker_id = int(match.group(1)) if match else 0
        
        # Find all files for this speaker
        speaker_files = []
        for seg in segments:
            if seg["speaker"] == speaker:
                file_path = f"audio2/audio/{seg['start']}.ogg"
                if os.path.exists(file_path):
                    speaker_files.append(file_path)
        
        logger.info(f"Applying effects to {len(speaker_files)} files for {speaker}")
        
        # Apply effects based on speaker ID
        for file_path in speaker_files:
            try:
                # Load audio
                audio = AudioSegment.from_file(file_path)
                
                # Apply speaker-specific effects
                if speaker_id % 2 == 0:  # Even speakers
                    # Slight pitch shift and EQ
                    modified = audio._spawn(audio.raw_data, overrides={
                        "frame_rate": int(audio.frame_rate * 1.03)
                    }).set_frame_rate(audio.frame_rate)
                    modified = modified.low_pass_filter(3000)
                else:  # Odd speakers
                    # Different pitch and EQ
                    modified = audio._spawn(audio.raw_data, overrides={
                        "frame_rate": int(audio.frame_rate * 0.97)
                    }).set_frame_rate(audio.frame_rate)
                    modified = modified.high_pass_filter(200)
                
                # Save back to the same file
                modified.export(file_path, format="ogg")
                logger.info(f"Applied effects to {file_path}")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
    
    logger.info("Voice effects applied successfully")
