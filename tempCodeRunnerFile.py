
    
    # Enhanced voice configuration
    enhanced_config = {}
    
    for speaker_id, gender in voice_config.items():
        # Ensure speaker_id is an integer (convert if it's a string)
        if isinstance(speaker_id, str) and speaker_id.isdigit():
            speaker_id = int(speaker_id)
            
        if gender == "male":
            # Assign voice and pitch based on male index
            pitch = male_pitches[current_male % len(male_pitches)]
            enhanced_config[speaker_id] = {
                'voice': male_voice,
                'pitch': pitch
            }
            current_male += 1
        else:  # female
            # Assign voice and pitch based on female index
            pitch = female_pitches[current_female % len(female_pitches)]
            enhanced_config[speaker_id] = {
                'voice': female_voice,
                'pitch': pitch
            }
            current_female += 1
    
    return enhanced_config

def generate_edge_tts(segments, target_language, voice_config=None,output_dir="audio2"):
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
    
    voice_config = configure_voice_characteristics(voice_config)

    # Default voice configuration if none provided
    if voice_config is None:
        voice_config = {}
    
    # Process each segment
    for i, segment in enumerate(segments):
        # Extract speaker ID
        speaker = segment.get('speaker', 'SPEAKER_00')
        match = re.search(r'SPEAKER_(\d+)', speaker)
        speaker_id = int(match.group(1)) if match else 0
                
        voice = voice_config[speaker_id].get('voice', "hi-IN-SwaraNeural")
        pitch = voice_config[speaker_id].get('pitch', 0)
        
        # Get text and timing information
        text = segment['text']
        start = segment['start']
        end = segment['end']
        duration = end - start
        
        # Create output filename
        output_file = f"audio/{start}.wav"
        
        logger.info(f"Processing segment {i+1} (Speaker {speaker_id}, Voice: {voice}):")
        logger.info(f"  Text: {text[:50]}{'...' if len(text) > 50 else ''}")
        logger.info(f"  Duration: {duration:.2f}s")
        
        # Generate the voice
        create_segmented_edge_tts(
            text=text,
            pitch=0,
            voice=voice,
            output_path=output_file,
            target_duration=duration,
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
    logger.info(f"  Final combined duration: {len(combined) / 1000:.2f}s")
    
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


