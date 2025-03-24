import re
segments = [
    {
        "start": 1.5,           # float value
        "end": 3.2,             # float value
        "text": "Hello world",   # string value
        "speaker": "SPEAKER_0"   # string value
    },
    {
        "start": 4.1,
        "end": 6.7,
        "text": "How are you today?",
        "speaker": "SPEAKER_1"
    },
    # ... more dictionaries in the list
]
unique_speakers = set()
for segment in segments:
    if 'speaker' in segment:
        unique_speakers.add(segment['speaker'])

voice_config = {} 
for speaker in sorted(list(unique_speakers)):  
            match = re.search(r'SPEAKER_(\d+)', speaker)
            if match:
                speaker_id = int(match.group(1))
                gender = input(f"Select voice gender for Speaker {speaker_id+1} (m/f): ").lower()
                voice_config[speaker_id] = "female" if gender.startswith("f") else "male"