import whisper
import json
from datetime import timedelta
import os

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    secs = td.seconds % 60
    millis = td.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def transcribe_audio(audio_path, model_size="base", language=None):
    """
    Transcribe audio file using OpenAI Whisper
    
    Args:
        audio_path: Path to audio file
        model_size: tiny, base, small, medium, large (larger = more accurate but slower)
        language: Language code (e.g., 'en') or None for auto-detect
    """
    
    # Check if file exists
    if not os.path.exists(audio_path):
        return {
            "error": f"Audio file not found: {audio_path}",
            "confidence": "low"
        }
    
    print(f"Loading Whisper model: {model_size}...")
    model = whisper.load_model(model_size)
    
    print(f"Transcribing: {audio_path}...")
    result = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=True,
        verbose=False
    )
    
    # Detect audio quality based on average log probability
    avg_logprob = sum([seg.get('avg_logprob', -1) for seg in result['segments']]) / len(result['segments'])
    if avg_logprob > -0.5:
        quality = "clear"
    elif avg_logprob > -1.0:
        quality = "moderate"
    else:
        quality = "noisy"
    
    # Build output JSON
    output = {
        "schema_version": "1.0",
        "transcription_metadata": {
            "language": result.get('language', 'unknown'),
            "audio_quality": quality,
            "speaker_count": 1,  # Whisper doesn't do diarization by default
            "source_type": "audio",
            "timestamp_format": "HH:MM:SS",
            "model_used": f"whisper-{model_size}",
            "audio_duration": result.get('segments', [{}])[-1].get('end', 0) if result.get('segments') else 0
        },
        "segments": []
    }
    
    # Process each segment
    for idx, segment in enumerate(result['segments'], 1):
        seg_data = {
            "segment_id": idx,
            "speaker": "Speaker_1",  # Default, needs diarization for multi-speaker
            "start_time": format_timestamp(segment['start']),
            "end_time": format_timestamp(segment['end']),
            "transcript": segment['text'].strip(),
            "confidence_score": round(min(1.0, max(0.0, (segment.get('avg_logprob', -1) + 1) / 1)), 2),
            "notes": None
        }
        
        # Add notes for low confidence or no-speech segments
        if segment.get('no_speech_prob', 0) > 0.8:
            seg_data['notes'] = "[silence or background noise]"
        elif seg_data['confidence_score'] < 0.5:
            seg_data['notes'] = "low confidence transcription"
        
        output['segments'].append(seg_data)
    
    return output

def save_transcription(output, json_path):
    """Save transcription to JSON file"""
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Transcription saved to: {json_path}")

if __name__ == "__main__":
    # Configuration
    AUDIO_FILE = "Speaker26_000.wav"
    OUTPUT_FILE = "Speaker26_000_transcription.json"
    MODEL_SIZE = "base"  # Options: tiny, base, small, medium, large
    LANGUAGE = "en"  # Set to None for auto-detection
    
    # Transcribe
    result = transcribe_audio(AUDIO_FILE, MODEL_SIZE, LANGUAGE)
    
    # Handle errors
    if "error" in result:
        print(f"ERROR: {result['error']}")
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(result, f, indent=2)
    else:
        # Save to file
        save_transcription(result, OUTPUT_FILE)
        
        # Print summary
        print("\n" + "="*60)
        print("TRANSCRIPTION SUMMARY")
        print("="*60)
        print(f"Language: {result['transcription_metadata']['language']}")
        print(f"Quality: {result['transcription_metadata']['audio_quality']}")
        print(f"Segments: {len(result['segments'])}")
        print(f"Duration: {result['transcription_metadata']['audio_duration']:.2f}s")
        print("\nFirst 3 segments:")
        for seg in result['segments'][:3]:
            print(f"  [{seg['start_time']} - {seg['end_time']}] {seg['transcript']}")
