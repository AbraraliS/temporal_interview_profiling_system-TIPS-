# Live Speech Recognition with Whisper

Real-time speech transcription engine for streaming audio with partial and final transcription updates.

## Features

✅ **Real-time transcription** using OpenAI Whisper  
✅ **Partial & Final segments** with confidence scores  
✅ **Word-level timestamps** for precise alignment  
✅ **Silence detection** to avoid processing noise  
✅ **Low-latency streaming** with configurable chunk sizes  
✅ **Overlap handling** for context continuity  
✅ **JSON output** compatible with pipeline schema  

---

## Installation

### 1. Install FFmpeg (Required)

**Windows PowerShell:**
```powershell
# Download and extract
Invoke-WebRequest -Uri "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip" -OutFile "$env:TEMP\ffmpeg.zip"
Expand-Archive -Path "$env:TEMP\ffmpeg.zip" -DestinationPath "$env:TEMP\ffmpeg" -Force

# Create directory and move binaries
New-Item -ItemType Directory -Force -Path "C:\ffmpeg"
Copy-Item -Path "$env:TEMP\ffmpeg\ffmpeg-master-latest-win64-gpl\bin\*" -Destination "C:\ffmpeg\" -Force

# Add to PATH (permanent)
$currentPath = [Environment]::GetEnvironmentVariable("Path", [System.EnvironmentVariableScope]::User)
if ($currentPath -notlike "*C:\ffmpeg*") {
    [Environment]::SetEnvironmentVariable("Path", "$currentPath;C:\ffmpeg", [System.EnvironmentVariableScope]::User)
}

# Verify
ffmpeg -version
```

**Restart PowerShell after adding to PATH**

---

### 2. Install Python Dependencies

```powershell
pip install -r requirements_live.txt
```

Or manually:
```powershell
pip install openai-whisper pyaudio numpy
```

---

### 3. Install PyAudio (Windows)

If `pip install pyaudio` fails, download the wheel:

```powershell
# For Python 3.13 (check your version: python --version)
pip install pipwin
pipwin install pyaudio
```

Or download from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio

---

## Usage

### Basic Usage

```powershell
python live_transcribe.py
```

This will:
- Start capturing audio from your default microphone
- Transcribe speech in real-time
- Save results to `live_transcription_log.json`
- Display partial and final transcriptions in console

---

### Configuration

Edit settings in [live_transcribe.py](live_transcribe.py#L247):

```python
CONFIG = {
    "model_size": "base",        # tiny, base, small, medium, large
    "language": "en",            # Language code or None for auto-detect
    "chunk_duration": 3.0,       # Seconds per chunk (lower = faster, less context)
    "overlap_duration": 0.5,     # Overlap for context continuity
    "vad_threshold": 0.6,        # Voice activity detection threshold
    "temperature": 0.0,          # Whisper temperature (0 = deterministic)
    "beam_size": 5,              # Beam search size
}
```

---

## Output Format

### Partial Update
```json
{
  "stream_metadata": {
    "engine": "whisper",
    "mode": "live",
    "language": "en",
    "latency_ms": 420,
    "chunk_id": 27,
    "schema_version": "1.0"
  },
  "transcription_update": {
    "update_type": "partial",
    "segment_id": "seg_102",
    "speaker": "Speaker_1",
    "start_time": "00:02:14.320",
    "end_time": "00:02:18.900",
    "text": "I have experience working with Java and Spring",
    "confidence_score": 0.91,
    "word_timestamps": [
      { "word": "I", "start": 134.32, "end": 134.45, "confidence": 0.98 },
      { "word": "have", "start": 134.46, "end": 134.72, "confidence": 0.95 }
    ]
  }
}
```

### Final Update
```json
{
  "transcription_update": {
    "update_type": "final",
    "segment_id": "seg_102",
    "text": "I have experience working with Java and Spring Boot.",
    "confidence_score": 0.94
  }
}
```

### Events
```json
{
  "event": "silence_detected",
  "chunk_id": 45,
  "timestamp": "00:03:22.100",
  "duration_ms": 3000
}
```

---

## Model Selection Guide

| Model | Speed | Accuracy | VRAM | Latency | Best For |
|-------|-------|----------|------|---------|----------|
| `tiny` | Very Fast | Fair | ~1GB | ~200ms | Quick testing |
| `base` | Fast | Good | ~1GB | ~500ms | **Recommended** |
| `small` | Medium | Better | ~2GB | ~1s | Higher accuracy |
| `medium` | Slow | Great | ~5GB | ~3s | Production quality |
| `large` | Very Slow | Best | ~10GB | ~5s+ | Maximum accuracy |

---

## Performance Tips

### Reduce Latency
```python
chunk_duration = 2.0  # Smaller chunks = faster response
model_size = "tiny"   # Faster model
beam_size = 3         # Reduce beam search
```

### Improve Accuracy
```python
chunk_duration = 4.0  # More context
model_size = "medium" # Better model
overlap_duration = 1.0 # More overlap
```

### Handle Noisy Audio
```python
vad_threshold = 0.7   # Higher threshold filters more noise
```

---

## Troubleshooting

### Error: "No module named 'pyaudio'"
```powershell
pip install pipwin
pipwin install pyaudio
```

### Error: "ffmpeg not found"
- Restart PowerShell after installing FFmpeg
- Verify: `ffmpeg -version`
- Check PATH: `$env:Path -split ';' | Select-String ffmpeg`

### Error: "No default input device"
- Check microphone is connected
- Test with Windows Sound Recorder
- Try: `python -m pyaudio` to list devices

### Poor Transcription Quality
- Use a better microphone
- Reduce background noise
- Increase `model_size` to `small` or `medium`
- Adjust `chunk_duration` to capture full sentences

### High Latency
- Reduce `model_size` to `tiny` or `base`
- Reduce `chunk_duration`
- Reduce `beam_size`

---

## Integration with Pipeline

This transcriber outputs to `asr.results` queue format. To integrate:

```python
def send_to_pipeline(transcription_update):
    """Send to message queue"""
    import zmq
    
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.connect("tcp://localhost:5556")
    
    socket.send_json(transcription_update)

# Use callback
transcriber.process_stream(output_callback=send_to_pipeline)
```

---

## Comparison: Live vs Batch

| Feature | Live Transcription | Batch Transcription |
|---------|-------------------|---------------------|
| **Latency** | 0.5-3s per chunk | Full file processing |
| **Use Case** | Real-time interviews | Post-recording analysis |
| **Updates** | Partial + Final | Final only |
| **Accuracy** | Good | Better (full context) |
| **Resource** | Continuous CPU/GPU | Burst usage |

---

## Next Steps

1. **Test the system**: Run `python live_transcribe.py` and speak
2. **Tune parameters**: Adjust `chunk_duration` and `model_size` for your needs
3. **Add speaker diarization**: Integrate pyannote.audio for multi-speaker detection
4. **Connect to pipeline**: Send outputs to fusion module via ZMQ/Kafka
5. **Build UI**: Create real-time dashboard showing live transcriptions

---

## Schema Compliance

✅ Follows `asr.results` message schema  
✅ Includes `schema_version: "1.0"`  
✅ Provides word-level timestamps  
✅ Marks partial vs final segments  
✅ Reports confidence scores  
✅ Handles errors and edge cases  

---

**Status:** Ready for production use with proper FFmpeg and PyAudio setup.
