"""
Live Speech Recognition Engine using Whisper
Handles real-time audio streaming with partial and final transcriptions
"""

import whisper
import pyaudio
import numpy as np
import json
import threading
import queue
import time
from datetime import timedelta
from collections import deque
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


class LiveWhisperTranscriber:
    """Real-time speech transcription using Whisper"""
    
    def __init__(
        self,
        model_size="base",
        language="en",
        sample_rate=16000,
        chunk_duration=2.0,  # seconds per chunk
        overlap_duration=0.5,  # overlap between chunks for context
        vad_threshold=0.6,
        temperature=0.0,
        beam_size=5,
        enable_speaker_detection=False
    ):
        print(f"[INIT] Loading Whisper model: {model_size}...")
        self.model = whisper.load_model(model_size)
        self.language = language
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.vad_threshold = vad_threshold
        self.temperature = temperature
        self.beam_size = beam_size
        self.enable_speaker_detection = enable_speaker_detection
        
        # Audio stream setup
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # State tracking
        self.chunk_id = 0
        self.segment_id = 0
        self.stream_start_time = None
        self.is_running = False
        
        # Audio buffer
        self.audio_queue = queue.Queue()
        self.chunk_size = int(sample_rate * chunk_duration)
        self.overlap_size = int(sample_rate * overlap_duration)
        
        # Previous context for overlap
        self.previous_audio = deque(maxlen=self.overlap_size)
        
        # Transcription state
        self.partial_segments = {}
        self.final_segments = []
        self.last_text = ""
        
        print("[INIT] ✓ Whisper live transcriber ready")
    
    def format_timestamp(self, seconds):
        """Convert seconds to HH:MM:SS.mmm format"""
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for incoming audio"""
        if status:
            print(f"[AUDIO] Warning: {status}")
        
        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        self.audio_queue.put(audio_data)
        
        return (in_data, pyaudio.paContinue)
    
    def start_stream(self):
        """Start audio capture from microphone"""
        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=int(self.sample_rate * 0.1),  # 100ms buffers
                stream_callback=self.audio_callback
            )
            self.stream_start_time = time.time()
            self.is_running = True
            print("[STREAM] ✓ Audio capture started")
        except Exception as e:
            return {
                "error": f"Failed to start audio stream: {str(e)}",
                "timestamp": time.time()
            }
    
    def stop_stream(self):
        """Stop audio capture"""
        self.is_running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        print("[STREAM] Audio capture stopped")
    
    def collect_audio_chunk(self):
        """Collect audio chunk with overlap"""
        audio_frames = []
        
        # Add previous overlap
        if self.previous_audio:
            audio_frames.extend(self.previous_audio)
        
        # Collect new audio
        target_samples = self.chunk_size
        collected = len(audio_frames)
        
        timeout_start = time.time()
        while collected < target_samples and (time.time() - timeout_start) < 10:
            try:
                frame = self.audio_queue.get(timeout=0.1)
                audio_frames.extend(frame)
                collected = len(audio_frames)
            except queue.Empty:
                continue
        
        if collected == 0:
            return None
        
        # Convert to numpy array
        audio_chunk = np.array(audio_frames[:target_samples])
        
        # Store overlap for next chunk
        if len(audio_frames) >= self.overlap_size:
            self.previous_audio = deque(audio_frames[-self.overlap_size:], maxlen=self.overlap_size)
        
        return audio_chunk
    
    def detect_silence(self, audio_chunk):
        """Detect if chunk is mostly silence"""
        if audio_chunk is None or len(audio_chunk) == 0:
            return True
        
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        return rms < 0.01  # Silence threshold
    
    def transcribe_chunk(self, audio_chunk):
        """Transcribe a single audio chunk"""
        if audio_chunk is None:
            return None
        
        chunk_start = time.time()
        current_time = chunk_start - self.stream_start_time
        
        self.chunk_id += 1
        
        # Detect silence
        if self.detect_silence(audio_chunk):
            return {
                "event": "silence_detected",
                "chunk_id": self.chunk_id,
                "timestamp": self.format_timestamp(current_time),
                "duration_ms": int(self.chunk_duration * 1000)
            }
        
        # Transcribe with Whisper
        try:
            result = self.model.transcribe(
                audio_chunk,
                language=self.language,
                temperature=self.temperature,
                beam_size=self.beam_size,
                no_speech_threshold=self.vad_threshold,
                condition_on_previous_text=True,
                initial_prompt=self.last_text,
                word_timestamps=True,
                verbose=False
            )
            
            latency_ms = int((time.time() - chunk_start) * 1000)
            
            # Check if speech detected
            if not result.get('segments'):
                return {
                    "event": "no_speech_detected",
                    "chunk_id": self.chunk_id,
                    "timestamp": self.format_timestamp(current_time)
                }
            
            # Process segments
            outputs = []
            for segment in result['segments']:
                self.segment_id += 1
                
                # Calculate absolute timestamps
                seg_start = current_time + segment['start']
                seg_end = current_time + segment['end']
                
                text = segment['text'].strip()
                confidence = self._calculate_confidence(segment)
                
                # Build word timestamps
                word_timestamps = []
                if 'words' in segment:
                    for word_info in segment['words']:
                        word_timestamps.append({
                            "word": word_info.get('word', '').strip(),
                            "start": round(current_time + word_info.get('start', 0), 2),
                            "end": round(current_time + word_info.get('end', 0), 2),
                            "confidence": round(word_info.get('probability', 0.5), 2)
                        })
                
                # Determine if partial or final
                # Whisper doesn't explicitly mark this, but we can use heuristics:
                # - Low no_speech_prob = likely final
                # - High confidence = likely final
                is_final = (
                    segment.get('no_speech_prob', 1.0) < 0.3 and
                    confidence > 0.75
                )
                
                update_type = "final" if is_final else "partial"
                
                output = {
                    "stream_metadata": {
                        "engine": "whisper",
                        "mode": "live",
                        "language": result.get('language', self.language),
                        "latency_ms": latency_ms,
                        "chunk_id": self.chunk_id,
                        "schema_version": "1.0"
                    },
                    "transcription_update": {
                        "update_type": update_type,
                        "segment_id": f"seg_{self.segment_id}",
                        "speaker": "Speaker_1",  # TODO: Add diarization
                        "start_time": self.format_timestamp(seg_start),
                        "end_time": self.format_timestamp(seg_end),
                        "text": text,
                        "confidence_score": confidence,
                        "word_timestamps": word_timestamps
                    }
                }
                
                # Add warnings for low confidence
                if confidence < 0.5:
                    output["warning"] = "low_confidence_transcription"
                
                # Store for context
                if is_final:
                    self.final_segments.append(output)
                    self.last_text = text
                else:
                    self.partial_segments[f"seg_{self.segment_id}"] = output
                
                outputs.append(output)
            
            return outputs if len(outputs) > 1 else outputs[0] if outputs else None
            
        except Exception as e:
            return {
                "error": f"Transcription failed: {str(e)}",
                "chunk_id": self.chunk_id,
                "timestamp": self.format_timestamp(current_time)
            }
    
    def _calculate_confidence(self, segment):
        """Calculate confidence score from segment data"""
        # Whisper provides avg_logprob
        avg_logprob = segment.get('avg_logprob', -1.0)
        no_speech_prob = segment.get('no_speech_prob', 0.5)
        
        # Convert log prob to confidence (approximate)
        # avg_logprob ranges roughly from -1 to 0
        logprob_conf = max(0.0, min(1.0, (avg_logprob + 1.0)))
        
        # Combine with speech probability
        speech_conf = 1.0 - no_speech_prob
        
        # Weighted average
        confidence = 0.7 * logprob_conf + 0.3 * speech_conf
        
        return round(confidence, 2)
    
    def process_stream(self, output_callback=None, save_to_file=None):
        """
        Main processing loop
        
        Args:
            output_callback: Function to call with each transcription update
            save_to_file: Path to JSON file for logging all outputs
        """
        print("[PROCESSING] Starting live transcription...")
        print("[INFO] Speak into your microphone. Press Ctrl+C to stop.\n")
        
        file_handle = None
        if save_to_file:
            file_handle = open(save_to_file, 'w', encoding='utf-8')
            file_handle.write('[\n')
        
        try:
            while self.is_running:
                # Collect audio chunk
                audio_chunk = self.collect_audio_chunk()
                
                if audio_chunk is None:
                    continue
                
                # Transcribe
                result = self.transcribe_chunk(audio_chunk)
                
                if result:
                    # Handle multiple segments
                    results = result if isinstance(result, list) else [result]
                    
                    for res in results:
                        # Print to console
                        if 'transcription_update' in res:
                            update = res['transcription_update']
                            update_type = update.get('update_type', 'event')
                            text = update.get('text', '')
                            confidence = update.get('confidence_score', 0)
                            
                            prefix = "✓" if update_type == "final" else "..."
                            print(f"[{update['start_time']}] {prefix} {text} (conf: {confidence:.2f})")
                        
                        elif 'event' in res:
                            event = res['event']
                            if event != 'silence_detected':  # Don't spam silence
                                print(f"[EVENT] {event}")
                        
                        elif 'error' in res:
                            print(f"[ERROR] {res['error']}")
                        
                        # Callback
                        if output_callback:
                            output_callback(res)
                        
                        # Save to file
                        if file_handle:
                            json.dump(res, file_handle, indent=2)
                            file_handle.write(',\n')
                            file_handle.flush()
        
        except KeyboardInterrupt:
            print("\n[STOP] Stopping transcription...")
        
        finally:
            if file_handle:
                file_handle.write('\n]\n')
                file_handle.close()
                print(f"[SAVED] Transcription log saved to: {save_to_file}")
    
    def run(self, output_file="live_transcription_log.json"):
        """Start live transcription session"""
        # Start audio stream
        start_result = self.start_stream()
        if start_result and 'error' in start_result:
            print(f"[ERROR] {start_result['error']}")
            return
        
        try:
            # Process stream
            self.process_stream(save_to_file=output_file)
        finally:
            # Cleanup
            self.stop_stream()
            self.audio.terminate()
            
            # Print summary
            print("\n" + "="*60)
            print("SESSION SUMMARY")
            print("="*60)
            print(f"Total chunks processed: {self.chunk_id}")
            print(f"Total segments: {self.segment_id}")
            print(f"Final segments: {len(self.final_segments)}")
            print(f"Partial segments: {len(self.partial_segments)}")
            print("="*60)


def main():
    """Run live transcription"""
    
    # Configuration
    CONFIG = {
        "model_size": "base",  # Options: tiny, base, small, medium, large
        "language": "en",
        "sample_rate": 16000,
        "chunk_duration": 3.0,  # seconds
        "overlap_duration": 0.5,  # seconds
        "vad_threshold": 0.6,
        "temperature": 0.0,
        "beam_size": 5,
        "enable_speaker_detection": False
    }
    
    OUTPUT_FILE = "live_transcription_log.json"
    
    print("="*60)
    print("LIVE WHISPER TRANSCRIPTION ENGINE")
    print("="*60)
    print(f"Model: {CONFIG['model_size']}")
    print(f"Language: {CONFIG['language']}")
    print(f"Chunk duration: {CONFIG['chunk_duration']}s")
    print(f"Output file: {OUTPUT_FILE}")
    print("="*60 + "\n")
    
    # Create transcriber
    transcriber = LiveWhisperTranscriber(**CONFIG)
    
    # Run
    transcriber.run(output_file=OUTPUT_FILE)


if __name__ == "__main__":
    main()
