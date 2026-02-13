import torch
import torchaudio
import numpy as np
from pathlib import Path
import tempfile
import os
import time
import warnings
import re
try:
    import config
except ImportError:
    # Fallback if config is not found (e.g. running script directly)
    class Config:
        chatterbox_model_path = "chatterbox-english-v1"
    config = Config()

# Suppress warnings
warnings.filterwarnings("ignore")


class AudioProcessor:
    """Utility class for audio processing"""

    @staticmethod
    def load_audio(file_path, target_sample_rate=16000):
        """Load audio file and resample if needed"""
        try:
            audio, sample_rate = torchaudio.load(file_path)

            # Convert to mono if stereo
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            # Resample if needed
            if sample_rate != target_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    sample_rate, target_sample_rate
                )
                audio = resampler(audio)

            return audio.squeeze(), target_sample_rate
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None, None

    @staticmethod
    def preprocess_audio(audio_tensor, max_length=30):
        """Preprocess audio tensor for voice cloning"""
        sample_rate = 16000
        max_samples = max_length * sample_rate

        # Trim or pad audio
        if len(audio_tensor) > max_samples:
            audio_tensor = audio_tensor[:max_samples]
        elif len(audio_tensor) < max_samples:
            padding = max_samples - len(audio_tensor)
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))

        # Normalize audio
        if torch.max(torch.abs(audio_tensor)) > 0:
            audio_tensor = audio_tensor / torch.max(torch.abs(audio_tensor))

        return audio_tensor

    @staticmethod
    def extract_features(audio_tensor, sample_rate=16000):
        """Extract audio features for voice cloning"""
        try:
            # Extract MFCC features
            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=13,
                melkwargs={"n_mels": 40, "n_fft": 400},
            )
            mfcc_features = mfcc_transform(audio_tensor.unsqueeze(0))

            # Extract mel-spectrogram
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate, n_mels=80
            )
            mel_features = mel_transform(audio_tensor.unsqueeze(0))

            return {"mfcc": mfcc_features, "mel_spectrogram": mel_features}
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None


class ChatterboxTTSWrapper:
    """Wrapper for Chatterbox TTS engine"""

    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = config.chatterbox_model_path

    def load_model(self, model_path=None):
        """Load Chatterbox TTS model"""
        path_to_use = model_path if model_path else self.model_path
        
        print(f"🔄 Attempting to load Chatterbox model from: {path_to_use}")
        
        # Check if model path exists
        if not os.path.exists(path_to_use) and not os.path.exists(os.getcwd()):
             # This check is a bit loose because the user might rely on a system path or a library name
             # For now, we will try to load it, but log a specific warning
             print(f"⚠️ Warning: Model path '{path_to_use}' not found on disk. Assuming it's a library resource.")

        try:
            from chatterbox import ChatterboxTTS
            
            # Use separate method for pretrained loading
            self.model = ChatterboxTTS.from_pretrained(self.device)
            
            print(f"✅ Chatterbox TTS model loaded on {self.device}")
            return True
        except ImportError:
            print("❌ Chatterbox library not installed.")
            return False
        except Exception as e:
            print(f"Error loading Chatterbox TTS: {e}")
            return False

    def clone_voice(self, audio_path, speaker_name, language="english"):
        """Clone voice using Chatterbox"""
        if not self.model:
            print("❌ Model not loaded")
            return False
            
        try:
            # Placeholder for cloning logic
            print(f"✅ Voice '{speaker_name}' cloned from {audio_path}")
            return True
        except Exception as e:
            print(f"Error cloning voice: {e}")
            return False

    def synthesize_to_file(self, text, speaker_ref, output_path, **kwargs):
        """Synthesize speech to file"""
        if not self.model:
            print("❌ Model not loaded")
            return False

        try:
            # params from kwargs
            exaggeration = kwargs.get('exaggeration', 0.5)
            cfg_weight = kwargs.get('cfg_scale', 0.5) # Map cfg_scale to cfg_weight
            temperature = kwargs.get('temperature', 0.8)
            seed = kwargs.get('seed', 0)
            
            if seed > 0:
                torch.manual_seed(seed)
                np.random.seed(seed)

            print(f"🎤 Synthesizing with real Chatterbox engine (Exaggeration={exaggeration}, CFG={cfg_weight})...")
            
            # Real synthesis call
            audio_tensor = self.model.generate(
                text,
                audio_prompt_path=speaker_ref,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature
            ).cpu()
            
            # Ensure normalization to avoid clipping or silence
            if torch.max(torch.abs(audio_tensor)) > 0:
                audio_tensor = audio_tensor / torch.max(torch.abs(audio_tensor))
            
            # Save using torchaudio - Chatterbox S3Gen SR is 24000
            torchaudio.save(output_path, audio_tensor, 24000)
            
            print(f"✅ Speech generated: {output_path}")
            return True
        except Exception as e:
            print(f"Error synthesizing speech: {e}")
            return False


class VoiceCloningManager:
    """Main voice cloning manager"""

    def __init__(self):
        self.chatterbox = ChatterboxTTSWrapper()
        self.audio_processor = AudioProcessor()
        self.cloned_voices = {}
        self.chatterbox_loaded = False
        # Create temp directory
        os.makedirs("temp_outputs", exist_ok=True)

    def cleanup_old_temp_files(self, max_age_hours=24):
        """Cleanup files older than max_age_hours in temp_outputs"""
        temp_dir = "temp_outputs"
        if not os.path.exists(temp_dir):
            return
            
        now = time.time()
        count = 0
        for f in os.listdir(temp_dir):
            f_path = os.path.join(temp_dir, f)
            if os.path.isfile(f_path):
                # Don't delete files created in the last few minutes
                if os.stat(f_path).st_mtime < now - (max_age_hours * 3600):
                    try:
                        os.unlink(f_path)
                        count += 1
                    except:
                        pass
        if count > 0:
            print(f"🧹 Cleaned up {count} old temporary files.")

    def initialize_chatterbox(self, model_path=None):
        """Initialize Chatterbox TTS"""
        success = self.chatterbox.load_model(model_path)
        self.chatterbox_loaded = success
        return success

    def clone_voice(
        self, reference_audio_path, voice_name, language="english"
    ):
        """Clone voice using Chatterbox"""
        if not self.chatterbox_loaded:
            print("❌ Chatterbox model not loaded")
            return False

        try:
            success = self.chatterbox.clone_voice(reference_audio_path, voice_name, language)
            
            if success:
                # Store voice information
                self.cloned_voices[voice_name] = {
                    "speaker_name": voice_name,
                    "reference_audio": reference_audio_path,
                    "language": language,
                    "model_type": "chatterbox",
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                return True
            return False

        except Exception as e:
            print(f"Error cloning voice: {e}")
            return False

    def generate_speech(self, text, voice_name, output_path=None):
        """Generate speech using cloned voice (Legacy method)"""
        if voice_name not in self.cloned_voices:
            raise ValueError(f"Voice '{voice_name}' not found")

        voice_info = self.cloned_voices[voice_name]
        # Allow legacy calls to use the new direct method internally if needed, 
        # but for now we basically map it to the direct logic using the stored ref audio.
        return self.generate_speech_direct(
            text, 
            voice_info["reference_audio"], 
            output_path,
            language=voice_info.get("language", "english")
        )

    def generate_speech_direct(self, text, reference_audio_path, output_path=None, progress_callback=None, **kwargs):
        """
        Generate speech directly from text and reference audio.
        Supports long text by chunking.
        """
        if not self.chatterbox_loaded:
            print("❌ Chatterbox model not loaded")
            return None

        try:
            if output_path is None:
                # Use a cleaner naming convention
                timestamp = int(time.time())
                output_path = f"temp_outputs/voiceover_{timestamp}.wav"
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Use robust sentence-based chunking
            chunks = self._split_text_into_chunks(text, max_chars=400)
            print(f"📜 Split text into {len(chunks)} chunks for better processing.")

            total_chunks = len(chunks)
            if progress_callback:
                progress_callback(0.0)

            # Generate audio for each chunk
            temp_files = []
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                    
                chunk_path = output_path.replace(".wav", f"_chunk_{i}.wav")
                print(f"Processing chunk {i+1}/{total_chunks}...")
                
                success = self.chatterbox.synthesize_to_file(
                    chunk,
                    reference_audio_path,
                    chunk_path,
                    **kwargs
                )
                
                # Simulate processing time for progress bar visibility
                time.sleep(0.5) 

                if success:
                    temp_files.append(chunk_path)
                else:
                    print(f"❌ Failed to generate chunk {i}")
                
                if progress_callback:
                    progress_callback((i + 1) / total_chunks)

            if not temp_files:
                return None

            # Concatenate chunks
            if len(temp_files) == 1:
                # Rename the single chunk to output_path if it differs
                if temp_files[0] != output_path:
                    if os.path.exists(output_path):
                        os.unlink(output_path)
                    os.rename(temp_files[0], output_path)
            else:
                print("🔗 Concatenating chunks...")
                combined_audio = []
                sample_rate = 24000 # Chatterbox default
                
                for tf in temp_files:
                    audio, sr = torchaudio.load(tf)
                    if sr != sample_rate:
                        # Resample if somehow different
                        resampler = torchaudio.transforms.Resample(sr, sample_rate)
                        audio = resampler(audio)
                    
                    combined_audio.append(audio)
                    # Clean up chunk file
                    try:
                        os.unlink(tf)
                    except:
                        pass
                
                if combined_audio:
                    final_tensor = torch.cat(combined_audio, dim=1)
                    torchaudio.save(output_path, final_tensor, sample_rate)

            return output_path

        except Exception as e:
            print(f"Error generating direct speech: {e}")
            return None

    def _split_text_into_chunks(self, text, max_chars=400):
        """
        Split text into chunks based on sentence boundaries.
        Ensures chunks don't exceed max_chars while maintaining semantic integrity.
        """
        if not text:
            return []
            
        # Split by punctuation followed by space or newline (sentence boundaries)
        # Using lookbehind and lookahead to keep punctuation
        sentence_ends = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_ends, text.strip())
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If a single sentence is longer than max_chars, split it by segments
            if len(sentence) > max_chars:
                # If current chunk has content, add it first
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # Split long sentence by clauses or just length
                clauses = re.split(r'(?<=,)\s+', sentence)
                for clause in clauses:
                    if len(current_chunk) + len(clause) + 1 <= max_chars:
                        current_chunk += " " + clause
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        
                        # If clause itself is still too long, forced split
                        while len(clause) > max_chars:
                            chunks.append(clause[:max_chars].strip())
                            clause = clause[max_chars:]
                        current_chunk = clause
                continue

            if len(current_chunk) + len(sentence) + 1 <= max_chars:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return [c for c in chunks if c.strip()]

    def get_cloned_voices(self):
        """Get list of cloned voices"""
        return list(self.cloned_voices.keys())

    def get_voice_info(self, voice_name):
        """Get information about a cloned voice"""
        return self.cloned_voices.get(voice_name, None)

    def delete_voice(self, voice_name):
        """Delete a cloned voice"""
        if voice_name in self.cloned_voices:
            del self.cloned_voices[voice_name]
            return True
        return False
