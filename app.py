import streamlit as st
import os
import tempfile
import time
from pathlib import Path
import torch
import torchaudio
import numpy as np
from io import BytesIO
import base64
import warnings

# Top-level error catching for Streamlit Cloud diagnostics
try:
    import streamlit as st
except Exception as e:
    print(f"FATAL: Streamlit import failed: {e}")

# Suppress warnings for cleaner UI
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Import voice cloning engine
from voice_cloning_engine import VoiceCloningManager, AudioProcessor

# Import TTS libraries
try:
    import chatterbox
    CHATTERBOX_AVAILABLE = True
except ImportError:
    CHATTERBOX_AVAILABLE = False

# Cached model loader for performance
@st.cache_resource
def get_voice_manager():
    """Create and initialize the voice manager, cached across reruns."""
    return VoiceCloningManager()

# Page configuration
st.set_page_config(
    page_title="Voice Cloning Studio",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for beautiful UI (Dark Theme)
st.markdown(
    """
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.4);
    }
    .stButton > button {
        width: 100%;
        background: #4f46e5;
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: #4338ca;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
    }
    .history-card {
        background: #1a1b23;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #303139;
        margin-bottom: 1rem;
    }
    .compact-text {
        font-size: 0.9rem;
        color: #9ca3af;
    }
</style>
""",
    unsafe_allow_html=True,
)


class VoiceCloningApp:
    def __init__(self):
        # Use singleton pattern for the manager
        self.voice_manager = get_voice_manager()
        self.setup_session_state()

    def setup_session_state(self):
        """Initialize session state variables"""
        # voice_manager is handled in __init__ to avoid circular dependency
        if "cloned_voices" not in st.session_state:
            st.session_state.cloned_voices = {}
        if "current_voice" not in st.session_state:
            st.session_state.current_voice = None
        if "chatterbox_loaded" not in st.session_state:
            st.session_state.chatterbox_loaded = False
        if "processing" not in st.session_state:
            st.session_state.processing = False
        if "generated_audio" not in st.session_state:
            st.session_state.generated_audio = None
        if "generation_history" not in st.session_state:
            st.session_state.generation_history = []

    def load_chatterbox_model(self):
        """Load Chatterbox TTS model with caching"""
        if not CHATTERBOX_AVAILABLE:
            st.error("❌ Chatterbox TTS is not available.")
            return False

        try:
            st.info("💡 **Note:** On Streamlit Cloud, the first load downloads ~2GB of AI weights. This can take up to 5 minutes. Please do not refresh the page.")
            
            with st.status("� System Initialization", expanded=True) as status:
                st.write("🔍 Checking environment...")
                time.sleep(1)
                
                st.write("📥 Loading AI Model weights (Chatterbox english-v1)...")
                # Use the existing manager instance
                success = self.voice_manager.initialize_chatterbox()
                
                if success:
                    st.write("✨ Optimizing audio pipeline...")
                    st.session_state.chatterbox_loaded = True
                    # Cleanup old files on first successful load
                    self.voice_manager.cleanup_old_temp_files()
                    status.update(label="✅ System Ready!", state="complete", expanded=False)
                    time.sleep(1)
                    st.rerun()
                    return True
                else:
                    status.update(label="❌ Initialization Failed", state="error")
                    st.error("❌ Failed to load Chatterbox TTS model. This might be due to RAM limits on Streamlit Cloud.")
                    return False
        except Exception as e:
            st.error(f"❌ Critical Error during initialization: {str(e)}")
            return False

    # Legacy methods removed for cleaner code

    def render_header(self):
        """Render app header"""
        st.markdown(
            """
        <div class="main-header">
            <h1>🎙️ Voice Cloning Studio</h1>
            <p>Advanced AI-powered voice cloning using Chatterbox TTS</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    def render_sidebar(self):
        """Render sidebar with configuration"""
        st.sidebar.markdown("## ⚙️ Configuration")
        
        # Model Status in Sidebar
        if st.session_state.chatterbox_loaded:
            st.sidebar.success("✅ Model Loaded")
        else:
            st.sidebar.warning("⚠️ Model Not Loaded")

        # Guidelines
        st.sidebar.markdown("### 💡 Quick Tips")
        st.sidebar.info("""
        • Use clear audio samples
        • 3-30 seconds recommended
        • Minimal background noise
        • Single speaker only
        """)

    # Legacy methods removed


    def render_main_content(self):
        """Render main content area"""
        
        # Top controls
        st.info("💡 Direct Mode: Upload audio, enter text, and generate instantly.")

        if not st.session_state.chatterbox_loaded:
             st.warning("👋 Welcome! Please initialize the AI engine to begin.")
             if st.button("🚀 Initialize System", type="primary", use_container_width=True):
                 self.load_chatterbox_model()
             st.caption("ℹ️ Note: Loading weights takes about personal 1-2 minutes on first run.")
             return

        # If we reach here, model is loaded, so show a mini success indicator in sidebar if not already there
        # (Though st.success in sidebar might be annoying, maybe just a status text)

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("### 1. Reference Audio")
            audio_file = st.file_uploader(
                "Upload a voice sample (WAV, MP3)", 
                type=["wav", "mp3", "m4a"],
                help="This audio will be cloned"
            )
            
            if audio_file:
                st.audio(audio_file)
                
            st.markdown("### 2. Script")
            text_input = st.text_area(
                "Enter text to speak", 
                height=300,
                placeholder="Enter your script here. Long texts will be automatically chunked..."
            )
            st.caption(f"Character count: {len(text_input)}")

        with col_right:
            st.markdown("### 3. Settings")
            
            language = st.selectbox(
                "Target Language",
                ["english", "hindi", "urdu", "multilingual"],
                help="Select target language for voice cloning",
            )
            
            # Exaggeration (Neutral = 0.5)
            st.markdown("### 🎚️ Exaggeration (Neutral = 0.5)")
            exaggeration = st.slider(
                "Exaggeration",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                label_visibility="collapsed",
                help="Controls the expressiveness of the voice. 0.5 is neutral."
            )

            # CFG/Pace Control
            st.markdown("### ⏱️ CFG/Pace Control")
            cfg_scale = st.slider(
                "CFG/Pace Control",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                label_visibility="collapsed",
                help="Controls the pace and consistency of the generation."
            )

            with st.expander("🔧 Advanced Options", expanded=False):
                col_adv1, col_adv2 = st.columns(2)
                
                with col_adv1:
                    st.markdown("**🎲 Random seed (0 = random)**")
                    seed = st.number_input(
                        "Random seed",
                        value=0,
                        step=1,
                        label_visibility="collapsed",
                        help="Set a specific seed for reproducible results. 0 for random."
                    )

                with col_adv2:
                    st.markdown("**🌡️ Temperature (higher = more random)**")
                    temperature = st.slider(
                        "Temperature",
                        min_value=0.05,
                        max_value=1.0,
                        value=0.8,
                        step=0.05,
                        label_visibility="collapsed"
                    )

            st.markdown("### 4. Generate")
            generate_btn = st.button("🚀 Generate Voiceover", type="primary", disabled=not (audio_file and text_input))

            if generate_btn:
                # Clear previous audio
                st.session_state.generated_audio = None
                
                if not audio_file:
                    st.error("❌ Please upload a reference audio file.")
                elif not text_input.strip():
                    st.error("❌ Please enter some text.")
                else:
                    self.process_generation(
                        text_input, 
                        audio_file, 
                        language=language,
                        exaggeration=exaggeration,
                        cfg_scale=cfg_scale,
                        seed=seed,
                        temperature=temperature
                    )
            
            # Display History
            self.render_history()
            
            # Display persisted result (latest generation)
            if st.session_state.generated_audio and os.path.exists(st.session_state.generated_audio):
                st.markdown("---")
                st.markdown("### 🎧 Result")
                st.audio(st.session_state.generated_audio, format="audio/wav")
                
                with open(st.session_state.generated_audio, "rb") as file:
                    st.download_button(
                        label="📥 Download Audio",
                        data=file.read(),
                        file_name=f"cloned_voice_{int(time.time())}.wav",
                        mime="audio/wav",
                    )

    def render_history(self):
        """Render generation history with download and delete capabilities"""
        if not st.session_state.generation_history:
            return
            
        st.markdown("---")
        st.markdown("### 🕒 Recent Generations")
        
        # Reverse history to show newest first
        for i, item in enumerate(reversed(st.session_state.generation_history)):
            idx = len(st.session_state.generation_history) - 1 - i
            
            with st.container():
                st.markdown(f"""<div class="history-card">""", unsafe_allow_html=True)
                
                col_text, col_actions = st.columns([4, 1])
                
                with col_text:
                    st.markdown(f"**{item['timestamp']}**")
                    st.markdown(f"""<p class="compact-text">{item['text'][:120]}{'...' if len(item['text']) > 120 else ''}</p>""", unsafe_allow_html=True)
                    if os.path.exists(item['path']):
                        st.audio(item['path'], format="audio/wav")
                
                with col_actions:
                    if os.path.exists(item['path']):
                        with open(item['path'], "rb") as f:
                            st.download_button("📥", f.read(), os.path.basename(item['path']), "audio/wav", key=f"dl_{idx}")
                    
                    if st.button("🗑️", key=f"del_{idx}"):
                        if os.path.exists(item['path']):
                            try: os.unlink(item['path'])
                            except: pass
                        st.session_state.generation_history.pop(idx)
                        if st.session_state.generated_audio == item['path']:
                            st.session_state.generated_audio = None
                        st.rerun()
                
                st.markdown("</div>", unsafe_allow_html=True)

    def process_generation(self, text, audio_file, **kwargs):
        """Handle the generation process"""
        try:
             # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_file.getvalue())
                tmp_file_path = tmp_file.name

            with st.spinner("🔊 Generating High-Fidelity Speech..."):
                start_time = time.time()
                
                # Create progress bar
                progress_bar = st.progress(0, text="Starting generation...")
                
                def update_progress(p):
                    progress_value = min(max(p, 0.0), 1.0)
                    progress_bar.progress(progress_value, text=f"Generating... {int(progress_value*100)}%")

                output_path = self.voice_manager.generate_speech_direct(
                    text, 
                    tmp_file_path, 
                    output_path=None,
                    progress_callback=update_progress,
                    **kwargs
                )
                
                # Complete the progress bar
                progress_bar.progress(1.0, text="Finalizing...")
                time.sleep(0.5) # Brief pause to show completion
                progress_bar.empty()
                
                # Cleanup reference
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass

                if output_path and os.path.exists(output_path):
                    elapsed = time.time() - start_time
                    st.success(f"✅ Generated in {elapsed:.2f}s")
                    
                    # Store in session state for persistence
                    st.session_state.generated_audio = output_path
                    
                    # Add to history
                    history_item = {
                        "id": int(time.time()),
                        "timestamp": time.strftime("%H:%M:%S"),
                        "text": text,
                        "path": output_path
                    }
                    st.session_state.generation_history.append(history_item)
                    
                    # Trigger rerun to show audio outside the spinner/button block
                    st.rerun()
                else:
                    st.error("❌ Generation failed. Check console logs.")

        except Exception as e:
            st.error(f"❌ Error during generation: {str(e)}")

    def run(self):
        """Run the Streamlit app"""
        self.render_header()
        with st.sidebar:
            self.render_sidebar()
            
        self.render_main_content()


# Main execution
if __name__ == "__main__":
    app = VoiceCloningApp()
    app.run()
