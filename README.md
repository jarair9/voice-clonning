# 🎙️ Voice Cloning Studio

A comprehensive voice cloning application built with Streamlit that supports both English (Chatterbox TTS) and Hindi/Urdu (Coqui TTS) voice cloning.

## ✨ Features

- **Multi-lingual Support**: English, Hindi, and Urdu voice cloning
- **Dual TTS Engines**: Chatterbox TTS for English, Coqui TTS for multi-lingual
- **Beautiful UI**: Modern, intuitive interface with real-time feedback
- **Audio Upload**: Support for WAV, MP3, M4A formats
- **Voice Management**: Save, select, and manage multiple cloned voices
- **High Quality**: Configurable audio quality settings
- **Real-time Processing**: Fast voice cloning and speech generation

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run app.py
```

### 3. Open in Browser

Navigate to `http://localhost:8501` in your web browser.

## 📋 Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for faster processing)
- At least 8GB RAM
- 2GB free disk space

## 🛠️ Installation Guide

### Windows

1. **Install Python** (3.8 or higher)
   ```bash
   # Download from python.org
   # Make sure to check "Add Python to PATH"
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv voice_cloning_env
   voice_cloning_env\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the App**
   ```bash
   streamlit run app.py
   ```

### macOS/Linux

1. **Install Python** (3.8 or higher)
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install python3.8 python3.8-venv python3-pip
   
   # macOS
   brew install python@3.8
   ```

2. **Create Virtual Environment**
   ```bash
   python3 -m venv voice_cloning_env
   source voice_cloning_env/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the App**
   ```bash
   streamlit run app.py
   ```

## 🎯 Usage Guide

### 1. Clone a Voice

1. **Select Model**: Choose between Chatterbox TTS (English) or Coqui TTS (Multi-lingual)
2. **Upload Audio**: Upload a clear voice sample (3-30 seconds recommended)
3. **Enter Voice Name**: Give your cloned voice a unique name
4. **Click "Clone Voice"**: Wait for the process to complete

### 2. Generate Speech

1. **Select Cloned Voice**: Choose from your saved voices
2. **Enter Text**: Type or paste the text you want to convert
3. **Click "Generate Speech"**: Listen to the generated audio

### 3. Best Practices

- **Audio Quality**: Use high-quality, clear recordings
- **Background Noise**: Minimize background noise for better results
- **Sample Length**: 3-30 seconds is optimal
- **Single Speaker**: Ensure only one person is speaking
- **Clear Speech**: Use normal, clear speaking voice

## 🔧 Configuration

### Model Settings

- **Chatterbox TTS**: Optimized for English voice cloning
- **Coqui TTS**: Multi-lingual support (Hindi, Urdu, English)
- **Quality Levels**: 1-5 (Higher = better quality, slower processing)

### Audio Settings

- **Sample Rate**: 16kHz, 22.05kHz, or 44.1kHz
- **Bit Depth**: 16, 24, or 32 bits
- **File Formats**: WAV, MP3, M4A supported

## 🎨 UI Features

- **Responsive Design**: Works on desktop and mobile
- **Real-time Feedback**: Progress indicators and status messages
- **Audio Player**: Built-in audio playback
- **Voice Management**: Save, delete, and switch between voices
- **Dark/Light Mode**: Automatic theme detection

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Check internet connection
   - Verify sufficient disk space
   - Restart the application

2. **Audio Processing Errors**
   - Ensure audio file is in supported format
   - Check file size (max 50MB)
   - Try with a shorter audio sample

3. **Memory Issues**
   - Close other applications
   - Use lower quality settings
   - Restart the application

### Performance Tips

- **GPU Acceleration**: Enable CUDA for faster processing
- **Audio Quality**: Start with lower quality settings
- **Sample Length**: Use shorter audio samples for testing

## 📚 API Reference

### VoiceCloningManager

```python
from voice_cloning_engine import VoiceCloningManager

# Initialize manager
manager = VoiceCloningManager()

# Initialize models
manager.initialize_chatterbox()
manager.initialize_coqui()

# Clone voice
success = manager.clone_voice_english("audio.wav", "my_voice")

# Generate speech
audio = manager.generate_speech("Hello world!", "my_voice")
```

### AudioProcessor

```python
from voice_cloning_engine import AudioProcessor

# Load audio
audio, sr = AudioProcessor.load_audio("audio.wav")

# Extract features
features = AudioProcessor.extract_features(audio)
```

## 🔒 Privacy & Security

- **Local Processing**: All voice processing happens locally
- **No Cloud Upload**: Your voice data never leaves your computer
- **Temporary Files**: All temporary files are automatically deleted
- **Secure Storage**: Voice embeddings are stored securely

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Chatterbox TTS**: For English voice cloning capabilities
- **Coqui TTS**: For multi-lingual voice synthesis
- **Streamlit**: For the beautiful web interface
- **PyTorch**: For deep learning infrastructure

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the documentation

## 🔄 Updates

### Version 1.0.0
- Initial release
- English voice cloning with Chatterbox TTS
- Hindi/Urdu voice cloning with Coqui TTS
- Beautiful Streamlit interface
- Audio upload and processing
- Voice management system

---

**Note**: This application is for educational and research purposes. Please ensure you have the right to use any voice samples you upload.


Developer: Jarair khan
For Project Queries: jarairkhan211@gmail.com
whatsapp  : +923184254344