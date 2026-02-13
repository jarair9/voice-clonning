# Voice Cloning Studio Configuration

# Application Settings
app_name = "Voice Cloning Studio"
version = "1.0.0"
debug = False

# Model Settings
chatterbox_model_path = "chatterbox-english-v1"  # Update this path to your actual model file/directory

# Audio Settings
default_sample_rate = 16000
max_audio_length = 30  # seconds
supported_formats = ["wav", "mp3", "m4a"]
max_file_size = 50  # MB

# Quality Settings
quality_levels = {
    1: {"name": "Fast", "description": "Quick processing, lower quality"},
    2: {"name": "Balanced", "description": "Good balance of speed and quality"},
    3: {"name": "High", "description": "High quality, moderate speed"},
    4: {"name": "Premium", "description": "Very high quality, slower"},
    5: {"name": "Ultra", "description": "Best quality, slowest"},
}

# Language Settings
supported_languages = {
    "english": {"name": "English", "model": "chatterbox"},
    # Add other languages here if your Chatterbox model supports them
}

# UI Settings
theme_colors = {
    "primary": "#667eea",
    "secondary": "#764ba2",
    "success": "#28a745",
    "warning": "#ffc107",
    "error": "#dc3545",
    "info": "#17a2b8",
}

# Processing Settings
batch_size = 1
max_workers = 4
timeout = 300  # seconds

# Storage Settings
temp_dir = "temp_outputs"
voices_dir = "cloned_voices"
uploads_dir = "uploads"

# Security Settings
allow_file_upload = True
max_concurrent_uploads = 5
allowed_mime_types = ["audio/wav", "audio/mpeg", "audio/mp3", "audio/mp4", "audio/m4a"]
