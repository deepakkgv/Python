import os
from pydub import AudioSegment
from transformers import pipeline

# Define the paths
m4a_path = r"D:\Audio Files\Bear_PROD.m4a"
wav_path = r"D:\Audio Files\Bear_PROD.wav"
model_path = r"D:\Python\Models\HugginFaceWhisperAI-L3\whisper-large-v3"

# Convert M4A to WAV using pydub
audio = AudioSegment.from_file(m4a_path, format="m4a")
audio.export(wav_path, format="wav")

# Load the Whisper model
whisper_pipeline = pipeline("automatic-speech-recognition", model=model_path)

# Transcribe the audio
transcription = whisper_pipeline(wav_path)["text"]

# Print the transcription
print("Transcription:")
print(transcription)

# Optionally, clean up by removing the WAV file if not needed
if os.path.exists(wav_path):
    os.remove(wav_path)
