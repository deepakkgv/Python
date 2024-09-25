import os
import json
import base64
import torch
import gc
from io import BytesIO
from pydub import AudioSegment
from transformers import pipeline

# Load the model once when the service starts
def init():
    global whisper_pipeline

    # Load the Whisper model from the local path
    model_path = '/var/azureml-app/azureml-models/Whisper-HFL3/1/whisper-large-v3'
    device = 0 if torch.cuda.is_available() else -1

    # Log the device being used
    if device == 0:
        print("Using GPU for processing")
    else:
        print("Using CPU for processing")

    # Load the model with the correct device
    whisper_pipeline = pipeline("automatic-speech-recognition", model=model_path, device=device)

# Convert base64-encoded audio to WAV format
def base64_to_wav(base64_audio, wav_path):
    """Converts a base64-encoded audio to a WAV file."""
    try:
        audio_data = base64.b64decode(base64_audio)
        audio_io = BytesIO(audio_data)

        # Convert base64 M4A audio to WAV
        audio = AudioSegment.from_file(audio_io, format="m4a")
        audio.export(wav_path, format="wav")
    except Exception as e:
        raise RuntimeError(f"Failed to convert base64 audio to WAV: {str(e)}")

# Handle the request for transcription
def run(raw_data):
    try:
        # Parse the request body (assuming the input is JSON)
        data = json.loads(raw_data)

        # Check if "audio" or "audios" is in the request
        if "audio" in data:
            base64_audios = [data["audio"]]
        elif "audios" in data:
            base64_audios = data["audios"]
            if not isinstance(base64_audios, list):
                return json.dumps({"error": "'audios' should be a list of base64-encoded strings."})
        else:
            return json.dumps({"error": "No audio(s) provided in the request."})

        # Initialize a list to hold transcriptions
        transcriptions = []

        for idx, base64_audio in enumerate(base64_audios):
            # Temporary path for the WAV file, unique per audio
            wav_path = f"temp_audio_{idx}.wav"

            # Convert base64 to WAV
            base64_to_wav(base64_audio, wav_path)

            # Transcribe the audio file
            transcription = whisper_pipeline(wav_path)["text"]

            # Append the transcription to the list
            transcriptions.append(transcription)

            # Clean up the WAV file
            if os.path.exists(wav_path):
                os.remove(wav_path)

        # Return the transcriptions as JSON
        if len(transcriptions) == 1:
            # Return a single transcription for backward compatibility
            return json.dumps({"transcription": transcriptions[0]})
        else:
            return json.dumps({"transcriptions": transcriptions})

    except Exception as e:
        # Handle any errors and return the message
        return json.dumps({"error": str(e)})

    finally:
        # Free up memory
        gc.collect()
