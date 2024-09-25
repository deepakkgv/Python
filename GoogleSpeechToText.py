import os
import io
import subprocess
from google.cloud import speech
from pydub import AudioSegment

print("Starting script execution")

# Set the path to your service account key file
service_account_path = r"D:\Python\caramel-galaxy-413606-280e3fb1bddb.json"
if os.path.exists(service_account_path):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_path
    print("Environment variable set for Google credentials")
else:
    print(f"Service account file not found: {service_account_path}")
    exit()

FFMPEG_PATH = r"C:\Users\deepakkumarg\Downloads\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"  # Replace with your FFmpeg path

def run_ffmpeg_command(command):
    try:
        print(f"Running command: {command}")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        output, error = process.communicate()
        if process.returncode != 0:
            raise Exception(f"ffmpeg error: {error.decode('utf-8')}")
        print(output.decode('utf-8'))
    except FileNotFoundError as e:
        raise Exception(f"FileNotFoundError: {str(e)} - Check if the ffmpeg path is correct.")
    except Exception as e:
        print(f"An error occurred while running ffmpeg: {e}")

def preprocess_audio(file_path, speed_factor):
    try:
        # Load your audio file
        audio = AudioSegment.from_file(file_path, format="m4a")
        
        # Convert audio file to mono and standardize sample rate to 44.1kHz
        audio = audio.set_channels(1).set_frame_rate(44100)
        
        # Normalize volume
        audio = audio.normalize()
        
        # Export the preprocessed audio to a temporary file
        temp_wav_path = "temp_audio_mono.wav"
        audio.export(temp_wav_path, format="wav")
        print("Audio file loaded, converted to mono, normalized, and exported to WAV format with 44.1kHz sample rate")

        # Apply noise reduction using FFmpeg
        noise_reduction_path = "temp_audio_mono_nr.wav"
        ffmpeg_command = f'"{FFMPEG_PATH}" -y -i "{temp_wav_path}" -af "highpass=f=200,lowpass=f=3000,afftdn=nf=-25" "{noise_reduction_path}"'
        run_ffmpeg_command(ffmpeg_command)
        print("Noise reduction completed.")

        # Slow down the noise-reduced wav file
        slowed_audio_path = "temp_audio_slowed.wav"
        ffmpeg_command = f'"{FFMPEG_PATH}" -y -i "{noise_reduction_path}" -filter:a "atempo={speed_factor}" "{slowed_audio_path}"'
        run_ffmpeg_command(ffmpeg_command)
        print("Slowing down audio completed.")

        return slowed_audio_path
    except Exception as e:
        print(f"An error occurred during audio preprocessing: {e}")
        return None

def split_audio(file_path, chunk_length_ms):
    audio = AudioSegment.from_wav(file_path)
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunk_path = f"chunk_{i // chunk_length_ms}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    return chunks

def transcribe_audio(file_path, speed_factor):
    try:
        processed_audio_path = preprocess_audio(file_path, speed_factor)
        if not processed_audio_path:
            print("Audio preprocessing failed")
            return

        chunk_length_ms = 59000  # 59 seconds chunks
        chunks = split_audio(processed_audio_path, chunk_length_ms)
        print(f"Audio file split into {len(chunks)} chunks.")

        client = speech.SpeechClient()
        full_transcript = ""

        for chunk_path in chunks:
            with io.open(chunk_path, "rb") as audio_file:
                content = audio_file.read()
            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=44100,
                language_code="en-US",
                enable_automatic_punctuation=True,
                model='video',
                use_enhanced=True
            )

            response = client.recognize(config=config, audio=audio)
            for result in response.results:
                full_transcript += result.alternatives[0].transcript + " "

        print("Transcript Paragraph:")
        print(full_transcript.strip())

        # Clean up temporary files
        if os.path.exists(processed_audio_path):
            os.remove(processed_audio_path)
        if os.path.exists("temp_audio_mono.wav"):
            os.remove("temp_audio_mono.wav")
        if os.path.exists("temp_audio_mono_nr.wav"):
            os.remove("temp_audio_mono_nr.wav")
        for chunk in chunks:
            if os.path.exists(chunk):
                os.remove(chunk)
        print("Temporary WAV files removed")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    file_path = r"D:\Audio Files\Pj_PROD.m4a"
    speed_factor = 0.75  # Adjust speed factor for slower speed
    print(f"Transcribing audio file: {file_path}")
    transcribe_audio(file_path, speed_factor)
    print("Transcription process completed")
