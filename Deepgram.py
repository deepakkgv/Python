import os
import asyncio
import subprocess
from pydub import AudioSegment
from deepgram import Deepgram

# Your Deepgram API key
DEEPGRAM_API_KEY = '39a40856620c0a7daa338a5e24ea5cb86556cda3'

# Initialize the Deepgram SDK
dg_client = Deepgram(DEEPGRAM_API_KEY)

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

async def transcribe_audio(file_path):
    # Load your audio file
    audio = AudioSegment.from_file(file_path, format="m4a")
    
    # Apply gain to increase volume
    audio = audio.apply_gain(10)

    # Apply high pass filter
    audio = audio.high_pass_filter(200)
    
    # Apply low pass filter
    audio = audio.low_pass_filter(3000)

    temp_filtered_file = os.path.splitext(file_path)[0] + "_filtered.wav"
    audio.export(temp_filtered_file, format="wav")

    # Reduce background noise using FFmpeg
    noise_reduction_file = os.path.splitext(file_path)[0] + "_nr.wav"
    ffmpeg_noise_reduction_command = (
        f'ffmpeg -y -i "{temp_filtered_file}" -af "afftdn=nf=-30,highpass=f=200,lowpass=f=3000" "{noise_reduction_file}"'
    )
    run_ffmpeg_command(ffmpeg_noise_reduction_command)

    # Slow down the audio using FFmpeg with atempo filter
    temp_wav_file = "temp_audio.wav"
    ffmpeg_command = f'ffmpeg -y -i "{noise_reduction_file}" -filter:a "atempo=0.75" "{temp_wav_file}"'
    run_ffmpeg_command(ffmpeg_command)

    # Read the WAV file
    with open(temp_wav_file, "rb") as audio_file:
        audio_data = audio_file.read()

    # Create a Deepgram request
    response = await dg_client.transcription.prerecorded({
        'buffer': audio_data,
        'mimetype': 'audio/wav'
    }, {
        'punctuate': True,
        'language': 'en'
    })

    # Print the transcription result
    print(response['results']['channels'][0]['alternatives'][0]['transcript'])

    # Clean up temporary files
    if os.path.exists(temp_wav_file):
        os.remove(temp_wav_file)
    if os.path.exists(temp_filtered_file):
        os.remove(temp_filtered_file)
    if os.path.exists(noise_reduction_file):
        os.remove(noise_reduction_file)

# Example usage
if __name__ == "__main__":
    file_path = r"D:\Audio Files\Pj_PROD.m4a"
    asyncio.run(transcribe_audio(file_path))
