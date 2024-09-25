import os
import subprocess
from pydub import AudioSegment
from pydub.effects import low_pass_filter, high_pass_filter
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import soundfile as sf

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

def process_audio_pydub(input_file, output_file, speed_factor):
    try:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        print("Loading the audio file...")
        audio = AudioSegment.from_file(input_file, format="m4a")
        print("Audio file loaded.")

        print("Applying gain to increase volume...")
        audio = audio.apply_gain(10)
        print("Gain applied.")

        print("Applying high pass filter...")
        audio = high_pass_filter(audio, cutoff=200)
        print("High pass filter applied.")
        
        print("Applying low pass filter...")
        audio = low_pass_filter(audio, cutoff=3000)
        print("Low pass filter applied.")

        temp_filtered_file = os.path.splitext(input_file)[0] + "_filtered.wav"
        print(f"Exporting the filtered audio to temporary file: {temp_filtered_file}...")
        audio.export(temp_filtered_file, format="wav")
        print("Export to temporary file done.")

        print("Reducing background noise using FFmpeg...")
        noise_reduction_file = os.path.splitext(input_file)[0] + "_nr.wav"
        ffmpeg_noise_reduction_command = (
            f'ffmpeg -y -i "{temp_filtered_file}" -af "afftdn=nf=-30,highpass=f=200,lowpass=f=3000" "{noise_reduction_file}"'
        )
        run_ffmpeg_command(ffmpeg_noise_reduction_command)
        audio = AudioSegment.from_file(noise_reduction_file, format="wav")
        print("Noise reduction applied.")

        temp_wav_file = os.path.splitext(output_file)[0] + "_temp.wav"
        print(f"Exporting the processed audio to WAV file: {temp_wav_file}...");
        audio.export(temp_wav_file, format="wav")
        print("Export to WAV file done.")

        print(f"Slowing down the audio using FFmpeg with atempo filter...");
        final_wav_file = os.path.splitext(output_file)[0] + "_final.wav"
        ffmpeg_command = f'ffmpeg -y -i "{temp_wav_file}" -filter:a "atempo={speed_factor}" "{final_wav_file}"'
        run_ffmpeg_command(ffmpeg_command)
        print("Audio slowed down using FFmpeg.")

        # Clean up temporary files
        if os.path.exists(temp_wav_file):
            os.remove(temp_wav_file)
            print("Temporary WAV file deleted.")
        if os.path.exists(temp_filtered_file):
            os.remove(temp_filtered_file)
            print("Temporary filtered WAV file deleted.")
        if os.path.exists(noise_reduction_file):
            os.remove(noise_reduction_file)
            print("Temporary noise-reduced WAV file deleted.")

        return final_wav_file

    except Exception as ex:
        print(f"An error occurred: {ex}")
        return None

def transcribe_audio_whisper(audio_file, local_model_dir=r"D:\Python\Models\HugginFaceWhisperAI-L3\whisper-large-v3"):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Load the model and processor from the local directory
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        local_model_dir, torch_dtype=torch_dtype, low_cpu_mem_usage=True
    ).to(device)

    processor = AutoProcessor.from_pretrained(local_model_dir)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    result = pipe(audio_file, return_timestamps=True)
    return result["text"]

if __name__ == "__main__":
    input_m4a_file = r"D:\Audio Files\Pj_PROD.m4a"  # Your specified input file path
    output_m4a_file = r"D:\Audio Files\Pj_PYDUB.m4a"  # Output file path (not used in final transcription)
    final_wav_file = r"D:\Audio Files\Pj_PYDUB_final.wav"  # Final WAV file for transcription

    # Check if input file exists
    if not os.path.exists(input_m4a_file):
        print(f"Input file does not exist: {input_m4a_file}")
    else:
        # Delete the output files if they already exist
        if os.path.exists(output_m4a_file):
            os.remove(output_m4a_file)
            print("Existing output file deleted.")
        if os.path.exists(final_wav_file):
            os.remove(final_wav_file)
            print("Existing final WAV file deleted.")

        print("Starting audio processing...")
        final_wav_file = process_audio_pydub(input_m4a_file, output_m4a_file, 0.75)  # Adjust speed factor for slower speed
        print("Audio processing completed.")

        if final_wav_file and os.path.exists(final_wav_file):
            print("Starting transcription...")
            transcription = transcribe_audio_whisper(final_wav_file, local_model_dir=r"D:\Python\Models\HugginFaceWhisperAI-L3\whisper-large-v3")
            print("Transcription:", transcription)
            print("Transcription completed.")
        else:
            print(f"Final WAV file was not created: {final_wav_file}")
