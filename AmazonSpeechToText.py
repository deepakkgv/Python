import os
import boto3
import time
import urllib.request
import json
from pydub import AudioSegment
from datetime import datetime

# Set up AWS credentials and region from the downloaded CSV file
AWS_ACCESS_KEY_ID = 'AKIAQLSIVW2R67QMXUHV'
AWS_SECRET_ACCESS_KEY = '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'
AWS_REGION = 'ap-southeast-2'

# Initialize the Transcribe client
transcribe = boto3.client('transcribe', 
                          aws_access_key_id=AWS_ACCESS_KEY_ID,
                          aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                          region_name=AWS_REGION)

def upload_to_s3(file_path, bucket_name, object_name):
    s3 = boto3.client('s3', 
                      aws_access_key_id=AWS_ACCESS_KEY_ID,
                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                      region_name=AWS_REGION)
    try:
        s3.upload_file(file_path, bucket_name, object_name)
        print(f"File {file_path} uploaded to {bucket_name}/{object_name}")
    except Exception as e:
        print(f"Error uploading file: {e}")

def transcribe_audio(file_path, bucket_name, object_name, job_name, language_code='en-US'):
    upload_to_s3(file_path, bucket_name, object_name)
    
    job_uri = f's3://{bucket_name}/{object_name}'
    
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': job_uri},
        MediaFormat='wav',
        LanguageCode=language_code
    )
    
    while True:
        status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
            break
        print("Waiting for transcription to complete...")
        time.sleep(10)
    
    if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
        transcript_url = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
        response = urllib.request.urlopen(transcript_url)
        data = response.read().decode('utf-8')
        transcript_json = json.loads(data)
        
        # Extract and print only the "transcripts" values
        transcripts = transcript_json['results']['transcripts']
        for transcript in transcripts:
            print(transcript['transcript'])
    else:
        print("Transcription failed.")

def preprocess_audio(file_path, output_format='wav'):
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    output_file = os.path.splitext(file_path)[0] + f'_processed.{output_format}'
    audio.export(output_file, format=output_format)
    return output_file

if __name__ == "__main__":
    file_path = r"D:\Audio Files\Pj_PROD.m4a"
    bucket_name = 'vcaaudio'
    object_name = 'audio/Pj_PROD_processed.wav'
    job_name = f'transcription_job_{datetime.now().strftime("%Y%m%d%H%M%S")}'
    
    processed_file_path = preprocess_audio(file_path)
    transcribe_audio(processed_file_path, bucket_name, object_name, job_name)
