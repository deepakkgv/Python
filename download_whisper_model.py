from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

model_id = "openai/whisper-large-v3"
local_dir = r"D:\Python\Models\HugginFaceWhisperAI-L3\whisper-large-v3"

# Download the model and save it locally
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
model.save_pretrained(local_dir)

# Download the processor and save it locally
processor = AutoProcessor.from_pretrained(model_id)
processor.save_pretrained(local_dir)

print(f"Model and processor saved to {local_dir}")
