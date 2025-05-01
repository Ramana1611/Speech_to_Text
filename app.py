import io
import os
from flask import Flask, request, jsonify, render_template, send_file
import torch
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline
from gtts import gTTS

app = Flask(__name__)

# Load ASR Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)

# Load Text Correction Model (small GPT-2 for demo, used as LLM)
text_correction_model = pipeline(
    "text-generation",
    model="gpt2",
    device=0 if torch.cuda.is_available() else -1,
    max_length=50,
    do_sample=False,
)

def speech_to_text(audio_bytes):
    """Convert audio bytes to text using Wav2Vec2."""
    # Convert bytes to numpy array int16
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0  # Normalize to [-1,1]
    inputs = processor(audio_np, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    with torch.no_grad():
        logits = asr_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0].lower().strip()
    return transcription

def correct_text(text):
    """Use LLM to correct/complete text."""
    if not text.strip():
        return ""
    prompt = f"Correct this text:\n{text}\nCorrected:"
    result = text_correction_model(prompt, max_length=50, num_return_sequences=1)
    generated = result[0]['generated_text']
    # Extract part after "Corrected:"
    if "Corrected:" in generated:
        corrected = generated.split("Corrected:")[-1].strip()
    else:
        corrected = generated.strip()
    # Clean line breaks
    return corrected.replace("\n", " ")

def text_to_speech(text):
    """Convert corrected text to speech audio bytes (mp3)."""
    tts = gTTS(text=text, lang='en', slow=False)
    audio_bytes = io.BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/record', methods=['POST'])
def api_record():
    if 'audio_data' not in request.files:
        return jsonify({"error": "No audio file sent"}), 400
    
    audio_file = request.files['audio_data']    
    audio_bytes = audio_file.read()

    try:
        # 1. Speech to Text
        transcription = speech_to_text(audio_bytes)
        # 2. Correct Text
        corrected = correct_text(transcription)
        # 3. Text to Speech
        audio_output = text_to_speech(corrected)
        
        # Return JSON with corrected text and audio file as base64
        corrected_text = corrected
        audio_output.seek(0)
        audio_bytes_out = audio_output.read()
        import base64
        audio_base64 = base64.b64encode(audio_bytes_out).decode('utf-8')

        return jsonify({
            'corrected_text': corrected_text,
            'audio_base64': audio_base64
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Ensure templates folder exists with index.html inside
    app.run(host='0.0.0.0', port=8000, debug=True)
  
