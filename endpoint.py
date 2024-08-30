import os
import torch
import torchaudio
from tempfile import NamedTemporaryFile
from fastapi import FastAPI, UploadFile, File
from langid.langid import LanguageIdentifier, model
from transformers import Wav2Vec2ForCTC, AutoProcessor

app = FastAPI()

model_id = "facebook/mms-1b-all"
processor = AutoProcessor.from_pretrained(model_id)
stt_model = Wav2Vec2ForCTC.from_pretrained(model_id)

@app.get("/")
async def root():
    return {"message": "Server running!"}

@app.post("/predict-voice-language")
async def predict(file: UploadFile = File(...)):
    with NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name

    audio_data, original_sampling_rate = torchaudio.load(temp_file_path, format="wav")
    os.remove(temp_file_path)

    resampled_audio_data = torchaudio.transforms.Resample(original_sampling_rate, 16000)(audio_data)
    inputs = processor(resampled_audio_data.numpy(), sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        outputs = stt_model(**inputs).logits
    ids = torch.argmax(outputs, dim=-1)[0]

    transcription = processor.decode(ids)
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    predicted_language = identifier.classify(transcription)[0]

    return {"language": predicted_language, "transcription": transcription}
