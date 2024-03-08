from transformers import Wav2Vec2ForCTC, AutoProcessor
from langid.langid import LanguageIdentifier, model
import speech_recognition
import torchaudio
import torch
import os


def wav_record():
    r = speech_recognition.Recognizer()

    while True:
        print("Recording... (Press Enter to stop)")
        with speech_recognition.Microphone() as source:
            audio = r.listen(source)

            if (input() == ""): 
                print("Finish Recording! Saving...")
                file_name = f"audios/recorded_audio.wav"

                with open(file_name, "wb") as f:
                    f.write(audio.get_wav_data())
                break  

    return os.getcwd() + f"/{file_name}"


def speech2text(wave_file_path):  
    model_id = "facebook/mms-1b-all"
    processor = AutoProcessor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)

    audio_data, original_sampling_rate = torchaudio.load((wave_file_path))
    resampled_audio_data = torchaudio.transforms.Resample(original_sampling_rate, 16000)(audio_data)
    inputs = processor(resampled_audio_data.numpy(), sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs).logits

    ids = torch.argmax(outputs, dim=-1)[0]
    transcription = processor.decode(ids)

    print(f"\nAudio Transcription: {transcription}")
    return transcription


def detect_language(wav_file_path):
    text = speech2text(wave_file_path=wav_file_path)
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    return identifier.classify(text)[0]


# === MAIN ===
audio_file_path = wav_record()
detected_language = detect_language(audio_file_path)
print("Detected language:", detected_language)
