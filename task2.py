import os
import torch
import librosa
import numpy as np
from sklearn.cluster import KMeans
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Загрузка модели и процессора Wav2Vec2 для распознавания речи
processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-russian")
model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-russian")

# Функция для извлечения признаков из аудиофрагмента
def extract_features(audio, sr):
    # Вычисление MFCC (Мел-частотных кепстральных коэффициентов)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    # Усреднение MFCC по временной оси
    return np.mean(mfcc.T, axis=0)

# Функция для транскрипции аудиофрагмента
def transcribe_audio(audio, sr):
    # Преобразование аудиоданных в тензор
    input_values = processor(audio, return_tensors="pt", sampling_rate=sr).input_values

    # Получение логитов модели без вычисления градиентов
    with torch.no_grad():
        logits = model(input_values).logits

    # Определение предсказанных идентификаторов
    predicted_ids = torch.argmax(logits, dim=-1)
    # Декодирование предсказанных идентификаторов в текст
    transcription = processor.batch_decode(predicted_ids)[0]

    # Возвращение транскрипции в нижнем регистре
    return transcription.lower()

# Путь к аудиофайлу
filename = "C:/Users/User/Desktop/audio/Phone_ARU_ON.wav"

# Загрузка аудиофайла с помощью librosa
audio, sr = librosa.load(filename, sr=16000)

# Разделение аудио на фрагменты
window_size = 5  # окно в секундах
step_size = 2    # шаг в секундах
segments = []  # список для хранения фрагментов аудио
features = []  # список для хранения признаков фрагментов

# Разделение аудио на фрагменты заданного размера с заданным шагом
for start in range(0, len(audio), int(step_size * sr)):
    end = start + int(window_size * sr)
    if end > len(audio):
        break
    segment = audio[start:end]
    segments.append((start, end, segment))
    feature = extract_features(segment, sr)
    features.append(feature)

# Кластеризация фрагментов
num_speakers = 2  # предполагаемое количество спикеров, может быть изменено
kmeans = KMeans(n_clusters=num_speakers, random_state=0).fit(features)
labels = kmeans.labels_

# Транскрипция и вывод результатов
speaker_transcriptions = {i: [] for i in range(num_speakers)}

# Транскрипция каждого фрагмента и распределение по спикерам
for label, (start, end, segment) in zip(labels, segments):
    transcription = transcribe_audio(segment, sr)
    speaker_transcriptions[label].append(transcription)

# Вывод результатов
for speaker, transcriptions in speaker_transcriptions.items():
    print(f"Спикер {speaker}:")
    for transcription in transcriptions:
        print(f"  {transcription}")

