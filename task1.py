import os
import torch
import torchaudio
import pyaudio
import wave
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from gtts import gTTS
import pygame
import time
import librosa

# Загрузка модели и процессора Wav2Vec 2.0
processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-russian")
model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-russian")

# Функция для записи аудио
def record_audio(filename, duration, fs=16000):
    chunk = 1024  # Размер буфера
    format = pyaudio.paInt16  # Формат записи
    channels = 1  # Количество каналов

    p = pyaudio.PyAudio()

    # Открытие потока для записи
    stream = p.open(format=format,
                    channels=channels,
                    rate=fs,
                    input=True,
                    frames_per_buffer=chunk)

    print("Говорите что-нибудь...")

    frames = []  # Список для хранения фреймов аудио

    # Запись аудио в течение заданного времени
    for i in range(0, int(fs / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    # Остановка и закрытие потока
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Сохранение записанного аудио в файл
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

# Функция для транскрипции аудио
def transcribe_audio(filename):
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Файл {filename} не найден")

    # Использование librosa для загрузки аудио
    audio, sr = librosa.load(filename, sr=None)

    # Преобразование аудио в тензор
    input_values = processor(audio, return_tensors="pt", sampling_rate=sr).input_values

    # Получение логитов модели без вычисления градиентов
    with torch.no_grad():
        logits = model(input_values).logits

    # Определение предсказанных идентификаторов
    predicted_ids = torch.argmax(logits, dim=-1)
    # Декодирование предсказанных идентификаторов в текст
    transcription = processor.batch_decode(predicted_ids)[0]

    return transcription.lower()

# Функция для генерации голосового ответа
def generate_response(text):
    tts = gTTS(text=text, lang='ru')
    filename = "response.mp3"
    tts.save(filename)

    # Инициализация и воспроизведение аудио с помощью pygame
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

    # Ожидание окончания воспроизведения
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    # Остановка и завершение микшера
    pygame.mixer.music.stop()
    pygame.mixer.quit()

    time.sleep(1)

    # Удаление файла с ответом
    try:
        os.remove(filename)
    except PermissionError:
        print(f"Не удалось удалить файл {filename}, так как он занят другим процессом.")

# Основная функция
def main():
    filename = "recorded_audio.wav"
    duration = 5  # Длительность записи в секундах
    record_audio(filename, duration)

    transcription = transcribe_audio(filename)
    print(f"Вы сказали: {transcription}")

    # Определение ответа на основе транскрипции
    if "привет я разработчик" in transcription:
        response = "сегодня выходной"
    elif "я сегодня не приду домой" in transcription:
        response = "Ну и катись отсюда"
    else:
        response = "Не распознано"

    print(f"Ответ: {response}")
    generate_response(response)

if __name__ == "__main__":
    pygame.init()
    main()
    pygame.quit()
