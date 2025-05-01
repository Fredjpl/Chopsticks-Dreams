import pyaudio
import wave

def record_audio(filename="recorded.wav", duration=5):
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 1
    rate = 16000  # Azure expects 16000Hz

    p = pyaudio.PyAudio()
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=rate,
                    frames_per_buffer=chunk,
                    input=True)

    print("Recording...")
    frames = []

    for _ in range(int(rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print("Done recording.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))

import azure.cognitiveservices.speech as speechsdk

def transcribe_audio(filename, speech_key, region) -> str:
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=region)
    audio_config = speechsdk.audio.AudioConfig(filename=filename)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    print("Transcribing...")
    result = speech_recognizer.recognize_once()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech and result.text:
        return result.text

    # NoMatch / Canceled → return empty string
    return ""
