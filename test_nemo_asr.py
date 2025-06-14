import sounddevice as sd
import numpy as np
from nemo.collections.asr.models import EncDecCTCModel

print("🔊 Loading NeMo ASR model...")
model = EncDecCTCModel.from_pretrained("nvidia/stt_en_conformer_ctc_large")
model.eval()

samplerate = 16000
duration = 3  # seconds

print("🎤 Speak for 3 seconds after the beep...")
beep = np.sin(2 * np.pi * 440 * np.linspace(0, 0.2, int(samplerate * 0.2)))
sd.play(beep, samplerate)
sd.wait()

recording = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='float32')
sd.wait()

print("🧠 Transcribing...")
transcript = model.transcribe([recording.squeeze()])[0]
print(f"\n📝 Transcript: {transcript}")
