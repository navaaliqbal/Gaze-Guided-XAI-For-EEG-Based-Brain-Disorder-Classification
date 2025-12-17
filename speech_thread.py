# speech_thread.py
import time
import json
import pyaudio
from vosk import Model, KaldiRecognizer
from PyQt5.QtCore import QThread, pyqtSignal

class SpeechRecognitionThread(QThread):
    # Emits a dict for each recognized word with absolute start/end times
    newSpeechWord = pyqtSignal(dict)

    def __init__(self, model_path="D:/whispertry/vosk-model-en-in-0.5/vosk-model-en-in-0.5", device_index=None, rate=16000, frames_per_buffer=4096):
        super().__init__()
        self.model_path = model_path
        self.device_index = device_index
        self.rate = rate
        self.frames_per_buffer = frames_per_buffer
        self._running = False

        # Create model & recognizer objects here (so errors show early)
        self.model = Model(self.model_path)
        self.rec = KaldiRecognizer(self.model, self.rate)
        self.rec.SetWords(True)

    def run(self):
        self._running = True
        self.audio_start_time = time.time()   # absolute reference for word offsets

        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16,
                         channels=1,
                         rate=self.rate,
                         input=True,
                         input_device_index=self.device_index,
                         frames_per_buffer=self.frames_per_buffer)
        stream.start_stream()

        try:
            while self._running:
                data = stream.read(self.frames_per_buffer, exception_on_overflow=False)
                if self.rec.AcceptWaveform(data):
                    res = json.loads(self.rec.Result())
                    if "result" in res:
                        # res["result"] is a list of words with start/end relative to audio_start
                        for w in res["result"]:
                            # build absolute timestamps (wall-clock seconds)
                            abs_start = self.audio_start_time + float(w["start"])
                            abs_end   = self.audio_start_time + float(w["end"])
                            payload = {
                                "word": w.get("word"),
                                "start": abs_start,
                                "end": abs_end,
                                "conf": w.get("conf", None)
                            }
                            self.newSpeechWord.emit(payload)
                # else: partial results are in rec.PartialResult() if you want interim display
        finally:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
            pa.terminate()

    def stop(self):
        # Request stop; run() will exit after next buffer read
        self._running = False
        # If the thread is still running, quit/wait will be used by caller
