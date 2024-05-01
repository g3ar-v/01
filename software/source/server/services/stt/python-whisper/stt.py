import os
import numpy as np
import whisper
from speech_recognition import AudioData


class Stt:
    MODELS = (
        "tiny.en",
        "tiny",
        "base.en",
        "base",
        "small.en",
        "small",
        "medium.en",
        "medium",
        "large",
        "large-v2",
    )

    def __init__(self, config):
        self.service_directory = config["service_directory"]
        model = os.getenv("PY_WHISPER_MODEL", "base,en")

        if not model:
            model = "base.en"
        assert model in self.MODELS  # TODO - better error handling

        self.engine = whisper.load_model(model)

    @staticmethod
    def audiodata2array(audio_data):
        # Convert buffer to float32 using NumPy
        audio_as_np_int16 = np.frombuffer(audio_data, dtype=np.int16)
        audio_as_np_float32 = audio_as_np_int16.astype(np.float32)

        # Normalise float32 array so that values are between -1.0 and +1.0
        max_int16 = 2**15
        data = audio_as_np_float32 / max_int16
        return data

    def stt(self, audio):
        if isinstance(audio, AudioData):
            result = self.engine.transcribe(
                self.audiodata2array(audio.get_raw_data()),
            )
        result = self.engine.transcribe(audio)
        text = result["text"].strip()
        return text
