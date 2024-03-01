from faster_whisper import WhisperModel

import logging

# logging.basicConfig()
# logging.getLogger("faster_whisper").setLevel(logging.DEBUG)

# model_size = "large-v3"
model_size = "/opt/scratch/lesheng/models/models--Systran--faster-whisper-large-v3/snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478/"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

audio_file = "../librispeech_dummy.wav"

segments, info = model.transcribe(audio_file, beam_size=1)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))