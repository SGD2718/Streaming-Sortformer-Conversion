import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import seaborn as sns
import numpy as np
import threading
import onnx2torch
import onnxscript
from nemo.collections.asr.models import SortformerEncLabelModel
from pydub import AudioSegment
import coremltools as ct
from pydub.playback import play as play_audio

# --- 1. Setup & Config ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
audio_file = "audio.wav"

# Load Audio for Playback (pydub uses milliseconds)
print("Loading audio file for playback...")
full_audio = AudioSegment.from_wav(audio_file)

# --- 2. Load Model ---
model = SortformerEncLabelModel.from_pretrained(
    "nvidia/diar_streaming_sortformer_4spk-v2.1",
    map_location=device
)
model.eval()
model.to(device)

print(model.output_names)

def streaming_input_examples(self):
    """Input tensor examples for exporting streaming version of model"""
    batch_size = 4
    feat_in = self.cfg.get("preprocessor", {}).get("features", 128)
    chunk = torch.rand([batch_size, 120, feat_in]).to(self.device)
    chunk_lengths = torch.tensor([120] * batch_size).to(self.device)
    spkcache = torch.randn([batch_size, 188, 512]).to(self.device)
    spkcache_lengths = torch.tensor([40, 188, 0, 68]).to(self.device)
    fifo = torch.randn([batch_size, 188, 512]).to(self.device)
    fifo_lengths = torch.tensor([50, 88, 0, 90]).to(self.device)
    return chunk, chunk_lengths, spkcache, spkcache_lengths, fifo, fifo_lengths


inputs = streaming_input_examples(model)

export_out = model.export("streaming-sortformer.onnx", input_example=inputs)
scripted_model = onnx2torch.convert('streaming-sortformer.onnx')

BATCH_SIZE = 4
CHUNK_LEN = 120
FEAT_DIM = 128
CACHE_LEN = 188
EMBED_DIM = 512

ct_inputs = [
    ct.TensorType(name="chunk",          shape=(BATCH_SIZE, CHUNK_LEN, FEAT_DIM)),
    ct.TensorType(name="chunk_lens",     shape=(BATCH_SIZE,)),
    ct.TensorType(name="spkcache",       shape=(BATCH_SIZE, CACHE_LEN, EMBED_DIM)),
    ct.TensorType(name="spkcache_lens",  shape=(BATCH_SIZE,)),
    ct.TensorType(name="fifo",           shape=(BATCH_SIZE, CACHE_LEN, EMBED_DIM)),
    ct.TensorType(name="fifo_lens",      shape=(BATCH_SIZE,)),
]

ct_outputs = [
    ct.TensorType(name="preds"),
    ct.TensorType(name="new_spkcache"),
    ct.TensorType(name="new_spkcache_lens"),
    ct.TensorType(name="new_fifo"),
    ct.TensorType(name="new_fifo_lens"),
]


ct.convert(
    scripted_model,
    inputs=ct_inputs,
    outputs=ct_outputs,
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.iOS17,
    compute_precision=ct.precision.FLOAT16,
)
