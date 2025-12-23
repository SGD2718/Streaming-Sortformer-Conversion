import time

import torch
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import sounddevice as sd
import queue
import threading
import soundfile as sf
import matplotlib.animation as animation  # Explicit import fixed
from nemo.collections.asr.models import SortformerEncLabelModel

# Use TkAgg for interactive plots
matplotlib.use("TkAgg")

# --- 1. Configuration ---
SAMPLE_RATE = 16000

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- 2. Load Model ---
print(f"Loading SortFormer on {DEVICE}...")
model = SortformerEncLabelModel.from_pretrained(
    "nvidia/diar_streaming_sortformer_4spk-v2.1",
    map_location=DEVICE
)
model.eval()
model.to(DEVICE)

model.sortformer_modules.chunk_len = 125
model.sortformer_modules.chunk_right_context = 0
model.sortformer_modules.fifo_len = 188
model.sortformer_modules.spkcache_update_period = 144
model.sortformer_modules.spkcache_len = 188

# --- 3. Live State Management ---
audio_queue = queue.Queue()
plot_data = np.zeros((4, 100))  # Store last 100 frames for plotting
is_running = True


def audio_callback(indata, frames, time, status):
    if status:
        print(f"Audio Status: {status}")
    audio_queue.put(indata.copy())


def inference_loop():
    global plot_data

    # --- CONFIGURATION ---
    # We need enough history for the model to "remember" speakers.
    # 10 seconds is usually a safe minimum for stability.
    CONTEXT_DURATION = 10.0
    CONTEXT_SAMPLES = int(SAMPLE_RATE * CONTEXT_DURATION)

    # The "Step" is how much NEW audio we process at a time.
    # Matches your chunk_len=6 config (0.48s)
    NEW_DATA_DURATION = 0.333
    NEW_DATA_SAMPLES = int(SAMPLE_RATE * NEW_DATA_DURATION)

    # Initialize buffer with silence
    # This buffer will strictly hold the last 10 seconds of audio
    context_buffer = np.zeros((CONTEXT_SAMPLES, 1), dtype=np.float32)

    # Accumulator for incoming mic data
    incoming_data_accum = np.zeros((0, 1), dtype=np.float32)

    print("🎤 Listening... (Speaker labels will now remain stable)")

    while is_running:
        try:
            # 1. Get new data
            new_data = audio_queue.get(timeout=0.1)
            incoming_data_accum = np.concatenate((incoming_data_accum, new_data))

            # 2. Check if we have enough NEW data to run a step
            if len(incoming_data_accum) >= NEW_DATA_SAMPLES:

                # Take exactly one "step" of new data
                fresh_chunk = incoming_data_accum[:NEW_DATA_SAMPLES]

                # Remove it from the accumulator
                incoming_data_accum = incoming_data_accum[NEW_DATA_SAMPLES:]

                # 3. Update the Context Buffer (Shift Left, Append New)
                # Roll buffer to discard oldest data
                context_buffer = np.roll(context_buffer, -len(fresh_chunk), axis=0)
                # Overwrite end with new data
                context_buffer[-len(fresh_chunk):] = fresh_chunk

                # 4. Save Context to File
                # The model sees the full 10s context, so it knows who is speaking
                temp_path = "/tmp/live_context.wav"
                sf.write(temp_path, context_buffer, SAMPLE_RATE)

                # 5. Run Inference
                with torch.no_grad():
                    start_time = time.time()
                    _, predicted_probs = model.diarize(
                        audio=[temp_path],
                        batch_size=1,
                        include_tensor_outputs=True,
                        verbose=False
                    )
                    end_time = time.time()
                    print(f"duration: {end_time - start_time}")

                    # 6. Extract ONLY the prediction for the NEW audio
                # Output shape: [1, Time, Speakers]
                probs = predicted_probs[0].squeeze().cpu().numpy()
                if probs.ndim == 1: probs = probs.reshape(1, -1)

                # Calculate how many output frames correspond to our NEW data
                # Sortformer output is subsampled (usually 80ms stride)
                # 0.48s input / 0.08s stride = ~6 frames
                num_new_frames = int(NEW_DATA_DURATION / 0.08)

                # We only want the *end* of the prediction (the new part)
                # The beginning of the prediction is just re-confirming old history
                new_predictions = probs[-num_new_frames:, :].T

                # 7. Update Plot
                shift = new_predictions.shape[1]
                if shift > 0:
                    plot_data = np.roll(plot_data, -shift, axis=1)
                    plot_data[:, -shift:] = new_predictions

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error: {e}")


# --- 4. Setup Plotting ---
fig, ax = plt.subplots(figsize=(10, 4))


def update_plot(frame):
    # Clear and redraw is slow; ideally use set_data or blit,
    # but seaborn heatmap is complex. We'll use a fast imshow here.
    ax.clear()
    sns.heatmap(
        plot_data,
        cmap="viridis",
        cbar=False,
        vmin=0, vmax=1,
        yticklabels=["Spk 0", "Spk 1", "Spk 2", "Spk 3"],
        ax=ax
    )
    ax.set_title("Live Diarization Feed (Ring Buffer)")
    ax.set_xlabel("Time (Frames)")


# Use caching if possible, but seaborn re-renders
anim = animation.FuncAnimation(fig, update_plot, interval=100)

# --- 5. Start Execution ---
t = threading.Thread(target=inference_loop)
t.daemon = True
t.start()

try:
    with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, callback=audio_callback):
        print("Stream started.")
        plt.show()
except KeyboardInterrupt:
    is_running = False
    print("Stopping...")