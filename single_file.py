import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import seaborn as sns
import numpy as np
import threading
import os
from nemo.collections.asr.models import SortformerEncLabelModel
from pydub import AudioSegment
from pydub.playback import play as play_audio

# Use TkAgg for interactive plots on Mac
matplotlib.use("TkAgg")

# --- 1. Setup & Config ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
script_dir = os.path.dirname(os.path.abspath(__file__))
audio_path = 'audio.wav'
audio_file = os.path.join(script_dir, audio_path)
config_file = os.path.join(script_dir, "config.yaml")

# Load Audio for Playback (pydub uses milliseconds)
print("Loading audio file for playback...")
full_audio = AudioSegment.from_wav(audio_file)

# --- 2. Load Model ---
# Load the expanded 8-speaker model
model_path = os.path.join(script_dir, "streaming-sortformer-8spk-smart.nemo")
model = SortformerEncLabelModel.from_pretrained("nvidia/diar_streaming_sortformer_4spk-v2.1", map_location=device)
model.eval()
model.to(device)

print(model.sortformer_modules)


def format_dict(d, indent_level: int = 0) -> str:
    res = '\t' * max(0, indent_level - 1) + '{'
    for k, v in d.items():
        if isinstance(v, dict):
            res += '\t' * indent_level + f"{k}: {format_dict(v, indent_level + 1)},\n"
        else:
            res += '\t' * indent_level + f'{k}: {v},\n'
    res += '\t' * max(0, indent_level - 1) + '}'
    return res


# 3. Configure for ~0.32s Latency
model.sortformer_modules.chunk_len = 6  # Adjusted per your snippet
model.sortformer_modules.chunk_right_context = 1
model.sortformer_modules.chunk_left_context = 1

model.sortformer_modules.fifo_len = 40
model.sortformer_modules.spkcache_update_period = 42
model.sortformer_modules.spkcache_len = 120


print(model.sortformer_modules)
print(format_dict(model._cfg))

# --- 4. Run Inference ---
try:
    print("Running inference...")
    predicted_segments, predicted_probs = model.diarize(
        audio=audio_file,
        batch_size=1,
        include_tensor_outputs=True
    )

    # --- 5. Process Output ---
    probs = predicted_probs[0].squeeze().cpu().numpy()
    heatmap_data = probs.T

    # --- 6. Interaction Logic ---
    def play_segment_thread(start_s, duration_s):
        """Plays audio in a separate thread so UI doesn't freeze"""
        start_ms = int(start_s * 1000)
        end_ms = int((start_s + duration_s) * 1000)

        print(f"▶️ Playing: {start_s:.2f}s -> {start_s + duration_s:.2f}s")
        segment = full_audio[start_ms:end_ms]
        play_audio(segment)


    def on_click(event):
        """Handle click events on the plot"""
        if event.inaxes is None:
            return

        # Check if click is inside any of our segment rectangles
        for patch in ax.patches:
            contains, _ = patch.contains(event)
            if contains:
                # Retrieve the metadata we attached to the patch
                meta = getattr(patch, 'audio_meta', None)
                if meta:
                    # Run playback in a thread
                    t = threading.Thread(target=play_segment_thread, args=(meta['start'], meta['dur']))
                    t.start()
                    return


    # --- 7. Plotting ---
    fig, ax = plt.subplots(figsize=(15, 6))

    # Connect the click event
    fig.canvas.mpl_connect('button_press_event', on_click)

    sns.heatmap(
        heatmap_data,
        cmap="viridis",
        yticklabels=[f"Spk {i}" for i in range(heatmap_data.shape[0])],
        cbar_kws={'label': 'Activity Probability'},
        vmin=0, vmax=1,
        ax=ax
    )

    # --- 8. Parse & Outline Segments ---
    # RTTM is a string, split by lines
    rttm_content = predicted_segments[0]
    frame_shift = 0.08

    print("Parsing Segments & Creating Clickable Areas:")

    # Fix: Iterate over split lines, not characters
    for line in rttm_content:
        parts = line.split()

        # Robust check for standard RTTM format
        # Format: SPEAKER <file> <ch> <start> <dur> <ortho> <stype> <name> <conf>

        start_time = float(parts[0])
        end_time = float(parts[1])
        duration = end_time - start_time
        speaker_label = parts[2]

        # Extract ID (e.g., "speaker_3" -> 3)
        spk_idx = int(speaker_label.split('_')[-1])

        # Convert Time to Frames
        start_frame = start_time / frame_shift
        width_frame = duration / frame_shift

        print(f"  {speaker_label}: {start_time:.2f}s -> {end_time:.2f}s")

        rect = patches.Rectangle(
            (start_frame, spk_idx),
            width=width_frame,
            height=1,
            linewidth=2,
            edgecolor='#FF00FF',
            facecolor='none',
            picker=True  # Enable picking for mouse events
        )

        # Attach audio metadata directly to the rectangle object
        rect.audio_meta = {'start': start_time, 'dur': duration}

        ax.add_patch(rect)

    plt.title("Streaming Diarization (Click a Pink Box to Play Audio)")
    plt.xlabel("Time Frames (80ms steps)")
    plt.ylabel("Speaker ID")
    plt.tight_layout()

    print("Inference complete. Interactive window opening...")
    plt.show()

except Exception as e:
    import traceback

    traceback.print_exc()
