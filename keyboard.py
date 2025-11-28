import numpy as np
import threading
import tkinter as tk
import sounddevice as sd
from inharmonic_karplus_strong import InharmonicKarplusStrong

# ============================================================
# Global audio / synth configuration
# ===============================================hhh=============

FS = 48000
BLOCK_SIZE = 256

# Shared state
state_lock = threading.Lock()

current_f0 = 220.0
current_inharmonicity = 0.0001
current_damping = 0.5
current_decay = 0.98
params_dirty = True

# The synthesizer instance
synth = None

# Burst excitation
burst_len_samples = int(0.02 * FS)  # 20 ms
burst_noise = np.zeros(burst_len_samples)
burst_pos = burst_len_samples  # >= len means "no active burst"


# ============================================================
# Initialize synthesizer
# ============================================================

def init_synth():
    global synth
    synth = InharmonicKarplusStrong(sr=FS, max_dispersion_sections=12, lagrange_order=5)
    # Warm up with silence
    for _ in range(100):
        synth.process_sample(0.0, current_f0, current_decay, current_damping, current_inharmonicity)


# ============================================================
# Audio callback
# ============================================================

def audio_callback(outdata, frames, time_info, status):
    global burst_pos, burst_noise
    global current_f0, current_inharmonicity, current_damping, current_decay
    global synth

    if status:
        print(status)

    out = np.zeros(frames, dtype=np.float32)

    with state_lock:
        for n in range(frames):
            # Get excitation (burst noise if active)
            if burst_pos < burst_len_samples:
                exc = burst_noise[burst_pos]
                burst_pos += 1
            else:
                exc = 0.0

            # Process through synth
            out[n] = synth.process_sample(
                excitation=exc,
                f0=current_f0,
                decay=current_decay,
                damping=current_damping,
                inharmonicity=current_inharmonicity
            )

    outdata[:, 0] = out


# ============================================================
# Tkinter GUI and keyboard handling
# ============================================================

root = tk.Tk()
root.title("Inharmonic Karplus-Strong Synth")
root.geometry("500x280")

info_label = tk.Label(
    root,
    text="Play notes with keys A W S E D F T G Y H U J K (chromatic C to C).\n"
         "Use Z/X to shift octave down/up.",
    justify=tk.LEFT
)
info_label.pack(pady=5)

current_octave = 4
octave_label = tk.Label(root, text=f"Octave: {current_octave}", font=("Arial", 12, "bold"))
octave_label.pack()

# Note mapping: chromatic scale C to C
note_map = {
    'a': 0,   # C
    'w': 1,   # C#
    's': 2,   # D
    'e': 3,   # D#
    'd': 4,   # E
    'f': 5,   # F
    't': 6,   # F#
    'g': 7,   # G
    'y': 8,   # G#
    'h': 9,   # A
    'u': 10,  # A#
    'j': 11,  # B
    'k': 12   # C (next octave)
}


# Slider callbacks
def on_inharmonicity_change(value):
    global current_inharmonicity
    with state_lock:
        current_inharmonicity = float(value)


def on_damping_change(value):
    global current_damping
    with state_lock:
        current_damping = float(value)


def on_decay_change(value):
    global current_decay
    with state_lock:
        current_decay = float(value)


# Sliders
inharmonicity_slider = tk.Scale(
    root, from_=0.0, to=0.025, resolution=0.00001, orient='horizontal',
    length=400, label="Inharmonicity (B)", command=on_inharmonicity_change
)
inharmonicity_slider.set(current_inharmonicity)
inharmonicity_slider.pack(pady=5)

damping_slider = tk.Scale(
    root, from_=0.0, to=0.9, resolution=0.01, orient='horizontal',
    length=400, label="Damping (IIR a1, higher = more damping)", command=on_damping_change
)
damping_slider.set(current_damping)
damping_slider.pack(pady=5)

decay_slider = tk.Scale(
    root, from_=0.9, to=1.0, resolution=0.001, orient='horizontal',
    length=400, label="Decay (Feedback gain)", command=on_decay_change
)
decay_slider.set(current_decay)
decay_slider.pack(pady=5)


# Keyboard handler
def on_key_press(event):
    global current_octave, current_f0
    global burst_noise, burst_pos

    ch = event.char.lower()

    if ch in note_map:
        # Calculate MIDI note (C4 = 60)
        semitone_offset = note_map[ch]
        midi_note = 12 + 12 * current_octave + semitone_offset
        f0_new = 440.0 * (2 ** ((midi_note - 69) / 12.0))

        with state_lock:
            current_f0 = float(f0_new)
            # Generate new burst excitation
            burst_noise = np.random.uniform(-0.5, 0.5, burst_len_samples).astype(np.float64)
            burst_pos = 0

        print(f"Note: {ch.upper()} | f0: {f0_new:.2f} Hz | MIDI: {midi_note} | "
              f"Dispersion sections: {synth.dispersion.active_sections}")

    elif ch == 'z':
        if current_octave > 1:
            current_octave -= 1
            octave_label.config(text=f"Octave: {current_octave}")
    elif ch == 'x':
        if current_octave < 6:
            current_octave += 1
            octave_label.config(text=f"Octave: {current_octave}")


root.bind('<KeyPress>', on_key_press)
root.focus_set()


# ============================================================
# Start audio stream and GUI mainloop
# ============================================================

# Initialize synth
with state_lock:
    init_synth()

print("Starting audio stream...")
print("Active dispersion sections:", synth.dispersion.active_sections)

stream = sd.OutputStream(
    samplerate=FS,
    blocksize=BLOCK_SIZE,
    channels=1,
    dtype='float32',
    callback=audio_callback
)
stream.start()

print("✓ Synth ready! Play notes with keyboard.")
print("  Keys: A W S E D F T G Y H U J K (chromatic scale)")
print("  Octave: Z (down) / X (up)")

try:
    root.mainloop()
finally:
    stream.stop()
    stream.close()
    print("Audio stream closed.")