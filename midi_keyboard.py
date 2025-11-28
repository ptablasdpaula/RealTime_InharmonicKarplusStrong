import numpy as np
import threading
import tkinter as tk
import sounddevice as sd
from inharmonic_karplus_strong import InharmonicKarplusStrong

# Test script to check MIDI
import mido
print("Available backends:", mido.backend.module.get_api_names())
print("MIDI inputs:", mido.get_input_names())

# ============================================================
# Global audio / synth configuration
# ============================================================

FS = 48000
BLOCK_SIZE = 256

# Shared state
state_lock = threading.Lock()

current_f0 = 220.0
current_inharmonicity = 0.0001
current_damping = 0.5
current_decay = 0.98

# The synthesizer instance
synth = None

# Burst excitation
burst_len_samples = int(0.02 * FS)  # 20 ms
burst_noise = np.zeros(burst_len_samples)
burst_pos = burst_len_samples  # >= len means "no active burst"
burst_amplitude = 1.0  # Scaled by MIDI velocity

# MIDI input
midi_input = None
midi_thread = None
midi_running = False


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
# MIDI handling
# ============================================================

def midi_to_hz(midi_note):
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2 ** ((midi_note - 69) / 12.0))


def handle_midi_message(msg):
    global current_f0, burst_noise, burst_pos, burst_amplitude

    if msg.type == 'note_on' and msg.velocity > 0:
        # Calculate frequency
        f0_new = midi_to_hz(msg.note)

        # Scale velocity (0-127) to amplitude (0.0-1.0)
        velocity_scaled = msg.velocity / 127.0

        with state_lock:
            current_f0 = float(f0_new)
            burst_amplitude = velocity_scaled
            # Generate new burst excitation
            burst_noise = np.random.uniform(-0.5, 0.5, burst_len_samples).astype(np.float64)
            burst_pos = 0

        print(f"MIDI Note ON: {msg.note} | f0: {f0_new:.2f} Hz | vel: {msg.velocity} | "
              f"Dispersion sections: {synth.dispersion.active_sections}")

    elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
        # Note off - could implement muting here if desired
        # For now, just let it decay naturally
        pass


def midi_listener():
    """Background thread to listen for MIDI messages."""
    global midi_running, midi_input

    print("MIDI listener started...")
    try:
        while midi_running:
            msg = midi_input.receive()
            if msg:
                handle_midi_message(msg)
    except Exception as e:
        print(f"MIDI listener error: {e}")
    finally:
        print("MIDI listener stopped.")


def start_midi_input(port_name):
    """Start listening to MIDI input from specified port."""
    global midi_input, midi_thread, midi_running

    try:
        midi_input = mido.open_input(port_name)
        midi_running = True
        midi_thread = threading.Thread(target=midi_listener, daemon=True)
        midi_thread.start()
        print(f"✓ MIDI input opened: {port_name}")
        return True
    except Exception as e:
        print(f"✗ Failed to open MIDI port: {e}")
        return False


def stop_midi_input():
    """Stop MIDI input."""
    global midi_input, midi_running

    midi_running = False
    if midi_input:
        midi_input.close()
        midi_input = None


# ============================================================
# Audio callback
# ============================================================

def audio_callback(outdata, frames, time_info, status):
    global burst_pos, burst_noise, burst_amplitude
    global current_f0, current_inharmonicity, current_damping, current_decay
    global synth

    if status:
        print(status)

    out = np.zeros(frames, dtype=np.float32)

    with state_lock:
        for n in range(frames):
            # Get excitation (burst noise if active)
            if burst_pos < burst_len_samples:
                exc = burst_noise[burst_pos] * burst_amplitude
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
# Tkinter GUI
# ============================================================

root = tk.Tk()
root.title("Inharmonic Karplus-Strong Synth (MIDI)")
root.geometry("500x350")

# MIDI port selection
midi_frame = tk.Frame(root)
midi_frame.pack(pady=10)

tk.Label(midi_frame, text="MIDI Input Port:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)

# Get available MIDI ports
available_ports = mido.get_input_names()
if not available_ports:
    available_ports = ["No MIDI devices found"]

midi_port_var = tk.StringVar(value=available_ports[0])
midi_dropdown = tk.OptionMenu(midi_frame, midi_port_var, *available_ports)
midi_dropdown.pack(side=tk.LEFT)


def on_midi_connect():
    port_name = midi_port_var.get()
    if port_name == "No MIDI devices found":
        status_label.config(text="✗ No MIDI devices available", fg="red")
        return

    stop_midi_input()
    if start_midi_input(port_name):
        status_label.config(text=f"✓ Connected to: {port_name}", fg="green")
    else:
        status_label.config(text="✗ Failed to connect", fg="red")


connect_button = tk.Button(midi_frame, text="Connect", command=on_midi_connect)
connect_button.pack(side=tk.LEFT, padx=5)

status_label = tk.Label(root, text="Not connected", fg="gray")
status_label.pack()

info_label = tk.Label(
    root,
    text="Play notes on your MIDI keyboard.\n"
         "Adjust parameters with sliders below.",
    justify=tk.CENTER
)
info_label.pack(pady=10)


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

print("✓ Synth ready! Connect MIDI device and play.")
print(f"Available MIDI ports: {available_ports}")

# Auto-connect to first MIDI port if available
if available_ports and available_ports[0] != "No MIDI devices found":
    if start_midi_input(available_ports[0]):
        status_label.config(text=f"✓ Auto-connected to: {available_ports[0]}", fg="green")

try:
    root.mainloop()
finally:
    stop_midi_input()
    stream.stop()
    stream.close()
    print("Audio stream closed.")