import numpy as np
import threading
import sounddevice as sd
from karplus_strong import InharmonicKarplusStrong
from utils import midi_to_hz


class RealtimeSynth:
    def __init__(self, sample_rate=48000, block_size=256):
        self.fs = sample_rate
        self.block_size = block_size

        self.state_lock = threading.Lock()

        # Synth parameters
        self.current_f0 = 220.0
        self.current_inharmonicity = 0.0001
        self.current_damping = 0.5
        self.current_decay = 0.98

        # Burst excitation state
        self.burst_len_samples = int(0.02 * self.fs)  # 20 ms
        self.burst_noise = np.zeros(self.burst_len_samples)
        self.burst_pos = self.burst_len_samples  # >= len means "no active burst"
        self.burst_amplitude = 1.0

        # Initialize synthesizer
        self.synth = InharmonicKarplusStrong(
            sr=self.fs,
            max_dispersion_sections=12,
            lagrange_order=5
        )

        # Warm up with silence
        for _ in range(100):
            self.synth.process_sample(
                0.0,
                self.current_f0,
                self.current_decay,
                self.current_damping,
                self.current_inharmonicity
            )

        # Audio stream
        self.stream = None

    def _audio_callback(self, outdata, frames, time_info, status):
        """Audio callback for sounddevice stream."""
        if status:
            print(status)

        out = np.zeros(frames, dtype=np.float32)

        with self.state_lock:
            for n in range(frames):
                # Get excitation (burst noise if active)
                if self.burst_pos < self.burst_len_samples:
                    exc = self.burst_noise[self.burst_pos] * self.burst_amplitude
                    self.burst_pos += 1
                else:
                    exc = 0.0

                # Process through synth
                out[n] = self.synth.process_sample(
                    excitation=exc,
                    f0=self.current_f0,
                    decay=self.current_decay,
                    damping=self.current_damping,
                    inharmonicity=self.current_inharmonicity
                )

        outdata[:, 0] = out

    def start(self):
        """Start the audio stream."""
        if self.stream is not None:
            print("Audio stream already running")
            return

        self.stream = sd.OutputStream(
            samplerate=self.fs,
            blocksize=self.block_size,
            channels=1,
            dtype='float32',
            callback=self._audio_callback
        )
        self.stream.start()
        print(f"✓ Audio stream started at {self.fs} Hz")
        print(f"  Active dispersion sections: {self.synth.dispersion.active_sections}")

    def stop(self):
        """Stop the audio stream."""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            print("Audio stream closed.")

    def note_on(self, midi_note, velocity=127):
        f0_new = midi_to_hz(midi_note)
        velocity_scaled = velocity / 127.0

        with self.state_lock:
            self.current_f0 = float(f0_new)
            self.burst_amplitude = velocity_scaled
            # Generate new burst excitation
            self.burst_noise = np.random.uniform(
                -0.5, 0.5, self.burst_len_samples
            ).astype(np.float64)
            self.burst_pos = 0

        print(f"Note ON: MIDI {midi_note} | f0: {f0_new:.2f} Hz | vel: {velocity} | "
              f"Dispersion: {self.synth.dispersion.active_sections}")

    def note_off(self, midi_note):
        pass

    def set_inharmonicity(self, value):
        with self.state_lock:
            self.current_inharmonicity = float(value)

    def set_damping(self, value):
        with self.state_lock:
            self.current_damping = float(value)

    def set_decay(self, value):
        with self.state_lock:
            self.current_decay = float(value)

    def get_parameters(self):
        with self.state_lock:
            return {
                'f0': self.current_f0,
                'inharmonicity': self.current_inharmonicity,
                'damping': self.current_damping,
                'decay': self.current_decay
            }