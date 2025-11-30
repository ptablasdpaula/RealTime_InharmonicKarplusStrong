import tkinter as tk
from realtime_synth import RealtimeSynth
from inputs import KeyboardInput, MIDIInput, VirtualMIDIInput


class SynthGUI:
    def __init__(self):
        self.synth = RealtimeSynth(sample_rate=48000, block_size=256)

        self.keyboard_input = KeyboardInput(
            self.synth,
            on_note_callback=self._on_note_event
        )
        self.midi_input = MIDIInput(
            self.synth,
            on_note_callback=self._on_note_event
        )
        self.virtual_midi_input = VirtualMIDIInput(
            self.synth,
            port_name="KarplusStrong Virtual In",
            on_note_callback=self._on_note_event
        )

        self.input_mode = 'keyboard'  # 'keyboard', 'midi', or 'virtual'

        self.root = tk.Tk()
        self.root.title("Inharmonic Karplus-Strong Synthesizer")
        self.root.geometry("550x520")

        self._build_gui()

        self.root.bind('<KeyPress>', self._on_key_press)
        self.root.focus_set()

        self.synth.start()

    def _build_gui(self):
        """Build all GUI elements."""
        # ========== Input Mode Selection ==========
        input_frame = tk.LabelFrame(self.root, text="Input Mode", padx=10, pady=10)
        input_frame.pack(pady=10, padx=10, fill='x')

        self.mode_var = tk.StringVar(value='keyboard')

        tk.Radiobutton(
            input_frame,
            text="Computer Keyboard",
            variable=self.mode_var,
            value='keyboard',
            command=self._on_mode_change
        ).pack(side=tk.LEFT, padx=5)

        tk.Radiobutton(
            input_frame,
            text="MIDI Hardware",
            variable=self.mode_var,
            value='midi',
            command=self._on_mode_change
        ).pack(side=tk.LEFT, padx=5)

        tk.Radiobutton(
            input_frame,
            text="Virtual MIDI",
            variable=self.mode_var,
            value='virtual',
            command=self._on_mode_change
        ).pack(side=tk.LEFT, padx=5)

        # ========== MIDI Connection (only visible in MIDI mode) ==========
        self.midi_frame = tk.Frame(self.root)

        tk.Label(
            self.midi_frame,
            text="MIDI Port:",
            font=("Arial", 9)
        ).pack(side=tk.LEFT, padx=5)

        # MIDI port dropdown
        available_ports = self.midi_input.get_available_ports()
        if not available_ports:
            available_ports = ["No MIDI devices found"]

        self.midi_port_var = tk.StringVar(value=available_ports[0])
        self.midi_dropdown = tk.OptionMenu(
            self.midi_frame,
            self.midi_port_var,
            *available_ports
        )
        self.midi_dropdown.pack(side=tk.LEFT, padx=5)

        self.midi_connect_btn = tk.Button(
            self.midi_frame,
            text="Connect",
            command=self._on_midi_connect
        )
        self.midi_connect_btn.pack(side=tk.LEFT, padx=5)

        self.midi_status_label = tk.Label(
            self.midi_frame,
            text="Not connected",
            fg="gray"
        )
        self.midi_status_label.pack(side=tk.LEFT, padx=5)

        # ========== Virtual MIDI Status (only visible in virtual mode) ==========
        self.virtual_midi_frame = tk.Frame(self.root)

        self.virtual_status_label = tk.Label(
            self.virtual_midi_frame,
            text="Virtual port not open",
            fg="gray",
            font=("Arial", 9)
        )
        self.virtual_status_label.pack(pady=5)

        # ========== Info Display ==========
        info_frame = tk.LabelFrame(self.root, text="Controls", padx=10, pady=10)
        info_frame.pack(pady=10, padx=10, fill='x')

        self.info_text = tk.Label(
            info_frame,
            text=self.keyboard_input.get_help_text(),
            justify=tk.LEFT,
            font=("Courier", 9)
        )
        self.info_text.pack()

        # Octave display (keyboard mode only)
        self.octave_frame = tk.Frame(info_frame)
        self.octave_frame.pack(pady=5)

        tk.Label(
            self.octave_frame,
            text="Octave:",
            font=("Arial", 10)
        ).pack(side=tk.LEFT, padx=5)

        self.octave_label = tk.Label(
            self.octave_frame,
            text=str(self.keyboard_input.get_octave()),
            font=("Arial", 12, "bold")
        )
        self.octave_label.pack(side=tk.LEFT)

        # ========== Parameter Controls ==========
        param_frame = tk.LabelFrame(self.root, text="Synthesis Parameters", padx=10, pady=10)
        param_frame.pack(pady=10, padx=10, fill='both', expand=True)

        # Inharmonicity
        tk.Label(param_frame, text="Inharmonicity (B)").pack(anchor='w')
        self.inharmonicity_slider = tk.Scale(
            param_frame,
            from_=0.0, to=0.025,
            resolution=0.00001,
            orient='horizontal',
            length=450,
            command=lambda v: self.synth.set_inharmonicity(float(v))
        )
        self.inharmonicity_slider.set(self.synth.current_inharmonicity)
        self.inharmonicity_slider.pack(fill='x', pady=2)

        # Damping
        tk.Label(param_frame, text="Damping (IIR a1, higher = more damping)").pack(anchor='w')
        self.damping_slider = tk.Scale(
            param_frame,
            from_=0.0, to=0.9,
            resolution=0.01,
            orient='horizontal',
            length=450,
            command=lambda v: self.synth.set_damping(float(v))
        )
        self.damping_slider.set(self.synth.current_damping)
        self.damping_slider.pack(fill='x', pady=2)

        # Decay
        tk.Label(param_frame, text="Decay (Feedback gain)").pack(anchor='w')
        self.decay_slider = tk.Scale(
            param_frame,
            from_=0.9, to=1.0,
            resolution=0.001,
            orient='horizontal',
            length=450,
            command=lambda v: self.synth.set_decay(float(v))
        )
        self.decay_slider.set(self.synth.current_decay)
        self.decay_slider.pack(fill='x', pady=2)

        # Update UI for initial mode
        self._update_mode_ui()

    def _on_mode_change(self):
        """Handle input mode change."""
        new_mode = self.mode_var.get()

        if new_mode == self.input_mode:
            return

        # Disconnect previous mode
        if self.input_mode == 'midi':
            self.midi_input.disconnect()
        elif self.input_mode == 'virtual':
            self.virtual_midi_input.close_port()

        self.input_mode = new_mode
        self._update_mode_ui()

    def _update_mode_ui(self):
        """Update UI elements based on current mode."""
        if self.input_mode == 'keyboard':
            # Show keyboard info
            self.midi_frame.pack_forget()
            self.virtual_midi_frame.pack_forget()
            self.octave_frame.pack(pady=5)
            self.info_text.config(text=self.keyboard_input.get_help_text())

        elif self.input_mode == 'midi':
            # Show MIDI info
            self.octave_frame.pack_forget()
            self.virtual_midi_frame.pack_forget()
            self.midi_frame.pack(pady=5, padx=10, fill='x')
            self.info_text.config(text=self.midi_input.get_help_text())

            # Auto-connect if available
            available = self.midi_input.get_available_ports()
            if available and available[0] != "No MIDI devices found":
                self._on_midi_connect()

        else:  # virtual
            # Show virtual MIDI info
            self.octave_frame.pack_forget()
            self.midi_frame.pack_forget()
            self.virtual_midi_frame.pack(pady=5, padx=10, fill='x')
            self.info_text.config(text=self.virtual_midi_input.get_help_text())

            # Auto-open virtual port
            if self.virtual_midi_input.open_port():
                port_name = self.virtual_midi_input.get_port_name()
                self.virtual_status_label.config(
                    text=f"✓ Virtual port open: '{port_name}'",
                    fg="green"
                )
            else:
                self.virtual_status_label.config(
                    text="✗ Failed to open virtual port",
                    fg="red"
                )

    def _on_midi_connect(self):
        """Handle MIDI connect button."""
        port_name = self.midi_port_var.get()

        if port_name == "No MIDI devices found":
            self.midi_status_label.config(text="✗ No devices", fg="red")
            return

        if self.midi_input.connect(port_name):
            self.midi_status_label.config(
                text=f"✓ Connected",
                fg="green"
            )
        else:
            self.midi_status_label.config(text="✗ Failed", fg="red")

    def _on_key_press(self, event):
        """Handle keyboard key press."""
        if self.input_mode != 'keyboard':
            return

        result = self.keyboard_input.handle_key_press(event.char)

        if result and result['action'] == 'octave_change':
            self.octave_label.config(text=str(result['octave']))

    def _on_note_event(self, midi_note, velocity):
        """Callback for note events (for potential future use)."""
        # Could update a note display here if desired
        pass

    def run(self):
        """Start the GUI main loop."""
        print("✓ Synthesizer GUI ready!")
        print(f"  Sample rate: {self.synth.fs} Hz")
        print(f"  Dispersion sections: {self.synth.synth.dispersion.active_sections}")

        try:
            self.root.mainloop()
        finally:
            self.midi_input.disconnect()
            self.virtual_midi_input.close_port()
            self.synth.stop()


if __name__ == "__main__":
    gui = SynthGUI()
    gui.run()