import threading
import mido

class KeyboardInput:
    """
    Handles computer keyboard input and converts to MIDI note events.
    """
    def __init__(self, synth, on_note_callback=None):
        """
        Args:
            synth: RealtimeSynth instance
            on_note_callback: Optional callback for note events (for display updates)
        """
        self.synth = synth
        self.on_note_callback = on_note_callback
        self.current_octave = 4

        # Maps keyboard keys to semitone offsets from C
        self.note_map = {
            'a': 0,  # C
            'w': 1,  # C#
            's': 2,  # D
            'e': 3,  # D#
            'd': 4,  # E
            'f': 5,  # F
            't': 6,  # F#
            'g': 7,  # G
            'y': 8,  # G#
            'h': 9,  # A
            'u': 10,  # A#
            'j': 11,  # B
            'k': 12  # C (next octave)
        }

    def handle_key_press(self, char):
        """
        Handle a keyboard character press.

        Args:
            char: Character from key press event

        Returns:
            dict with action info, or None if no action taken
        """
        ch = char.lower()

        if ch in self.note_map:
            # Calculate MIDI note (C4 = 60)
            semitone_offset = self.note_map[ch]
            midi_note = 12 + 12 * self.current_octave + semitone_offset

            # Send note to synth (fixed velocity for keyboard)
            self.synth.note_on(midi_note, velocity=100)

            if self.on_note_callback:
                self.on_note_callback(midi_note, 100)

            return {
                'action': 'note',
                'midi_note': midi_note,
                'key': ch.upper()
            }

        elif ch == 'z':
            # Octave down
            if self.current_octave > 1:
                self.current_octave -= 1
                return {
                    'action': 'octave_change',
                    'octave': self.current_octave
                }

        elif ch == 'x':
            # Octave up
            if self.current_octave < 6:
                self.current_octave += 1
                return {
                    'action': 'octave_change',
                    'octave': self.current_octave
                }

        return None

    def get_octave(self):
        return self.current_octave

    def get_help_text(self):
        """Get help text for keyboard controls."""
        return (
            "Keyboard Controls:\n"
            "  A W S E D F T G Y H U J K - Play notes (chromatic C to C)\n"
            "  Z - Octave down\n"
            "  X - Octave up"
        )


class MIDIInput:
    """
    Handles MIDI hardware input and converts to synth events.
    """

    def __init__(self, synth, on_note_callback=None):
        """
        Args:
            synth: RealtimeSynth instance
            on_note_callback: Optional callback for note events (for display updates)
        """
        self.synth = synth
        self.on_note_callback = on_note_callback

        # MIDI state
        self.midi_input = None
        self.midi_thread = None
        self.midi_running = False
        self.current_port = None

    def get_available_ports(self):
        """Get list of available MIDI input ports."""
        ports = mido.get_input_names()
        return ports if ports else []

    def is_connected(self):
        """Check if MIDI input is currently connected."""
        return self.midi_input is not None and self.midi_running

    def get_current_port(self):
        """Get name of currently connected port."""
        return self.current_port

    def connect(self, port_name):
        """
        Connect to a MIDI input port.

        Args:
            port_name: Name of MIDI port to connect to

        Returns:
            True if successful, False otherwise
        """
        # Disconnect existing connection
        self.disconnect()

        try:
            self.midi_input = mido.open_input(port_name)
            self.midi_running = True
            self.current_port = port_name

            # Start listener thread
            self.midi_thread = threading.Thread(
                target=self._midi_listener,
                daemon=True
            )
            self.midi_thread.start()

            print(f"✓ MIDI connected: {port_name}")
            return True

        except Exception as e:
            print(f"✗ MIDI connection failed: {e}")
            self.current_port = None
            return False

    def disconnect(self):
        """Disconnect from current MIDI port."""
        self.midi_running = False
        if self.midi_input:
            self.midi_input.close()
            self.midi_input = None
        self.current_port = None

    def _handle_midi_message(self, msg):
        """Process incoming MIDI message."""
        if msg.type == 'note_on' and msg.velocity > 0:
            self.synth.note_on(msg.note, msg.velocity)

            if self.on_note_callback:
                self.on_note_callback(msg.note, msg.velocity)

        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            self.synth.note_off(msg.note)

    def _midi_listener(self):
        """Background thread to listen for MIDI messages."""
        print("MIDI listener started...")
        try:
            while self.midi_running:
                msg = self.midi_input.receive()
                if msg:
                    self._handle_midi_message(msg)
        except Exception as e:
            print(f"MIDI listener error: {e}")
        finally:
            print("MIDI listener stopped.")

    def get_help_text(self):
        """Get help text for MIDI controls."""
        available = self.get_available_ports()
        if available:
            return (
                f"MIDI Hardware Controls:\n"
                f"  Connect to your MIDI keyboard and play\n"
                f"  Available ports: {', '.join(available)}"
            )
        else:
            return "MIDI Hardware Controls:\n  No MIDI devices detected"


class VirtualMIDIInput:
    """
    Creates a virtual MIDI port for inter-application communication.
    Other scripts can send MIDI to this port.
    """

    def __init__(self, synth, port_name="KarplusStrong Virtual In", on_note_callback=None):
        """
        Args:
            synth: RealtimeSynth instance
            port_name: Name for the virtual MIDI port
            on_note_callback: Optional callback for note events (for display updates)
        """
        self.synth = synth
        self.on_note_callback = on_note_callback
        self.port_name = port_name

        # Virtual MIDI state
        self.midi_input = None
        self.midi_thread = None
        self.midi_running = False

    def is_connected(self):
        """Check if virtual port is open."""
        return self.midi_input is not None and self.midi_running

    def get_port_name(self):
        """Get the name of the virtual port."""
        return self.port_name

    def open_port(self):
        """
        Open a virtual MIDI input port.

        Returns:
            True if successful, False otherwise
        """
        # Close existing port
        self.close_port()

        try:
            # Create virtual input port
            self.midi_input = mido.open_input(self.port_name, virtual=True)
            self.midi_running = True

            # Start listener thread
            self.midi_thread = threading.Thread(
                target=self._midi_listener,
                daemon=True
            )
            self.midi_thread.start()

            print(f"✓ Virtual MIDI port opened: '{self.port_name}'")
            print(f"  Other scripts can connect to this port to send MIDI")
            return True

        except Exception as e:
            print(f"✗ Failed to open virtual MIDI port: {e}")
            return False

    def close_port(self):
        """Close the virtual MIDI port."""
        self.midi_running = False
        if self.midi_input:
            self.midi_input.close()
            self.midi_input = None

    def _handle_midi_message(self, msg):
        """Process incoming MIDI message."""
        if msg.type == 'note_on' and msg.velocity > 0:
            self.synth.note_on(msg.note, msg.velocity)

            if self.on_note_callback:
                self.on_note_callback(msg.note, msg.velocity)

        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            self.synth.note_off(msg.note)

    def _midi_listener(self):
        """Background thread to listen for MIDI messages."""
        print(f"Virtual MIDI listener started on '{self.port_name}'...")
        try:
            while self.midi_running:
                msg = self.midi_input.receive()
                if msg:
                    self._handle_midi_message(msg)
        except Exception as e:
            print(f"Virtual MIDI listener error: {e}")
        finally:
            print("Virtual MIDI listener stopped.")

    def get_help_text(self):
        """Get help text for virtual MIDI."""
        return (
            f"Virtual MIDI Port:\n"
            f"  Port name: '{self.port_name}'\n"
            f"  Other Python scripts can send to this port\n"
            f"  Example: mido.open_output('{self.port_name}').send(msg)"
        )