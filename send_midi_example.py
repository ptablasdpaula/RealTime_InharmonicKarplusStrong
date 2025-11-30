"""
Example script: Send MIDI notes to the Karplus-Strong synth via virtual MIDI.

This demonstrates how another Python script can control the synth by
sending MIDI messages to its virtual port.

Usage:
1. Start the synth GUI with Virtual MIDI mode selected
2. Run this script in a separate terminal
3. Watch/hear the synth respond to programmatic MIDI
"""
import mido
import time

# Name of the virtual port (must match what's in virtual_midi_input.py)
VIRTUAL_PORT_NAME = "KarplusStrong Virtual In"


def play_scale():
    """Play a simple ascending C major scale."""
    try:
        # Connect to the virtual port
        output = mido.open_output(VIRTUAL_PORT_NAME)
        print(f"✓ Connected to virtual port: '{VIRTUAL_PORT_NAME}'")
        print("Playing C major scale...")

        # C major scale: C, D, E, F, G, A, B, C
        scale = [60, 62, 64, 65, 67, 69, 71, 72]  # MIDI note numbers

        for note in scale:
            # Note on
            msg = mido.Message('note_on', note=note, velocity=100)
            output.send(msg)
            print(f"  Sent: Note {note} ON")

            # Wait
            time.sleep(0.5)

            # Note off
            msg = mido.Message('note_off', note=note)
            output.send(msg)
            print(f"  Sent: Note {note} OFF")

            # Brief pause between notes
            time.sleep(0.1)

        print("✓ Scale complete!")
        output.close()

    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nMake sure:")
        print("  1. The synth GUI is running")
        print("  2. Virtual MIDI mode is selected")
        print(f"  3. The virtual port '{VIRTUAL_PORT_NAME}' is open")


def play_chord_progression():
    """Play a simple chord progression."""
    try:
        output = mido.open_output(VIRTUAL_PORT_NAME)
        print(f"✓ Connected to virtual port: '{VIRTUAL_PORT_NAME}'")
        print("Playing chord progression...")

        # Simple I-IV-V-I progression in C major
        chords = [
            [60, 64, 67],  # C major (C, E, G)
            [65, 69, 72],  # F major (F, A, C)
            [67, 71, 74],  # G major (G, B, D)
            [60, 64, 67],  # C major (C, E, G)
        ]

        for i, chord in enumerate(chords, 1):
            print(f"\n  Chord {i}: {chord}")

            # Play all notes in chord
            for note in chord:
                msg = mido.Message('note_on', note=note, velocity=80)
                output.send(msg)
                time.sleep(0.02)  # Slight stagger for nice effect

            # Hold chord
            time.sleep(1.0)

            # Release all notes
            for note in chord:
                msg = mido.Message('note_off', note=note)
                output.send(msg)

            # Brief pause
            time.sleep(0.2)

        print("\n✓ Chord progression complete!")
        output.close()

    except Exception as e:
        print(f"✗ Error: {e}")


def interactive_mode():
    """Interactive MIDI sender - type MIDI note numbers."""
    try:
        output = mido.open_output(VIRTUAL_PORT_NAME)
        print(f"✓ Connected to virtual port: '{VIRTUAL_PORT_NAME}'")
        print("\nInteractive Mode")
        print("Enter MIDI note numbers (0-127) to play, or 'q' to quit")
        print("Example: 60 = Middle C, 69 = A4")

        while True:
            user_input = input("\nMIDI note: ").strip()

            if user_input.lower() == 'q':
                break

            try:
                note = int(user_input)
                if 0 <= note <= 127:
                    # Send note on
                    msg = mido.Message('note_on', note=note, velocity=100)
                    output.send(msg)
                    print(f"  Sent: Note {note} ON")

                    # Auto-release after 1 second
                    time.sleep(1.0)

                    msg = mido.Message('note_off', note=note)
                    output.send(msg)
                    print(f"  Sent: Note {note} OFF")
                else:
                    print("  Note must be 0-127")
            except ValueError:
                print("  Invalid input, enter a number or 'q'")

        print("✓ Exiting interactive mode")
        output.close()

    except Exception as e:
        print(f"✗ Error: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("MIDI Sender Example")
    print("=" * 60)
    print("\nChoose a demo:")
    print("  1 - Play scale")
    print("  2 - Play chord progression")
    print("  3 - Interactive mode (type note numbers)")
    print("  q - Quit")

    choice = input("\nChoice: ").strip()

    if choice == '1':
        play_scale()
    elif choice == '2':
        play_chord_progression()
    elif choice == '3':
        interactive_mode()
    elif choice.lower() == 'q':
        print("Goodbye!")
    else:
        print("Invalid choice")