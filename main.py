from synth_gui import SynthGUI

def main():
    """Start the synthesizer application."""
    print("=" * 60)
    print("Inharmonic Karplus-Strong Synthesizer")
    print("=" * 60)
    print()

    gui = SynthGUI()
    gui.run()

if __name__ == "__main__":
    main()