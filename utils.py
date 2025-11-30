def midi_to_hz(midi_note : int) -> float:
    return 440.0 * (2 ** ((midi_note - 69) / 12.0))

def hz_to_samples(hz: float, sr: int = 16000) -> float:
    return sr / hz