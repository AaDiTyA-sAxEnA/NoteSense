import streamlit as st
import numpy as np
import librosa
import io
import matplotlib.pyplot as plt

A4_FREQ = 440.0
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
              'F#', 'G', 'G#', 'A', 'A#', 'B']

# ---------- Math utilities ----------
def midi_from_freq(freq):
    return 69 + 12 * np.log2(freq / A4_FREQ)

def freq_from_midi(m):
    return A4_FREQ * (2 ** ((m - 69) / 12))

def freq_to_note_name(freq):
    if freq <= 0:
        return None, None, None
    midi = midi_from_freq(freq)
    midi_rounded = int(round(midi))
    note_index = midi_rounded % 12
    octave = (midi_rounded // 12) - 1
    target_freq = freq_from_midi(midi_rounded)
    return NOTE_NAMES[note_index], octave, target_freq

def classify_octave(octave):
    if octave <= 3:
        return "Lower"
    elif 4 <= octave <= 5:
        return "Middle"
    else:
        return "Higher"

def detect_pitch(signal, sr):
    f0 = librosa.yin(signal, fmin=50, fmax=5000, sr=sr)
    f0 = f0[np.isfinite(f0)]
    if len(f0) == 0:
        return None
    return float(np.median(f0))

# ---------- Streamlit App ----------
st.title("🎵 Web Note & Octave Detector")

uploaded = st.file_uploader("Upload an audio file (WAV or MP3)", type=["wav", "mp3"])

if uploaded:
    y, sr = librosa.load(io.BytesIO(uploaded.read()), sr=None)
    detected = detect_pitch(y, sr)
    if detected:
        note, octave, target_freq = freq_to_note_name(detected)
        octave_name = classify_octave(octave)
        cents = 1200 * np.log2(detected / target_freq)
        st.success(f"Detected: {note} ({octave_name} octave)")
        st.info(f"Frequency: {detected:.2f} Hz  |  Deviation: {cents:+.1f} cents")
        plot_note_log_scale(detect_freq=detected)
    else:
        st.warning("No pitch detected.")


# ---------- Explanations ----------
st.header("📘 Formulas (short)")
with st.expander("Show formulas"):
    st.markdown(
        "- Pitch estimate: use YIN to get fundamental frequency `f`.\n"
        "- MIDI number: `MIDI = 69 + 12 * log2(f / 440)`.\n"
        "- Note: `note_index = round(MIDI) % 12` → map to chromatic names.\n"
        "- Octave: `octave = (round(MIDI) // 12) - 1`.\n"
        "- Target freq for that note: `f_target = 440 * 2^((round(MIDI)-69)/12)`.\n"
        "- Deviation (cents): `1200 * log2(f / f_target)`."
    )

st.header("🔁 Cycle of Reasoning")
with st.expander("Show reasoning flow"):
    st.markdown("""
    1. Upload an audio sample.  
    2. Extract waveform data and compute the fundamental frequency using YIN.  
    3. Convert frequency to a logarithmic MIDI scale centered on A4 = 440 Hz.  
    4. Round to nearest semitone to get a chromatic note and determine octave.  
    5. Calculate ideal equal-tempered frequency for that note.  
    6. Compute deviation (in cents) between actual and ideal.  
    7. Present results with a logarithmic frequency chart marking your detected tone.
    """)
