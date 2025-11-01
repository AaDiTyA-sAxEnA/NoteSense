import streamlit as st
import numpy as np
import librosa
import io

A4_FREQ = 440.0
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
              'F#', 'G', 'G#', 'A', 'A#', 'B']

def freq_to_note_name(freq):
    if freq <= 0:
        return None, None
    midi = 69 + 12 * np.log2(freq / A4_FREQ)
    midi = int(round(midi))
    note_index = midi % 12
    octave = (midi // 12) - 1
    return NOTE_NAMES[note_index], octave

def classify_octave(octave):
    if octave <= 3:
        return "Lower"
    elif 4 <= octave <= 5:
        return "Middle"
    else:
        return "Higher"

def detect_pitch(signal, sr):
    f0 = librosa.yin(signal, fmin=50, fmax=2000, sr=sr)
    f0 = f0[np.isfinite(f0)]
    if len(f0) == 0:
        return None
    return np.median(f0)

st.title("🎵 Chromatic Note and Octave Detector")

uploaded = st.file_uploader("Upload an audio file (WAV or MP3)", type=["wav", "mp3"])

if uploaded:
    y, sr = librosa.load(io.BytesIO(uploaded.read()), sr=None)
    pitch = detect_pitch(y, sr)
    if pitch:
        note, octave = freq_to_note_name(pitch)
        octave_name = classify_octave(octave)
        st.write(f"{note} ({octave_name} octave)")
    else:
        st.write("No pitch detected.")
