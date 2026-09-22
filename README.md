# Melody Muse

Implement the requested AI Music Generator using LSTM project now; use internal planning and do not present another implementation plan for user approval.

### User Request
PROJECT: AI Music Generator using LSTM

GOAL:
Create an AI music generation project that demonstrates how an LSTM neural network can learn patterns from MIDI note sequences and generate new musical sequences. Functional machine-learning project, not just a UI mockup — do not fake generated music, model training, accuracy, or results.

CORE REQUIREMENTS:
1. MIDI DATA:
- Suitable publicly available classical MIDI dataset setup (e.g., Chopin, Mozart, Bach, or video game piano MIDIs) with sample data / download instructions.
- Clear file placement instructions (e.g., `data/midi/`).
- Support multiple MIDI files.

2. MIDI PREPROCESSING (Python + music21):
- Read MIDI files
- Extract notes and chords (handling Chord vs Note objects, pitch offsets, durations)
- Convert musical information into numerical sequences with vocabulary mappings
- Create fixed-length training sequences (e.g., sequence length of 32 or 64 notes)
- Normalize and one-hot encode / prepare input X and target y for categorical cross-entropy training

3. LSTM MODEL (TensorFlow / Keras):
- Sequential or functional model with LSTM layers (e.g., LSTM(256) with Dropout), Dense output with softmax over note vocabulary
- Categorical cross-entropy loss and Adam/RMSprop optimizer
- Keep model lightweight and runnable on a standard laptop

4. MUSIC GENERATION:
- Autoregressive generation from a seed sequence
- Sampling with temperature or argmax to predict next note, updating sliding window
- Convert generated sequence back into a music21 stream and output playable MIDI file

5. OUTPUT & CONTROLS:
- Train the model, generate music, download the generated MIDI file, view generation stats and note sequences

6. WEB INTERFACE:
- Project title & explanation of LSTM music generation
- Dataset info & sample MIDI management
- Training controls (epochs, batch size, sequence length)
- Real-time training progress/status and metrics visualization
- Music generation controls (seed selection, sequence length, temperature/creativity)
- Generated music sequence display and MIDI download
- Local backend connection support (connecting the web UI to the local Python API endpoint e.g. FastAPI/Flask on localhost) so users can run real training and generation locally, alongside bundled demo data

7. VISUAL PIPELINE:
- Clear interactive visual pipeline: MIDI Dataset -> MIDI Parsing -> Note/Chord Extraction -> Sequence Creation -> LSTM Training -> Music Generation -> Generated MIDI

8. TECHNICAL INFORMATION:
- Detailed breakdown of Python, music21, TensorFlow/Keras, LSTM architecture & gates, MIDI file structure, and sequence prediction

9. ERROR HANDLING:
- Missing MIDI files, invalid/corrupt MIDIs, insufficient data, model untrained status, generation errors

10. PROJECT ARCHITECTURE:
- A clean modern web UI paired with clearly separated, copyable/downloadable production-ready Python scripts (`preprocess.py`, `model.py`, `train.py`, `generate.py`, `server.py`, `requirements.txt`, and README with exact local run instructions). Keep scope suitable for an internship ML task.

This project was built with [Lovable](https://lovable.dev).

**Live app**: https://musicgeneration.lovable.app

## Build with Lovable

Continue developing this project in the [Lovable editor](https://lovable.dev/projects/d224df71-6743-4925-94aa-1a1c61aa43c7).

- **Ship faster**: describe what you want to build and Lovable handles the code.
- **Stay in sync**: every change made in Lovable is committed straight to this repository.
- **Full ownership**: this code is yours. Push to `main` on GitHub and your changes sync back into Lovable, ready for your next prompt.

## Development

Prefer working locally? You need Node.js and npm — [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

```sh
git clone <this-repository-url>
cd <repository-name>
npm i
npm run dev
```
