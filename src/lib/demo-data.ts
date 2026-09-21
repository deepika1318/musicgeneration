// Illustrative encoding of the opening of the public-domain Minuet in G (Petzold,
// long attributed to J.S. Bach), written in exactly the token format preprocess.py
// produces: "<pitch or chord>_<quarterLength>".
//
// This is bundled demo data so the encoding and pipeline can be inspected without
// a backend. It is NOT model output — real note prediction requires training the
// LSTM locally through the Python API.
export const DEMO_TOKENS: string[] = [
  "D5_1.0",
  "G4_0.25",
  "A4_0.25",
  "B4_0.25",
  "C5_0.25",
  "D5_0.5",
  "G4_0.5",
  "G4_0.5",
  "E5_1.0",
  "C5_0.25",
  "D5_0.25",
  "E5_0.25",
  "F#5_0.25",
  "G5_0.5",
  "G4_0.5",
  "G4_0.5",
  "C5_0.5",
  "D5_0.25",
  "C5_0.25",
  "B4_0.25",
  "A4_0.25",
  "B4_0.5",
  "C5_0.25",
  "B4_0.25",
  "A4_0.25",
  "G4_0.25",
  "F#4_0.5",
  "G4_0.25",
  "A4_0.25",
  "B4_0.5",
  "7.11.2_1.0",
  "REST_1.0",
  "G4_0.5",
  "A4_0.5",
  "B4_0.5",
  "C5_0.5",
  "D5_1.0",
  "0.4.7_1.0",
];

export const DEMO_DATASET = {
  name: "Minuet in G (excerpt)",
  composer: "Christian Petzold / J.S. Bach notebook",
  license: "Public domain",
  events: DEMO_TOKENS.length,
  vocabulary: new Set(DEMO_TOKENS).size,
};

export const DATASET_SOURCES = [
  {
    name: "Classical Piano MIDI Page",
    url: "http://www.piano-midi.de",
    note: "Chopin, Bach, Mozart, Beethoven — small, clean, single-instrument piano files. Best starting point.",
  },
  {
    name: "MAESTRO (Magenta)",
    url: "https://magenta.tensorflow.org/datasets/maestro",
    note: "200 hours of virtuosic piano performance. Use a 20-file subset on a laptop.",
  },
  {
    name: "Lakh MIDI Dataset",
    url: "https://colinraffel.com/projects/lmd/",
    note: "176k multi-genre MIDI files. Filter to piano-only tracks before training.",
  },
  {
    name: "Video game piano MIDIs",
    url: "https://www.vgmusic.com",
    note: "The classic Final Fantasy set used in most LSTM music tutorials.",
  },
];
