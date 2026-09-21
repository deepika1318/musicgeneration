import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";

const SECTIONS = [
  {
    id: "stack",
    title: "Python stack: music21, TensorFlow/Keras, NumPy",
    body: [
      "music21 is an MIT computational musicology toolkit. It parses MIDI into a tree of Score / Part / Measure objects containing Note, Chord and Rest elements, each with a pitch, an offset (position in quarter notes from the start) and a duration.quarterLength. It also writes streams back out as MIDI, which is how the generated sequence becomes a playable file.",
      "TensorFlow/Keras provides the LSTM layers, the training loop (model.fit), checkpointing and the softmax classifier head. NumPy holds the (samples, sequence_length, 1) input tensor and the one-hot target matrix.",
      "Nothing here needs a GPU: two LSTM(256) layers over a 64-step window with a vocabulary of a few hundred tokens trains in minutes per epoch on a laptop CPU.",
    ],
  },
  {
    id: "midi",
    title: "MIDI file structure",
    body: [
      "MIDI stores events, not audio. A file is a set of tracks; each track is a time-ordered list of messages: note_on (pitch 0-127, velocity 0-127), note_off, program_change (instrument), tempo and time-signature meta events. Timing is measured in ticks, with ticks-per-quarter-note declared in the header.",
      "Because it is symbolic, a MIDI file is tiny and perfectly quantifiable — which is exactly why it suits sequence models. Pitch 60 is middle C (C4); an octave is 12 semitones, so pitch class = pitch % 12.",
      "This project ignores velocity and tempo and models pitch plus duration. That keeps the vocabulary small; adding velocity buckets to the token is a natural extension.",
    ],
  },
  {
    id: "encoding",
    title: "Encoding musical events as tokens",
    body: [
      "Every event becomes one string token: a single note is its pitch name with octave ('C4', 'F#5'); a chord is the dotted normal-order pitch classes ('4.7.11' = E, G, B); a silence is 'REST'. The quantised duration is appended after an underscore, so 'C4_0.5' is an eighth note middle C.",
      "Durations snap to eight buckets (0.25 to 4.0 quarter notes). Without that quantisation, expressive performance data produces thousands of near-duplicate tokens and the softmax never converges.",
      "The sorted set of distinct tokens is the vocabulary. note_to_int maps token to id, int_to_note maps back. Ids are divided by the vocabulary size so the LSTM sees inputs in [0, 1).",
    ],
  },
  {
    id: "lstm",
    title: "LSTM architecture and gates",
    body: [
      "A plain RNN multiplies the same weight matrix at every timestep, so gradients over 64 steps either vanish or explode. An LSTM cell adds a cell state C_t that information can travel along almost unchanged, plus three gates that are each a sigmoid over the concatenated [previous hidden state, current input].",
      "Forget gate f_t = sigma(W_f . [h_(t-1), x_t] + b_f) decides how much of the old cell state survives. Input gate i_t decides how much of the new candidate C~_t = tanh(W_C . [h_(t-1), x_t] + b_C) is written. The cell updates as C_t = f_t * C_(t-1) + i_t * C~_t. Output gate o_t gates what becomes the hidden state: h_t = o_t * tanh(C_t).",
      "Musically, that additive cell-state path is what lets the network remember the key and the motif established 40 notes earlier while the gates decide, note by note, what to keep.",
      "Layer 1 uses return_sequences=True so it emits a hidden state per timestep for layer 2. Layer 2 returns only its final state, a fixed-length summary of the whole window. Dropout(0.3) zeroes random activations during training so the model generalises rather than memorising a single piece.",
    ],
  },
  {
    id: "prediction",
    title: "Sequence prediction and training objective",
    body: [
      "This is next-token classification, identical in shape to character-level language modelling. Input: the last 64 tokens. Output: a softmax distribution over every token in the vocabulary. Loss: categorical cross-entropy, -sum(y_true * log(y_pred)), which is minimised when the true next note gets probability 1.",
      "Adam (adaptive moments) is the default optimizer; RMSprop is the classic choice for RNNs and is supported via --optimizer rmsprop. Accuracy here means 'the argmax token was exactly right' — 30-60% is normal and good music does not require higher, because many continuations are musically valid.",
      "Generation is autoregressive: predict, sample, append, drop the oldest token, repeat. The model's own outputs become its inputs, so errors compound — which is why temperature matters.",
    ],
  },
  {
    id: "temperature",
    title: "Temperature sampling",
    body: [
      "Temperature reshapes the softmax before sampling: p_i' = exp(log(p_i) / T) / sum_j exp(log(p_j) / T).",
      "T -> 0 (argmax): always the single most likely note. Deterministic, harmonically safe, usually loops after a few bars.",
      "T = 0.7-1.0: the sweet spot. Coherent phrasing with genuine variation.",
      "T > 1.2: flattens the distribution, so unlikely notes get picked. Adventurous, frequently out of key.",
    ],
  },
];

export function TechnicalInfo() {
  return (
    <Accordion type="single" collapsible defaultValue="lstm" className="w-full">
      {SECTIONS.map((section) => (
        <AccordionItem key={section.id} value={section.id}>
          <AccordionTrigger className="text-left text-base font-semibold">
            {section.title}
          </AccordionTrigger>
          <AccordionContent>
            <div className="space-y-3 text-sm leading-relaxed text-muted-foreground">
              {section.body.map((paragraph) => (
                <p key={paragraph.slice(0, 32)}>{paragraph}</p>
              ))}
            </div>
          </AccordionContent>
        </AccordionItem>
      ))}
    </Accordion>
  );
}
