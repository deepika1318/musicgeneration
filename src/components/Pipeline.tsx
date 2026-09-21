import { useState } from "react";
import { cn } from "@/lib/utils";

const STAGES = [
  {
    id: "dataset",
    title: "MIDI Dataset",
    short: "data/midi/*.mid",
    detail:
      "10-30 piano MIDI files from one composer, dropped into data/midi/. Subfolders are scanned recursively; corrupt files are skipped with a warning instead of killing the run.",
    code: "paths = glob.glob('data/midi/**/*.mid', recursive=True)",
  },
  {
    id: "parse",
    title: "MIDI Parsing",
    short: "music21.converter",
    detail:
      "converter.parse() loads each file. partitionByInstrument() picks a single part (usually piano) so overlapping instruments do not scramble the sequence.",
    code: "score = converter.parse(path)\nparts = instrument.partitionByInstrument(score)",
  },
  {
    id: "extract",
    title: "Note / Chord Extraction",
    short: "Note | Chord | Rest",
    detail:
      "Note objects become their pitch name ('C4'). Chord objects become dotted pitch-class normal order ('4.7.11'). Offset gaps become rests. Each event carries a quantised duration.",
    code: "'C4_0.5'   # note + quarterLength\n'4.7.11_1.0'  # chord\n'REST_1.0'",
  },
  {
    id: "sequence",
    title: "Sequence Creation",
    short: "sliding window",
    detail:
      "A window of 64 tokens is the input, token 65 is the target. The window slides one step at a time, so N events give N-64 training samples. Inputs are normalised by vocabulary size, targets are one-hot.",
    code: "X.shape = (samples, 64, 1)   # ids / vocab_size\ny.shape = (samples, vocab_size)  # one-hot",
  },
  {
    id: "train",
    title: "LSTM Training",
    short: "2x LSTM(256) + softmax",
    detail:
      "Two stacked LSTM layers with dropout, then a softmax over the whole note vocabulary. Categorical cross-entropy with Adam. Best weights are checkpointed each epoch.",
    code: "model.fit(X, y, epochs=30, batch_size=64,\n          validation_split=0.1)",
  },
  {
    id: "generate",
    title: "Music Generation",
    short: "autoregressive sampling",
    detail:
      "A random 64-token window from the corpus seeds the model. The predicted distribution is sampled with temperature, the token is appended, the window slides — repeat for the requested length.",
    code: "p = model.predict(window)[0]\nnext_id = sample_with_temperature(p, 0.9)",
  },
  {
    id: "midi",
    title: "Generated MIDI",
    short: "output/generated.mid",
    detail:
      "Tokens are turned back into music21 Note and Chord objects at running offsets, collected in a Stream, and written with stream.write('midi'). Open it in MuseScore, a DAW, or any MIDI player.",
    code: "stream.Stream(notes).write('midi', fp='output/generated.mid')",
  },
];

export function Pipeline() {
  const [active, setActive] = useState(STAGES[0].id);
  const stage = STAGES.find((s) => s.id === active) ?? STAGES[0];

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-stretch gap-2">
        {STAGES.map((item, index) => (
          <div key={item.id} className="flex items-stretch gap-2">
            <button
              onClick={() => setActive(item.id)}
              className={cn(
                "min-w-[9.5rem] rounded-xl border px-4 py-3 text-left transition-all",
                item.id === active
                  ? "border-primary bg-primary/15 shadow-[0_0_0_1px_var(--color-primary)]"
                  : "border-border bg-card hover:border-primary/50 hover:bg-secondary",
              )}
            >
              <div className="text-[0.65rem] font-mono uppercase tracking-widest text-muted-foreground">
                Step {index + 1}
              </div>
              <div className="mt-1 text-sm font-semibold text-foreground">{item.title}</div>
              <div className="mt-0.5 font-mono text-[0.68rem] text-muted-foreground">
                {item.short}
              </div>
            </button>
            {index < STAGES.length - 1 && (
              <div className="hidden self-center text-muted-foreground lg:block">&rarr;</div>
            )}
          </div>
        ))}
      </div>

      <div className="rounded-xl border border-border bg-card p-5">
        <h4 className="text-base font-semibold text-foreground">{stage.title}</h4>
        <p className="mt-2 max-w-3xl text-sm leading-relaxed text-muted-foreground">
          {stage.detail}
        </p>
        <pre className="mt-4 overflow-x-auto rounded-lg bg-secondary p-4 font-mono text-xs leading-relaxed text-foreground">
          {stage.code}
        </pre>
      </div>
    </div>
  );
}
