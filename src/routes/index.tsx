import { createFileRoute } from "@tanstack/react-router";
import { Brain, Github, Music4, Terminal } from "lucide-react";

import { Pipeline } from "@/components/Pipeline";
import { ScriptsPanel } from "@/components/ScriptsPanel";
import { Studio } from "@/components/Studio";
import { TechnicalInfo } from "@/components/TechnicalInfo";
import { Badge } from "@/components/ui/badge";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "AI Music Generator using LSTM — train & generate MIDI" },
      {
        name: "description",
        content:
          "An end-to-end LSTM music generation project: parse MIDI with music21, train a Keras LSTM on note sequences, and generate new playable MIDI with temperature sampling.",
      },
      { property: "og:title", content: "AI Music Generator using LSTM" },
      {
        property: "og:description",
        content:
          "Preprocess MIDI with music21, train a two-layer Keras LSTM, and generate new music — with copyable Python scripts and a local FastAPI backend.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Index,
});

function Section({
  id,
  eyebrow,
  title,
  description,
  children,
}: {
  id: string;
  eyebrow: string;
  title: string;
  description: string;
  children: React.ReactNode;
}) {
  return (
    <section id={id} className="scroll-mt-20 space-y-5">
      <div>
        <div className="font-mono text-[0.7rem] uppercase tracking-[0.2em] text-primary">
          {eyebrow}
        </div>
        <h2 className="mt-1.5 text-2xl font-semibold tracking-tight text-foreground">{title}</h2>
        <p className="mt-1.5 max-w-3xl text-sm leading-relaxed text-muted-foreground">
          {description}
        </p>
      </div>
      {children}
    </section>
  );
}

function Index() {
  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card/60 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-5 py-4">
          <div className="flex items-center gap-2.5">
            <Music4 className="h-5 w-5 text-primary" />
            <span className="font-semibold tracking-tight text-foreground">LSTM Music Lab</span>
          </div>
          <nav className="hidden gap-5 text-sm text-muted-foreground md:flex">
            <a href="#pipeline" className="hover:text-foreground">
              Pipeline
            </a>
            <a href="#studio" className="hover:text-foreground">
              Studio
            </a>
            <a href="#code" className="hover:text-foreground">
              Python code
            </a>
            <a href="#theory" className="hover:text-foreground">
              How it works
            </a>
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-6xl space-y-16 px-5 py-12">
        <div className="space-y-5">
          <Badge variant="secondary" className="font-mono text-xs">
            Python · music21 · TensorFlow/Keras
          </Badge>
          <h1 className="max-w-3xl text-4xl font-semibold leading-tight tracking-tight text-foreground sm:text-5xl">
            AI Music Generator using <span className="text-primary">LSTM</span>
          </h1>
          <p className="max-w-3xl text-base leading-relaxed text-muted-foreground">
            A recurrent network learns which note tends to follow the last 64 notes of a piano
            piece. Feed it Chopin, Bach or Mozart MIDI files, train a two-layer LSTM with a softmax
            over the note vocabulary, then let it write its own sequence one note at a time and
            export the result as a playable MIDI file. Everything runs locally — the Python scripts
            below are the whole project, and this page drives them through a small FastAPI server.
          </p>
          <div className="grid gap-3 sm:grid-cols-3">
            {[
              {
                icon: Terminal,
                title: "Real code, not a mockup",
                body: "preprocess.py, model.py, train.py, generate.py and server.py are production-ready and copyable.",
              },
              {
                icon: Brain,
                title: "Genuine training",
                body: "Metrics on this page come from Keras callbacks on your machine. Nothing is simulated.",
              },
              {
                icon: Github,
                title: "Laptop-sized",
                body: "Two LSTM(256) layers, CPU-friendly, minutes per epoch on ~20 MIDI files.",
              },
            ].map((item) => (
              <div key={item.title} className="rounded-xl border border-border bg-card p-4">
                <item.icon className="h-4 w-4 text-primary" />
                <div className="mt-2 text-sm font-medium text-foreground">{item.title}</div>
                <div className="mt-1 text-xs leading-relaxed text-muted-foreground">{item.body}</div>
              </div>
            ))}
          </div>
        </div>

        <Section
          id="pipeline"
          eyebrow="Architecture"
          title="The pipeline, end to end"
          description="Click any stage to see what happens there and the code that does it."
        >
          <Pipeline />
        </Section>

        <Section
          id="studio"
          eyebrow="Run it"
          title="Training and generation studio"
          description="Connect the local Python API to train on your own MIDI folder, watch live epoch metrics, and download the generated MIDI file."
        >
          <Studio />
        </Section>

        <Section
          id="code"
          eyebrow="Source"
          title="Python project files"
          description="Copy or download each file into an empty folder, install the requirements, and the project runs standalone — the web UI is optional."
        >
          <ScriptsPanel />
        </Section>

        <Section
          id="theory"
          eyebrow="Technical detail"
          title="How LSTM music generation actually works"
          description="MIDI structure, the token encoding, the gates inside an LSTM cell, the training objective, and what temperature does to sampling."
        >
          <TechnicalInfo />
        </Section>
      </main>

      <footer className="border-t border-border py-8 text-center text-xs text-muted-foreground">
        Built for an ML internship task · MIDI datasets are public domain or openly licensed —
        check each source before redistributing.
      </footer>
    </div>
  );
}
