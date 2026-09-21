import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Download,
  Loader2,
  Music4,
  Plug,
  RefreshCw,
} from "lucide-react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  api,
  ApiError,
  DEFAULT_BACKEND_URL,
  type DatasetInfo,
  type GenerationResult,
  type HealthInfo,
  type TrainingStatus,
} from "@/lib/api";
import { DEMO_DATASET, DEMO_TOKENS, DATASET_SOURCES } from "@/lib/demo-data";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Slider } from "@/components/ui/slider";
import { cn } from "@/lib/utils";

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-lg border border-border bg-secondary/50 px-3 py-2">
      <div className="text-[0.65rem] uppercase tracking-widest text-muted-foreground">{label}</div>
      <div className="mt-0.5 font-mono text-sm text-foreground">{value}</div>
    </div>
  );
}

function TokenList({ tokens }: { tokens: string[] }) {
  return (
    <div className="flex max-h-56 flex-wrap gap-1.5 overflow-y-auto rounded-lg bg-secondary/50 p-3">
      {tokens.map((token, index) => {
        const symbol = token.split("_")[0] ?? token;
        const isChord = symbol.includes(".") || /^\d+$/.test(symbol);
        const isRest = symbol === "REST";
        return (
          <span
            key={index + token}
            className={cn(
              "rounded px-1.5 py-0.5 font-mono text-[0.7rem]",
              isRest
                ? "bg-muted text-muted-foreground"
                : isChord
                  ? "bg-accent/25 text-accent-foreground"
                  : "bg-primary/20 text-foreground",
            )}
          >
            {token}
          </span>
        );
      })}
    </div>
  );
}

export function Studio() {
  const [baseUrl, setBaseUrl] = useState(DEFAULT_BACKEND_URL);
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [connecting, setConnecting] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);

  const [dataset, setDataset] = useState<DatasetInfo | null>(null);
  const [datasetError, setDatasetError] = useState<string | null>(null);

  const [epochs, setEpochs] = useState(30);
  const [batchSize, setBatchSize] = useState(64);
  const [sequenceLength, setSequenceLength] = useState(64);
  const [lstmUnits, setLstmUnits] = useState(256);
  const [training, setTraining] = useState<TrainingStatus | null>(null);
  const [trainError, setTrainError] = useState<string | null>(null);

  const [genLength, setGenLength] = useState(200);
  const [temperature, setTemperature] = useState(0.9);
  const [seedIndex, setSeedIndex] = useState<string>("");
  const [generating, setGenerating] = useState(false);
  const [generation, setGeneration] = useState<GenerationResult | null>(null);
  const [genError, setGenError] = useState<string | null>(null);

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const connected = health?.ok === true;

  const loadDataset = useCallback(async (url: string) => {
    try {
      setDataset(await api.dataset(url));
      setDatasetError(null);
    } catch (error) {
      setDataset(null);
      setDatasetError(error instanceof ApiError ? error.message : "Could not read the dataset.");
    }
  }, []);

  const connect = useCallback(async () => {
    setConnecting(true);
    setConnectionError(null);
    try {
      const info = await api.health(baseUrl);
      setHealth(info);
      await loadDataset(baseUrl);
      const status = await api.trainingStatus(baseUrl);
      setTraining(status);
    } catch (error) {
      setHealth(null);
      setConnectionError(error instanceof ApiError ? error.message : "Connection failed.");
    } finally {
      setConnecting(false);
    }
  }, [baseUrl, loadDataset]);

  // Poll live epoch metrics while a run is in flight.
  useEffect(() => {
    if (!connected || training?.status !== "training") {
      if (pollRef.current) clearInterval(pollRef.current);
      pollRef.current = null;
      return;
    }
    pollRef.current = setInterval(async () => {
      try {
        const status = await api.trainingStatus(baseUrl);
        setTraining(status);
        if (status.status !== "training") {
          setHealth((current) =>
            current ? { ...current, model_trained: status.model_trained } : current,
          );
        }
      } catch {
        /* transient: the next tick retries */
      }
    }, 1500);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [connected, training?.status, baseUrl]);

  const startTraining = async () => {
    setTrainError(null);
    try {
      await api.startTraining(baseUrl, {
        epochs,
        batch_size: batchSize,
        sequence_length: sequenceLength,
        lstm_units: lstmUnits,
        dropout: 0.3,
        optimizer: "adam",
      });
      setTraining({
        status: "training",
        message: "",
        epochs: [],
        summary: null,
        elapsed: 0,
        model_trained: false,
      });
    } catch (error) {
      setTrainError(error instanceof ApiError ? error.message : "Training could not start.");
    }
  };

  const runGeneration = async () => {
    setGenerating(true);
    setGenError(null);
    try {
      const parsedSeed = seedIndex.trim() === "" ? null : Number(seedIndex);
      const result = await api.generate(baseUrl, {
        length: genLength,
        temperature,
        seed_index: parsedSeed !== null && Number.isFinite(parsedSeed) ? parsedSeed : null,
      });
      setGeneration(result);
    } catch (error) {
      setGeneration(null);
      setGenError(error instanceof ApiError ? error.message : "Generation failed.");
    } finally {
      setGenerating(false);
    }
  };

  const epochData = training?.epochs ?? [];
  const latest = epochData[epochData.length - 1];
  const progress = latest ? (latest.epoch / latest.total_epochs) * 100 : 0;
  const modelTrained = training?.model_trained ?? health?.model_trained ?? false;

  return (
    <div className="space-y-6">
      {/* Backend connection */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            <Plug className="h-4 w-4 text-primary" /> Local backend
          </CardTitle>
          <CardDescription>
            Run <code className="font-mono text-foreground">uvicorn server:app --port 8000</code> in
            the project folder, then connect. Training and generation happen entirely on your
            machine.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap items-end gap-3">
            <div className="min-w-[16rem] flex-1">
              <Label htmlFor="backend-url">API URL</Label>
              <Input
                id="backend-url"
                value={baseUrl}
                onChange={(event) => setBaseUrl(event.target.value)}
                placeholder="http://localhost:8000"
                className="mt-1.5 font-mono"
              />
            </div>
            <Button onClick={connect} disabled={connecting}>
              {connecting ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <RefreshCw className="mr-2 h-4 w-4" />
              )}
              {connected ? "Reconnect" : "Connect"}
            </Button>
            <div className="flex items-center gap-2">
              <span
                className={cn(
                  "h-2.5 w-2.5 rounded-full",
                  connected ? "bg-chart-5" : "bg-muted-foreground",
                )}
              />
              <span className="text-sm text-muted-foreground">
                {connected ? "Connected" : "Offline — demo data only"}
              </span>
              {modelTrained && <Badge variant="secondary">Model trained</Badge>}
            </div>
          </div>

          {connectionError && (
            <Alert variant="destructive">
              <AlertTriangle className="h-4 w-4" />
              <AlertTitle>Cannot reach the API</AlertTitle>
              <AlertDescription>{connectionError}</AlertDescription>
            </Alert>
          )}
          {!connected && !connectionError && (
            <Alert>
              <AlertTriangle className="h-4 w-4" />
              <AlertTitle>No backend connected</AlertTitle>
              <AlertDescription>
                The bundled demo below shows the real token encoding so you can inspect the
                pipeline. Actual model training and music generation require the local Python API —
                nothing on this page is simulated.
              </AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Dataset */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Dataset</CardTitle>
          <CardDescription>
            Place .mid files in <code className="font-mono text-foreground">data/midi/</code>{" "}
            (subfolders included). 10-30 files by one composer is the sweet spot.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {connected ? (
            datasetError ? (
              <Alert variant="destructive">
                <AlertTriangle className="h-4 w-4" />
                <AlertTitle>Dataset problem</AlertTitle>
                <AlertDescription>{datasetError}</AlertDescription>
              </Alert>
            ) : dataset ? (
              <>
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <Stat label="Files parsed" value={dataset.files.filter((f) => f.status === "ok").length} />
                  <Stat label="Files skipped" value={dataset.files.filter((f) => f.status === "skipped").length} />
                  <Stat label="Total events" value={dataset.total_events} />
                  <Stat label="Folder" value={dataset.data_dir} />
                </div>
                <div className="max-h-56 overflow-y-auto rounded-lg border border-border">
                  {dataset.files.map((file) => (
                    <div
                      key={file.file}
                      className="flex items-center justify-between border-b border-border px-3 py-2 text-sm last:border-0"
                    >
                      <span className="truncate font-mono text-xs text-foreground">{file.file}</span>
                      <span className="flex items-center gap-2">
                        <span className="font-mono text-xs text-muted-foreground">
                          {file.tokens} events
                        </span>
                        <Badge variant={file.status === "ok" ? "secondary" : "destructive"}>
                          {file.status}
                        </Badge>
                      </span>
                    </div>
                  ))}
                </div>
                <Button variant="secondary" size="sm" onClick={() => loadDataset(baseUrl)}>
                  <RefreshCw className="mr-2 h-3.5 w-3.5" /> Rescan folder
                </Button>
              </>
            ) : null
          ) : (
            <>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <Stat label="Demo piece" value={DEMO_DATASET.name} />
                <Stat label="Source" value={DEMO_DATASET.license} />
                <Stat label="Events" value={DEMO_DATASET.events} />
                <Stat label="Vocabulary" value={DEMO_DATASET.vocabulary} />
              </div>
              <div>
                <p className="mb-2 text-sm text-muted-foreground">
                  Bundled sample, encoded exactly as <code className="font-mono">preprocess.py</code>{" "}
                  would encode it:
                </p>
                <TokenList tokens={DEMO_TOKENS} />
              </div>
            </>
          )}

          <div className="grid gap-2 sm:grid-cols-2">
            {DATASET_SOURCES.map((source) => (
              <a
                key={source.url}
                href={source.url}
                target="_blank"
                rel="noreferrer"
                className="rounded-lg border border-border bg-card p-3 transition-colors hover:border-primary/60"
              >
                <div className="text-sm font-medium text-foreground">{source.name}</div>
                <div className="mt-1 text-xs text-muted-foreground">{source.note}</div>
              </a>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Training */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Training</CardTitle>
          <CardDescription>
            Fits the two-layer LSTM on sliding windows of your corpus and checkpoints the best
            weights to <code className="font-mono text-foreground">artifacts/model.keras</code>.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <div>
              <Label htmlFor="epochs">Epochs</Label>
              <Input
                id="epochs"
                type="number"
                min={1}
                max={500}
                value={epochs}
                onChange={(e) => setEpochs(Number(e.target.value))}
                className="mt-1.5"
              />
            </div>
            <div>
              <Label htmlFor="batch">Batch size</Label>
              <Input
                id="batch"
                type="number"
                min={1}
                max={512}
                value={batchSize}
                onChange={(e) => setBatchSize(Number(e.target.value))}
                className="mt-1.5"
              />
            </div>
            <div>
              <Label htmlFor="seqlen">Sequence length</Label>
              <Input
                id="seqlen"
                type="number"
                min={8}
                max={256}
                value={sequenceLength}
                onChange={(e) => setSequenceLength(Number(e.target.value))}
                className="mt-1.5"
              />
            </div>
            <div>
              <Label htmlFor="units">LSTM units</Label>
              <Input
                id="units"
                type="number"
                min={32}
                max={512}
                value={lstmUnits}
                onChange={(e) => setLstmUnits(Number(e.target.value))}
                className="mt-1.5"
              />
            </div>
          </div>

          <Button onClick={startTraining} disabled={!connected || training?.status === "training"}>
            {training?.status === "training" ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : null}
            {training?.status === "training" ? "Training..." : "Start training"}
          </Button>
          {!connected && (
            <p className="text-xs text-muted-foreground">
              Connect the local API to train. Or run{" "}
              <code className="font-mono">python train.py --epochs 30</code> directly.
            </p>
          )}

          {trainError && (
            <Alert variant="destructive">
              <AlertTriangle className="h-4 w-4" />
              <AlertTitle>Training could not start</AlertTitle>
              <AlertDescription>{trainError}</AlertDescription>
            </Alert>
          )}
          {training?.status === "error" && (
            <Alert variant="destructive">
              <AlertTriangle className="h-4 w-4" />
              <AlertTitle>Training failed</AlertTitle>
              <AlertDescription>{training.message}</AlertDescription>
            </Alert>
          )}

          {epochData.length > 0 && (
            <div className="space-y-4">
              <div>
                <div className="mb-1.5 flex justify-between text-sm">
                  <span className="text-muted-foreground">
                    Epoch {latest?.epoch} / {latest?.total_epochs}
                  </span>
                  <span className="font-mono text-foreground">{Math.round(progress)}%</span>
                </div>
                <Progress value={progress} />
              </div>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <Stat label="Loss" value={latest ? latest.loss.toFixed(4) : "-"} />
                <Stat label="Accuracy" value={latest ? (latest.accuracy * 100).toFixed(2) + "%" : "-"} />
                <Stat
                  label="Val loss"
                  value={latest?.val_loss != null ? latest.val_loss.toFixed(4) : "-"}
                />
                <Stat label="Elapsed" value={Math.round(training?.elapsed ?? 0) + "s"} />
              </div>
              <div className="h-56 w-full rounded-lg border border-border bg-card p-3">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={epochData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                    <XAxis dataKey="epoch" stroke="var(--color-muted-foreground)" fontSize={11} />
                    <YAxis stroke="var(--color-muted-foreground)" fontSize={11} />
                    <Tooltip
                      contentStyle={{
                        background: "var(--color-card)",
                        border: "1px solid var(--color-border)",
                        borderRadius: 8,
                        color: "var(--color-foreground)",
                      }}
                    />
                    <Line
                      type="monotone"
                      dataKey="loss"
                      stroke="var(--color-chart-1)"
                      dot={false}
                      strokeWidth={2}
                    />
                    <Line
                      type="monotone"
                      dataKey="accuracy"
                      stroke="var(--color-chart-2)"
                      dot={false}
                      strokeWidth={2}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {training?.status === "done" && training.summary && (
            <Alert>
              <CheckCircle2 className="h-4 w-4" />
              <AlertTitle>Training finished</AlertTitle>
              <AlertDescription>
                {training.summary.epochs_run} epochs · final loss{" "}
                {training.summary.final_loss.toFixed(4)} · accuracy{" "}
                {(training.summary.final_accuracy * 100).toFixed(2)}% · vocabulary{" "}
                {training.summary.vocab_size} tokens · {training.summary.training_samples} samples.
              </AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Generation */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            <Music4 className="h-4 w-4 text-primary" /> Generate music
          </CardTitle>
          <CardDescription>
            Seeds the model with a window from the corpus, then samples one note at a time and
            writes a playable MIDI file.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          <div className="grid gap-5 sm:grid-cols-3">
            <div>
              <Label htmlFor="genlen">Notes to generate: {genLength}</Label>
              <Slider
                id="genlen"
                className="mt-3"
                min={16}
                max={800}
                step={8}
                value={[genLength]}
                onValueChange={(value) => setGenLength(value[0] ?? 200)}
              />
            </div>
            <div>
              <Label htmlFor="temp">Creativity (temperature): {temperature.toFixed(2)}</Label>
              <Slider
                id="temp"
                className="mt-3"
                min={0}
                max={2}
                step={0.05}
                value={[temperature]}
                onValueChange={(value) => setTemperature(value[0] ?? 1)}
              />
              <p className="mt-2 text-xs text-muted-foreground">
                0 = argmax (safe, repetitive) · 0.7-1.0 = balanced · 1.2+ = wild
              </p>
            </div>
            <div>
              <Label htmlFor="seed">Seed index (blank = random)</Label>
              <Input
                id="seed"
                className="mt-1.5"
                placeholder="random"
                value={seedIndex}
                onChange={(event) => setSeedIndex(event.target.value)}
              />
            </div>
          </div>

          <div className="flex flex-wrap gap-3">
            <Button onClick={runGeneration} disabled={!connected || !modelTrained || generating}>
              {generating ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
              Generate
            </Button>
            {generation && (
              <Button variant="secondary" asChild>
                <a href={api.downloadUrl(baseUrl, generation.download_url)} download>
                  <Download className="mr-2 h-4 w-4" /> Download {generation.filename}
                </a>
              </Button>
            )}
          </div>

          {connected && !modelTrained && (
            <Alert>
              <AlertTriangle className="h-4 w-4" />
              <AlertTitle>Model not trained yet</AlertTitle>
              <AlertDescription>
                Train the model first — generation needs artifacts/model.keras and
                artifacts/vocabulary.pkl.
              </AlertDescription>
            </Alert>
          )}
          {genError && (
            <Alert variant="destructive">
              <AlertTriangle className="h-4 w-4" />
              <AlertTitle>Generation error</AlertTitle>
              <AlertDescription>{genError}</AlertDescription>
            </Alert>
          )}

          {generation && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <Stat label="Events" value={generation.length} />
                <Stat label="Unique tokens" value={generation.unique_tokens} />
                <Stat label="Chords" value={generation.chord_count} />
                <Stat label="Rests" value={generation.rest_count} />
              </div>
              <div>
                <p className="mb-2 text-sm text-muted-foreground">Seed window from the corpus:</p>
                <TokenList tokens={generation.seed_tokens} />
              </div>
              <div>
                <p className="mb-2 text-sm text-muted-foreground">Generated sequence:</p>
                <TokenList tokens={generation.tokens} />
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
