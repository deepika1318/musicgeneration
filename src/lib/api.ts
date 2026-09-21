export type DatasetFile = {
  file: string;
  status: "ok" | "skipped";
  tokens: number;
};

export type DatasetInfo = {
  files: DatasetFile[];
  total_events: number;
  data_dir: string;
};

export type EpochMetric = {
  epoch: number;
  total_epochs: number;
  loss: number;
  accuracy: number;
  val_loss: number | null;
  val_accuracy: number | null;
};

export type TrainingStatus = {
  status: "idle" | "training" | "done" | "error";
  message: string;
  epochs: EpochMetric[];
  summary: null | {
    epochs_run: number;
    final_loss: number;
    final_accuracy: number;
    vocab_size: number;
    training_samples: number;
    sequence_length: number;
  };
  elapsed: number;
  model_trained: boolean;
};

export type HealthInfo = {
  ok: boolean;
  data_dir: string;
  model_trained: boolean;
  training_status: string;
};

export type GenerationResult = {
  tokens: string[];
  seed_tokens: string[];
  length: number;
  unique_tokens: number;
  temperature: number;
  chord_count: number;
  rest_count: number;
  filename: string;
  download_url: string;
};

export type TrainConfig = {
  epochs: number;
  batch_size: number;
  sequence_length: number;
  lstm_units: number;
  dropout: number;
  optimizer: string;
};

export type GenerateConfig = {
  length: number;
  temperature: number;
  seed_index: number | null;
};

export const DEFAULT_BACKEND_URL = "http://localhost:8000";

export function normalizeBaseUrl(url: string): string {
  const trimmed = url.trim().replace(/\/+$/, "");
  if (!trimmed) return DEFAULT_BACKEND_URL;
  return /^https?:\/\//.test(trimmed) ? trimmed : "http://" + trimmed;
}

export class ApiError extends Error {}

async function request<T>(baseUrl: string, path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(normalizeBaseUrl(baseUrl) + path, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
  } catch {
    throw new ApiError(
      "Could not reach the Python API. Is `uvicorn server:app --port 8000` running?",
    );
  }
  if (!response.ok) {
    let detail = response.status + " " + response.statusText;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body?.detail) detail = body.detail;
    } catch {
      /* keep the status text */
    }
    throw new ApiError(detail);
  }
  return (await response.json()) as T;
}

export const api = {
  health: (baseUrl: string) => request<HealthInfo>(baseUrl, "/health"),
  dataset: (baseUrl: string) => request<DatasetInfo>(baseUrl, "/dataset"),
  startTraining: (baseUrl: string, config: TrainConfig) =>
    request<{ started: boolean }>(baseUrl, "/train", {
      method: "POST",
      body: JSON.stringify(config),
    }),
  trainingStatus: (baseUrl: string) => request<TrainingStatus>(baseUrl, "/train/status"),
  generate: (baseUrl: string, config: GenerateConfig) =>
    request<GenerationResult>(baseUrl, "/generate", {
      method: "POST",
      body: JSON.stringify(config),
    }),
  downloadUrl: (baseUrl: string, path: string) => normalizeBaseUrl(baseUrl) + path,
};
