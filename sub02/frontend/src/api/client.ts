import type { ExportFormat, FlareInput, ModelInfo, Run, RunListItem } from "./types";

// One environment variable for the API address (see .env.example).
const BASE = ((import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://localhost:8000").replace(/\/$/, "");
export const API = `${BASE}/api/v1`;

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public code: string,
    public details?: unknown,
  ) {
    super(message);
  }
  get unreachable(): boolean {
    return this.status === 0;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API}${path}`, init);
  } catch {
    throw new ApiError("Cannot reach the server. Check that the API is running and try again.", 0, "network");
  }
  if (!res.ok) {
    let body: { error?: { code?: string; message?: string; details?: unknown } } = {};
    try {
      body = await res.json();
    } catch {
      /* not JSON */
    }
    throw new ApiError(body.error?.message ?? `The server answered with an error (${res.status}).`, res.status,
      body.error?.code ?? "http_error", body.error?.details);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  modelInfo: () => request<ModelInfo>("/model-info"),
  samples: () => request<FlareInput[]>("/samples"),
  predict: (flares: FlareInput[], saveRun: boolean, runName: string | null) =>
    request<Run>("/predict", json({ flares, save_run: saveRun, run_name: runName })),
  predictFile: (file: File, saveRun: boolean, runName: string | null) => {
    const form = new FormData();
    form.append("file", file);
    form.append("save_run", String(saveRun));
    if (runName) form.append("run_name", runName);
    return request<Run>("/predict/file", { method: "POST", body: form });
  },
  runs: () => request<RunListItem[]>("/runs"),
  run: (id: string) => request<Run>(`/runs/${id}`),
  deleteRun: (id: string) => request<void>(`/runs/${id}`, { method: "DELETE" }),
};

export const exportUrl = (runId: string, format: ExportFormat) => `${API}/runs/${runId}/export?format=${format}`;
