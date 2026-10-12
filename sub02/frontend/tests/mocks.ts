import { HttpResponse, http } from "msw";
import { setupServer } from "msw/node";

import type { FlareResult, ModelInfo, Run } from "../src/api/types";

export const API = "http://localhost:8000/api/v1";

export function flare(over: Partial<FlareResult> = {}): FlareResult {
  return {
    flare_id: "TESS_1", star_id: 1, status: "ok", calibrated_probability: 0.51, confidence_lower: 0.32,
    confidence_upper: 0.59, predicted_label: "eruptive", reliability: "confident", features_missing: 0,
    out_of_range: false, warnings: [], raw_score: 0.9, conformal_set: "E",
    features: { rise10_min: 6.1, g5: null },
    curve: { t_min: [-20, -10, 0, 10, 30], g: [0, 0.01, 1, 0.5, 0.2] },
    ...over,
  };
}

export const notOk = (id: string, status: FlareResult["status"]): FlareResult =>
  flare({ flare_id: id, status, calibrated_probability: null, confidence_lower: null, confidence_upper: null,
    predicted_label: null, reliability: null, features: null, curve: null, features_missing: null });

export function run(results: FlareResult[], over: Partial<Run> = {}): Run {
  const ok = results.filter((r) => r.status === "ok");
  return {
    run_id: "abc", name: "test run", model_version: "3", created_at: "2026-10-21T10:00:00",
    summary: {
      n_flares: results.length, n_ok: ok.length,
      n_eruptive: ok.filter((r) => r.predicted_label === "eruptive").length,
      n_confined: ok.filter((r) => r.predicted_label === "confined").length,
      n_confident: ok.filter((r) => r.reliability === "confident").length,
      n_uncertain: ok.filter((r) => r.reliability === "uncertain").length,
    },
    results, ...over,
  };
}

export const modelInfo: ModelInfo = {
  bundle_version: "3", created: "2026-10-21T10:00:00", data_hash: "abc123", training_summary: {},
  limitations: ["Trained on solar flares only."],
  metrics: {
    cv5_grouped: { tss: 0.149, auc: 0.605, accuracy: 0.6, hit_rate: 0.47, clear_rate: 0.67, balanced_accuracy: 0.57 },
    "solar_test_raw_0.5": { tss: 0.058, auc: 0.588, accuracy: 0.56, hit_rate: 0.38, clear_rate: 0.67, balanced_accuracy: 0.52 },
    solar_test_conformal_weighted: { coverage_eruptive: 0.9, coverage_confined: 0.93, single_label_share: 0.24, single_label_accuracy: 0.69 },
    brier_test: 0.2456, brier_test_base_rate_baseline: 0.2446, base_rate_cal: 0.263,
    n_solar_with_features: 712, n_eruptive: 250, split: { train: 321, calibration: 137, test: 254 },
  },
};

export const server = setupServer(
  http.get(`${API}/model-info`, () => HttpResponse.json(modelInfo)),
  http.get(`${API}/runs`, () => HttpResponse.json([])),
  http.get(`${API}/runs/abc`, () => HttpResponse.json(run([flare()]))),
);
