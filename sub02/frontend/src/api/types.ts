import type { components } from "./schema";

type S = components["schemas"];

export type FlareInput = S["FlareInput"];
export type FlareResult = S["FlareResult"];
export type Run = S["RunResponse"];
export type RunSummary = S["RunSummary"];
export type RunListItem = S["RunListItem"];
export type ModelInfo = S["ModelInfo"];
export type Curve = S["Curve"];

export type ExportFormat = "handoff_csv" | "full_csv" | "json";
