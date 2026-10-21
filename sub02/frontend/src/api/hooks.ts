import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { ApiError, api } from "./client";
import type { FlareInput, Run } from "./types";

export const LATEST = "latest"; // route id for a run that was not saved

export const useModelInfo = () => useQuery({ queryKey: ["model-info"], queryFn: api.modelInfo });
export const useRuns = () => useQuery({ queryKey: ["runs"], queryFn: api.runs });

/** A saved run from the API, or the last unsaved run kept in the query cache. */
export function useRun(runId: string | undefined) {
  const qc = useQueryClient();
  return useQuery({
    queryKey: ["run", runId],
    queryFn: async (): Promise<Run> => {
      const cached = qc.getQueryData<Run>(["run", runId]);
      if (runId === LATEST) {
        if (cached) return cached;
        throw new Error("No unsaved run is available.");
      }
      return api.run(runId as string);
    },
    enabled: !!runId,
    staleTime: runId === LATEST ? Infinity : 30_000,
    // retry once for network or server errors, never for "not found" and other client errors
    retry: (count, err) => runId !== LATEST && count < 1 && !(err instanceof ApiError && err.status >= 400 && err.status < 500),
  });
}

export interface PredictInput {
  file?: File | null;
  flares?: FlareInput[] | null;
  saveRun: boolean;
  runName: string | null;
}

export function usePredict() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ file, flares, saveRun, runName }: PredictInput) =>
      file ? api.predictFile(file, saveRun, runName) : api.predict(flares ?? [], saveRun, runName),
    onSuccess: (run) => {
      const id = run.run_id ?? LATEST;
      qc.setQueryData(["run", id], run);
      if (run.run_id) qc.invalidateQueries({ queryKey: ["runs"] });
    },
  });
}

export const useSamples = () => useMutation({ mutationFn: api.samples });

export function useDeleteRun() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.deleteRun,
    onSuccess: (_d, id) => {
      qc.removeQueries({ queryKey: ["run", id] });
      qc.invalidateQueries({ queryKey: ["runs"] });
    },
  });
}
