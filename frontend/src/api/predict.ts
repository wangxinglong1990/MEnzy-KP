import { api } from "./client";
import type { HealthCheck, PredictInput, PredictResult, BatchResult } from "./types";

export async function healthCheck() {
  const { data } = await api.get<HealthCheck>("/health");
  return data;
}

export async function predictSingle(input: PredictInput) {
  const { data } = await api.post<PredictResult>("/predict/single", input);
  return data;
}

export async function predictBatch(formData: FormData) {
  const { data } = await api.post<BatchResult>("/predict/batch", formData);
  return data;
}
