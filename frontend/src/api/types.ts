/* Shared TypeScript types for DLKin API */

export interface PredictInput {
  protein: string;
  smiles: string;
}

export interface PredictResult {
  status: "ok";
  log10_km: number;
  km: number;
  log10_kcat: number;
  kcat: number;
  kcat_over_km: number;
}

export interface BatchResult {
  status: "ok";
  columns: string[];
  row_count: number;
  rows: Record<string, unknown>[];
}

export interface ClusterMetrics {
  n_total: number;
  n_clusters_excluding_noise: number;
  n_noise: number;
  noise_ratio: number;
  silhouette_umap: number;
  davies_bouldin_umap: number;
  calinski_harabasz_umap: number;
  shannon_entropy_normalized: number;
  effective_number_of_clusters: number;
}

export interface ClusteringResult {
  status: "ok";
  csv_preview: Record<string, unknown>[];
  umap_coords: [number, number][];
  labels: number[];
  metrics: ClusterMetrics;
}

export interface DockingJob {
  success: boolean;
  if_delta_REU: number | null;
  total_score_REU: number | null;
  work_dir: string;
}

export interface HealthCheck {
  status: string;
  models_loaded: boolean;
}
