import type { PredictResult } from "../../api/types";

export function PredictionResult({ result }: { result: PredictResult }) {
  return (
    <div className="mt-6 rounded-xl border border-gray-200 bg-white p-6">
      <h3 className="mb-4 font-semibold text-gray-900">Prediction Results</h3>
      <div className="grid grid-cols-2 gap-4">
        <ResultItem label="Km" value={result.km} unit="M" sci />
        <ResultItem label="kcat" value={result.kcat} unit="s⁻¹" sci />
        <ResultItem label="log10(Km)" value={result.log10_km} />
        <ResultItem label="log10(kcat)" value={result.log10_kcat} />
        <ResultItem label="kcat / Km" value={result.kcat_over_km} unit="" sci />
      </div>
    </div>
  );
}

function ResultItem({ label, value, unit, sci }: { label: string; value: number; unit?: string; sci?: boolean }) {
  return (
    <div className="rounded-lg bg-gray-50 p-3">
      <div className="mb-1 text-xs font-medium text-gray-400 uppercase">{label}</div>
      <div className="font-mono text-lg font-semibold text-gray-900">
        {sci ? value.toExponential(3) : value.toFixed(4)}
        {unit && <span className="ml-1 text-sm font-normal text-gray-500">{unit}</span>}
      </div>
    </div>
  );
}
