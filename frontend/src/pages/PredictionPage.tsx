import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { predictSingle } from "../api/predict";
import { PredictionResult } from "../components/prediction/PredictionResult";
import type { PredictResult } from "../api/types";

const EXAMPLE_PROTEIN = "MKFLILFNILVSTLATSLPLLAKPKEAKPEAKPEAKPEAKPEAKPEAQAKPEAKPEAQATPKAELAKPKADLAKPKELAKPKEAKPEAKPESLAKPKEA";
const EXAMPLE_SMILES = "COC(=O)CC[NH3+]";

export function PredictionPage() {
  const [protein, setProtein] = useState("");
  const [smiles, setSmiles] = useState("");
  const mutation = useMutation({ mutationFn: predictSingle });

  const handleSubmit = () => mutation.mutate({ protein: protein.trim(), smiles: smiles.trim() });

  return (
    <div className="mx-auto max-w-3xl">
      <h2 className="mb-6 text-2xl font-bold text-gray-900">Single Prediction</h2>
      <div className="rounded-xl border border-gray-200 bg-white p-6">
        <label className="mb-1 block text-sm font-medium text-gray-700">
          Protein Sequence
        </label>
        <textarea
          className="mb-4 w-full rounded-lg border border-gray-300 p-3 font-mono text-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
          rows={4}
          placeholder="Enter amino acid sequence..."
          value={protein}
          onChange={(e) => setProtein(e.target.value)}
        />
        <label className="mb-1 block text-sm font-medium text-gray-700">
          Substrate SMILES
        </label>
        <input
          className="mb-4 w-full rounded-lg border border-gray-300 p-3 font-mono text-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
          placeholder="Enter SMILES string..."
          value={smiles}
          onChange={(e) => setSmiles(e.target.value)}
        />
        <div className="flex gap-3">
          <button
            className="rounded-lg bg-indigo-600 px-5 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50"
            onClick={handleSubmit}
            disabled={mutation.isPending || !protein.trim() || !smiles.trim()}
          >
            {mutation.isPending ? "Predicting..." : "Predict"}
          </button>
          <button
            className="rounded-lg border border-gray-300 px-4 py-2 text-sm text-gray-600 hover:bg-gray-50"
            onClick={() => { setProtein(EXAMPLE_PROTEIN); setSmiles(EXAMPLE_SMILES); }}
          >
            Load Example
          </button>
        </div>
      </div>
      {mutation.error && (
        <div className="mt-4 rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">
          {mutation.error.message}
        </div>
      )}
      {mutation.data && <PredictionResult result={mutation.data} />}
    </div>
  );
}
