import { Link } from "react-router-dom";

export function BatchPage() {
  return (
    <div className="mx-auto max-w-3xl">
      <h2 className="mb-6 text-2xl font-bold text-gray-900">Batch Prediction</h2>
      <div className="rounded-xl border border-gray-200 bg-white p-8 text-center">
        <p className="mb-4 text-gray-500">CSV batch prediction is coming soon.</p>
        <Link to="/predict" className="text-sm font-medium text-indigo-600 hover:underline">
          Try Single Prediction
        </Link>
      </div>
    </div>
  );
}
