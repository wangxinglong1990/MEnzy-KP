import { Link } from "react-router-dom";
import { TestTube, FlaskConical, GitGraph, Dna } from "lucide-react";

const CARDS = [
  { to: "/predict", icon: TestTube, title: "Single Prediction", desc: "Predict Km & kcat from one enzyme-substrate pair" },
  { to: "/batch", icon: FlaskConical, title: "Batch Prediction", desc: "Upload CSV and predict hundreds of pairs at once" },
  { to: "/clustering", icon: GitGraph, title: "Clustering Analysis", desc: "Cluster top candidates and visualize with UMAP" },
  { to: "/docking", icon: Dna, title: "Molecular Docking", desc: "Validate binding with RosettaLigand" },
];

export function HomePage() {
  return (
    <div className="mx-auto max-w-4xl">
      <h2 className="mb-2 text-3xl font-bold text-gray-900">Welcome to DLKin</h2>
      <p className="mb-8 text-gray-500">
        Deep Learning Kinetics Predictor — joint Km &amp; kcat prediction for enzyme-substrate pairs
      </p>
      <div className="grid grid-cols-2 gap-4">
        {CARDS.map(({ to, icon: Icon, title, desc }) => (
          <Link
            key={to}
            to={to}
            className="rounded-xl border border-gray-200 bg-white p-6 transition-shadow hover:shadow-md"
          >
            <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-lg bg-indigo-50 text-indigo-600">
              <Icon size={22} />
            </div>
            <h3 className="mb-1 font-semibold text-gray-900">{title}</h3>
            <p className="text-sm text-gray-500">{desc}</p>
          </Link>
        ))}
      </div>
    </div>
  );
}
