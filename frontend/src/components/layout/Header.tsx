import { useQuery } from "@tanstack/react-query";
import { healthCheck } from "../../api/predict";
import { Circle } from "lucide-react";

export function Header() {
  const { data } = useQuery({ queryKey: ["health"], queryFn: healthCheck, refetchInterval: 30000 });
  const ready = data?.models_loaded;

  return (
    <header className="flex items-center justify-between border-b border-gray-200 bg-white px-6 py-3">
      <h1 className="text-base font-semibold text-gray-800">
        DLKin Enzyme Kinetics Predictor
      </h1>
      <div className="flex items-center gap-2 text-sm">
        <Circle size={10} fill={ready ? "#22c55e" : "#f59e0b"} stroke="none" />
        <span className={ready ? "text-green-700" : "text-amber-700"}>
          {ready ? "Models Ready" : "Initializing..."}
        </span>
      </div>
    </header>
  );
}
