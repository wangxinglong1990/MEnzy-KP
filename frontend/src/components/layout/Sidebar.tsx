import { NavLink } from "react-router-dom";
import { FlaskConical, TestTube, ArrowUpDown, GitGraph, Dna, Cpu, Home } from "lucide-react";

const NAV = [
  { to: "/", icon: Home, label: "Home" },
  { to: "/predict", icon: TestTube, label: "Single Predict" },
  { to: "/batch", icon: FlaskConical, label: "Batch Predict" },
  { to: "/ranking", icon: ArrowUpDown, label: "Ranking" },
  { to: "/clustering", icon: GitGraph, label: "Clustering" },
  { to: "/docking", icon: Dna, label: "Docking" },
  { to: "/training", icon: Cpu, label: "Training" },
];

export function Sidebar() {
  return (
    <aside className="flex w-56 flex-col border-r border-gray-200 bg-white p-4">
      <div className="mb-6 px-2 text-lg font-bold text-indigo-600">DLKin</div>
      <nav className="flex flex-col gap-1">
        {NAV.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            className={({ isActive }) =>
              `flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                isActive
                  ? "bg-indigo-50 text-indigo-700"
                  : "text-gray-600 hover:bg-gray-100 hover:text-gray-900"
              }`
            }
          >
            <Icon size={18} />
            {label}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}
