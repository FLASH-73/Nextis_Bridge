import { Shield } from "lucide-react";

const indicators = [
  { label: "Torque monitoring", active: true },
  { label: "Joint limits enforced", active: true },
  { label: "E-STOP ready (ESC)", active: true },
];

export default function SafetyIndicators() {
  return (
    <div>
      <h4 className="text-xs font-semibold uppercase tracking-wider text-neutral-500 dark:text-zinc-400 mb-2 flex items-center gap-1.5">
        <Shield className="w-3.5 h-3.5" />
        Safety
      </h4>
      <div className="flex flex-wrap gap-3">
        {indicators.map((ind) => (
          <div key={ind.label} className="flex items-center gap-2">
            <div
              className={`w-2 h-2 rounded-full ${
                ind.active ? "bg-emerald-500" : "bg-neutral-300 dark:bg-zinc-600"
              }`}
            />
            <span className="text-xs text-neutral-600 dark:text-zinc-400">
              {ind.label}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
