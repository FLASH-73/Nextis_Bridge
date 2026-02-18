import { Grid3X3, Columns2, Maximize2 } from "lucide-react";

export type LayoutMode = "grid" | "focus" | "fullscreen";

interface CameraLayoutSelectorProps {
  mode: LayoutMode;
  onChange: (mode: LayoutMode) => void;
}

const layouts: { mode: LayoutMode; icon: typeof Grid3X3; label: string }[] = [
  { mode: "grid", icon: Grid3X3, label: "Grid" },
  { mode: "focus", icon: Columns2, label: "Focus" },
  { mode: "fullscreen", icon: Maximize2, label: "Fullscreen" },
];

export default function CameraLayoutSelector({
  mode,
  onChange,
}: CameraLayoutSelectorProps) {
  return (
    <div className="flex items-center gap-0.5 bg-white/5 border border-white/10 rounded-xl p-1">
      {layouts.map(({ mode: m, icon: Icon, label }) => (
        <button
          key={m}
          onClick={() => onChange(m)}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-200 ${
            mode === m
              ? "bg-white/15 text-white shadow-sm"
              : "text-white/40 hover:text-white/60"
          }`}
          title={label}
        >
          <Icon className="w-3.5 h-3.5" />
          <span className="hidden sm:inline">{label}</span>
        </button>
      ))}
    </div>
  );
}
