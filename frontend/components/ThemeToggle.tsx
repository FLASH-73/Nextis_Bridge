"use client";

import { Sun, Moon, Monitor } from "lucide-react";
import { useTheme } from "../lib/ThemeContext";

export function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  const cycleTheme = () => {
    setTheme(theme === "light" ? "dark" : theme === "dark" ? "system" : "light");
  };

  return (
    <button
      onClick={cycleTheme}
      className="p-2 rounded-lg hover:bg-neutral-100 dark:hover:bg-zinc-800 transition-colors"
      title={`Theme: ${theme}`}
    >
      {theme === "light" && <Sun className="w-4 h-4 text-amber-500" />}
      {theme === "dark" && <Moon className="w-4 h-4 text-blue-400" />}
      {theme === "system" && <Monitor className="w-4 h-4 text-neutral-500 dark:text-zinc-400" />}
    </button>
  );
}
