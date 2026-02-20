import { Database } from "lucide-react";
import SafetyIndicators from "./SafetyIndicators";

interface DatasetConfigProps {
  datasetConfig: { repo_id: string; task: string };
  setDatasetConfig: (config: { repo_id: string; task: string }) => void;
}

export default function DatasetConfig({
  datasetConfig,
  setDatasetConfig,
}: DatasetConfigProps) {
  return (
    <div className="bg-neutral-50 dark:bg-zinc-800/50 rounded-xl border border-neutral-100 dark:border-zinc-700 p-4 space-y-4">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-500 dark:text-zinc-400 flex items-center gap-1.5">
        <Database className="w-3.5 h-3.5" />
        Dataset
      </h3>

      <div className="space-y-3">
        <div>
          <label className="block text-xs font-medium text-neutral-700 dark:text-zinc-300 mb-1">
            Name
          </label>
          <input
            type="text"
            value={datasetConfig.repo_id}
            onChange={(e) =>
              setDatasetConfig({ ...datasetConfig, repo_id: e.target.value })
            }
            className="w-full px-3 py-2 bg-white dark:bg-zinc-800 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm focus:ring-2 focus:ring-black/20 dark:focus:ring-white/20 focus:border-neutral-400 dark:focus:border-zinc-500 outline-none transition-all placeholder-neutral-300 dark:placeholder-zinc-500 text-neutral-900 dark:text-zinc-100"
            placeholder="user/dataset-name"
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-neutral-700 dark:text-zinc-300 mb-1">
            Task Description
          </label>
          <input
            type="text"
            value={datasetConfig.task}
            onChange={(e) =>
              setDatasetConfig({ ...datasetConfig, task: e.target.value })
            }
            className="w-full px-3 py-2 bg-white dark:bg-zinc-800 border border-neutral-200 dark:border-zinc-700 rounded-lg text-sm focus:ring-2 focus:ring-black/20 dark:focus:ring-white/20 focus:border-neutral-400 dark:focus:border-zinc-500 outline-none transition-all placeholder-neutral-300 dark:placeholder-zinc-500 text-neutral-900 dark:text-zinc-100"
            placeholder="e.g. Pick up the red cube"
          />
        </div>
      </div>

      <div className="pt-2 border-t border-neutral-100 dark:border-zinc-700">
        <SafetyIndicators />
      </div>
    </div>
  );
}
