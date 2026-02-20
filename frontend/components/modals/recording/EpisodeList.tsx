import type { EpisodeRecord } from "../../../lib/api/types";

interface EpisodeListProps {
  episodes: EpisodeRecord[];
}

export default function EpisodeList({ episodes }: EpisodeListProps) {
  const visible = episodes.slice(-10).reverse();

  if (visible.length === 0) {
    return (
      <p className="text-xs text-zinc-500 text-center py-4">
        No episodes yet.
      </p>
    );
  }

  return (
    <div className="space-y-1">
      {visible.map((ep) => (
        <div
          key={ep.index}
          className="flex items-center justify-between px-2 py-1.5 rounded-lg bg-zinc-800/50"
        >
          <span className="text-xs text-zinc-300 font-mono">#{ep.index}</span>
          <span className="text-xs text-zinc-400 font-mono">
            {ep.duration.toFixed(1)}s
          </span>
          {ep.status === "saved" ? (
            <span className="text-xs text-emerald-400">&#10003; saved</span>
          ) : (
            <span className="text-xs text-red-400">&#10007; discarded</span>
          )}
        </div>
      ))}
    </div>
  );
}
