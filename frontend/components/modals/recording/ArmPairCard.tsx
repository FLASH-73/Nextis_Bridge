import { ArrowLeftRight } from "lucide-react";
import type { Arm, Pairing } from "../../../lib/api/types";

interface ArmPairCardProps {
  pairing: Pairing;
  leader: Arm | undefined;
  follower: Arm | undefined;
  selected: boolean;
  onToggle: () => void;
}

function ConnectionDot({ status }: { status: string }) {
  const color =
    status === "connected"
      ? "bg-emerald-500"
      : status === "error"
        ? "bg-red-500"
        : "bg-neutral-300 dark:bg-zinc-600";
  return <div className={`w-2 h-2 rounded-full flex-shrink-0 ${color}`} />;
}

export default function ArmPairCard({
  pairing,
  leader,
  follower,
  selected,
  onToggle,
}: ArmPairCardProps) {
  const pairingLabel =
    pairing.name || `${pairing.leader_id} → ${pairing.follower_id}`;

  return (
    <div
      className={`bg-white dark:bg-zinc-800 rounded-lg border p-3 transition-all cursor-pointer ${
        selected
          ? "border-black dark:border-white ring-1 ring-black/10 dark:ring-white/10"
          : "border-neutral-100 dark:border-zinc-700 opacity-60 hover:opacity-80"
      }`}
      onClick={onToggle}
    >
      {/* Header: pairing name + record checkbox */}
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm font-semibold text-neutral-800 dark:text-zinc-200 truncate">
          {pairingLabel}
        </span>
        <label
          className="flex items-center gap-1.5 cursor-pointer"
          onClick={(e) => e.stopPropagation()}
        >
          <input
            type="checkbox"
            checked={selected}
            onChange={onToggle}
            className="accent-black dark:accent-white w-3.5 h-3.5"
          />
          <span className="text-xs font-medium text-neutral-500 dark:text-zinc-400">
            Record
          </span>
        </label>
      </div>

      {/* Follower row */}
      {follower && (
        <div className="flex items-center justify-between py-1">
          <div className="flex items-center gap-2">
            <span className="text-xs font-medium text-neutral-600 dark:text-zinc-300">
              Follower:
            </span>
            <span className="text-xs text-neutral-800 dark:text-zinc-200">
              {follower.name || follower.id}
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            <ConnectionDot status={follower.status} />
            <span className="text-[10px] text-neutral-400 dark:text-zinc-500 capitalize">
              {follower.status}
            </span>
          </div>
        </div>
      )}

      {/* Leader row (muted, not recorded) */}
      {leader && (
        <div className="flex items-center justify-between py-1">
          <div className="flex items-center gap-2">
            <ArrowLeftRight className="w-3 h-3 text-neutral-300 dark:text-zinc-600" />
            <span className="text-xs text-neutral-400 dark:text-zinc-500">
              Leader: {leader.name || leader.id}
            </span>
          </div>
          <span className="text-[10px] text-neutral-300 dark:text-zinc-600 italic">
            not recorded
          </span>
        </div>
      )}
    </div>
  );
}
