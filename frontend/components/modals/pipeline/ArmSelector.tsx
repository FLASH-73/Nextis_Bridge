import { useEffect, useMemo } from "react";
import useSWR from "swr";
import { armsApi } from "../../../lib/api";
import type { Pairing } from "../../../lib/api/types";

interface ArmSelectorProps {
  selectedArms: string[];
  onChange: (arms: string[]) => void;
}

export default function ArmSelector({ selectedArms, onChange }: ArmSelectorProps) {
  const { data: pairingsData } = useSWR<{ pairings: Pairing[] }>(
    "/arms/pairings",
    () => armsApi.listPairings()
  );

  const pairings = useMemo(() => pairingsData?.pairings ?? [], [pairingsData]);

  // Derive selected pairing names from active arm IDs
  const selectedPairingNames = useMemo(() => {
    return pairings
      .filter(
        (p) =>
          selectedArms.includes(p.leader_id) && selectedArms.includes(p.follower_id)
      )
      .map((p) => p.name);
  }, [pairings, selectedArms]);

  // Auto-select all on first load
  useEffect(() => {
    if (pairings.length > 0 && selectedArms.length === 0) {
      const allArms = pairings.flatMap((p) => [p.leader_id, p.follower_id]);
      onChange(allArms);
    }
  }, [pairings, selectedArms.length, onChange]);

  const togglePairing = (pairing: Pairing) => {
    const isSelected = selectedPairingNames.includes(pairing.name);
    if (isSelected) {
      onChange(
        selectedArms.filter(
          (id) => id !== pairing.leader_id && id !== pairing.follower_id
        )
      );
    } else {
      onChange([...selectedArms, pairing.leader_id, pairing.follower_id]);
    }
  };

  return (
    <div className="bg-neutral-50 dark:bg-zinc-800/50 rounded-xl border border-neutral-100 dark:border-zinc-700 p-4">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-500 dark:text-zinc-400 mb-3">
        Arm Pairs
      </h3>
      {pairings.length === 0 ? (
        <p className="text-xs text-neutral-400 dark:text-zinc-500 text-center py-3">
          No pairings configured
        </p>
      ) : (
        <div className="space-y-2">
          {pairings.map((p) => {
            const selected = selectedPairingNames.includes(p.name);
            return (
              <button
                key={p.name}
                onClick={() => togglePairing(p)}
                className={`w-full p-3 rounded-xl border-2 transition-all text-left ${
                  selected
                    ? "border-black dark:border-white bg-neutral-100 dark:bg-zinc-700"
                    : "border-neutral-200 dark:border-zinc-700 hover:border-neutral-400 dark:hover:border-zinc-500"
                }`}
              >
                <span className="text-sm font-medium text-neutral-900 dark:text-zinc-100">
                  {p.name}
                </span>
                <span className="block text-xs text-neutral-400 dark:text-zinc-500 mt-0.5">
                  {p.leader_id} &rarr; {p.follower_id}
                </span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
