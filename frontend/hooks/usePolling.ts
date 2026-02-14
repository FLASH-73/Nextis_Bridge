import { useEffect, useRef } from "react";

/**
 * Generic polling hook. Replaces repeated setInterval + useRef + cleanup patterns.
 * Calls `callback` immediately on enable, then every `intervalMs` milliseconds.
 * Automatically cleans up on unmount or when `enabled` becomes false.
 */
export function usePolling(
  callback: () => void | Promise<void>,
  intervalMs: number,
  enabled: boolean
) {
  const savedCallback = useRef(callback);

  // Update ref on every render so the interval always calls the latest callback
  useEffect(() => {
    savedCallback.current = callback;
  });

  useEffect(() => {
    if (!enabled) return;

    // Immediate first call
    savedCallback.current();

    const id = setInterval(() => {
      savedCallback.current();
    }, intervalMs);

    return () => clearInterval(id);
  }, [intervalMs, enabled]);
}
