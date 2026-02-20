import { Wrench } from "lucide-react";
import type { Tool, Trigger, ListenerStatus } from "../../../lib/api/types";

interface ToolsListProps {
  tools: Tool[];
  triggers: Trigger[];
  listenerStatus: ListenerStatus | null;
}

function statusDot(connected: boolean) {
  return (
    <div
      className={`w-2 h-2 rounded-full flex-shrink-0 ${
        connected ? "bg-emerald-500" : "bg-neutral-300 dark:bg-zinc-600"
      }`}
    />
  );
}

export default function ToolsList({
  tools,
  triggers,
  listenerStatus,
}: ToolsListProps) {
  if (tools.length === 0 && triggers.length === 0) return null;

  return (
    <div className="bg-neutral-50 dark:bg-zinc-800/50 rounded-xl border border-neutral-100 dark:border-zinc-700 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-500 dark:text-zinc-400 flex items-center gap-1.5">
          <Wrench className="w-3.5 h-3.5" />
          Tools
        </h3>
        <span className="text-[10px] font-medium text-neutral-400 dark:text-zinc-500 bg-neutral-100 dark:bg-zinc-700/50 px-2 py-0.5 rounded-full">
          auto-included
        </span>
      </div>

      <div className="space-y-2">
        {tools.map((tool) => {
          const isActive = listenerStatus?.tool_states?.[tool.id] ?? false;
          const speed =
            typeof tool.config?.speed === "number" ? tool.config.speed : null;

          return (
            <div
              key={tool.id}
              className="flex items-center justify-between py-1.5"
            >
              <div className="flex items-center gap-2">
                {statusDot(tool.status === "connected")}
                <span className="text-sm text-neutral-700 dark:text-zinc-300">
                  {tool.name}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-neutral-400 dark:text-zinc-500">
                  {tool.status === "connected"
                    ? isActive
                      ? speed !== null
                        ? `${speed} RPM`
                        : "Active"
                      : "Idle"
                    : "Disconnected"}
                </span>
              </div>
            </div>
          );
        })}

        {triggers.map((trigger) => {
          const isPressed =
            listenerStatus?.trigger_states?.[trigger.id] ?? false;
          const isListening = listenerStatus?.running ?? false;

          return (
            <div
              key={trigger.id}
              className="flex items-center justify-between py-1.5"
            >
              <div className="flex items-center gap-2">
                {statusDot(isListening)}
                <span className="text-sm text-neutral-700 dark:text-zinc-300">
                  {trigger.name}
                </span>
              </div>
              <span className="text-xs text-neutral-400 dark:text-zinc-500">
                {isListening
                  ? isPressed
                    ? "Pressed"
                    : "Released"
                  : "Not listening"}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
