"use client";

import { useState } from "react";
import { Plus, Terminal, Minimize2, Maximize2, X, Play } from "lucide-react";
import { useDraggable } from "../hooks/useDraggable";
import { API_BASE } from "../lib/api";

interface Message {
  role: "user" | "assistant";
  content: string;
}

interface ChatWindowProps {
  isChatOpen: boolean;
  setIsChatOpen: (open: boolean) => void;
  maximizedWindow: string | null;
  setMaximizedWindow: (window: string | null) => void;
  onPlanReceived: (plan: any) => void;
}

export default function ChatWindow({
  isChatOpen,
  setIsChatOpen,
  maximizedWindow,
  setMaximizedWindow,
  onPlanReceived,
}: ChatWindowProps) {
  const chatDrag = useDraggable({ x: 20, y: 80 });
  const [isChatMinimized, setIsChatMinimized] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [currentPlan, setCurrentPlan] = useState<any>(null);
  const [isThinking, setIsThinking] = useState(false);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMsg = { role: "user" as const, content: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsThinking(true);

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: input,
          messages: [...messages, userMsg],
        }),
      });

      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.reply },
      ]);

      if (data.plan) {
        setCurrentPlan(data.plan);
        onPlanReceived(data.plan);
      }
      setIsThinking(false);
    } catch (error) {
      console.error(error);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Error connecting to server." },
      ]);
      setIsThinking(false);
    }
  };

  const startNewChat = () => {
    setMessages([]);
    setCurrentPlan(null);
    onPlanReceived(null);
    setInput("");
  };

  return (
    <>
      {/* FLOATING CHAT WINDOW (Draggable) */}
      <div
        ref={maximizedWindow === "chat" ? null : chatDrag.ref}
        style={{
          width:
            maximizedWindow === "chat"
              ? "calc(100vw - 40px)"
              : isChatMinimized
                ? "auto"
                : "380px",
          height:
            maximizedWindow === "chat"
              ? "calc(100vh - 120px)"
              : isChatMinimized
                ? "auto"
                : "550px",
          top: maximizedWindow === "chat" ? "100px" : "auto",
          left: maximizedWindow === "chat" ? "20px" : "auto",
          opacity: isChatOpen ? 1 : 0,
          pointerEvents: isChatOpen ? "auto" : "none",
          zIndex: maximizedWindow === "chat" ? 100 : 40,
        }}
        className={`absolute flex flex-col shadow-2xl rounded-2xl overflow-hidden glass border border-white/50 dark:border-zinc-700/50 backdrop-blur-3xl transition-all duration-300 ease-[cubic-bezier(0.25,1,0.5,1)] ${!isChatOpen && "scale-90"}`}
      >
        {/* Window Handle */}
        <div
          onMouseDown={
            maximizedWindow !== "chat" ? chatDrag.handleMouseDown : undefined
          }
          onDoubleClick={() =>
            setMaximizedWindow(maximizedWindow === "chat" ? null : "chat")
          }
          className={`h-10 bg-white/40 dark:bg-zinc-800/40 border-b border-black/5 dark:border-white/5 flex items-center justify-between px-4 select-none ${maximizedWindow !== "chat" ? "cursor-grab active:cursor-grabbing" : "cursor-default"}`}
        >
          <div className="flex items-center gap-2">
            <div className="flex gap-1.5 mr-2">
              <button
                onClick={() => {
                  setIsChatOpen(false);
                  if (maximizedWindow === "chat") setMaximizedWindow(null);
                }}
                className="w-3 h-3 rounded-full bg-[#FF5F57] border border-black/10 hover:brightness-90 transition-all flex items-center justify-center group"
              >
                <X className="w-2 h-2 text-black/50 opacity-0 group-hover:opacity-100" />
              </button>
              <button
                onClick={() => {
                  setIsChatMinimized(!isChatMinimized);
                  if (maximizedWindow === "chat") setMaximizedWindow(null);
                }}
                className="w-3 h-3 rounded-full bg-[#FEBC2E] border border-black/10 hover:brightness-90 transition-all flex items-center justify-center group"
              >
                <Minimize2 className="w-2 h-2 text-black/50 opacity-0 group-hover:opacity-100" />
              </button>
              <button
                onClick={() => {
                  setMaximizedWindow(
                    maximizedWindow === "chat" ? null : "chat"
                  );
                  if (isChatMinimized) setIsChatMinimized(false);
                }}
                className="w-3 h-3 rounded-full bg-[#28C840] border border-black/10 hover:brightness-90 transition-all flex items-center justify-center group"
              >
                <Maximize2 className="w-2 h-2 text-black/50 opacity-0 group-hover:opacity-100" />
              </button>
            </div>
            <span className="ml-2 text-xs font-semibold text-neutral-600 dark:text-zinc-400 tracking-wide">
              Assistant
            </span>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={startNewChat}
              title="New Chat"
              className="p-1 hover:bg-black/5 dark:hover:bg-white/5 rounded-md transition-colors text-neutral-400 dark:text-zinc-500 hover:text-black dark:hover:text-white"
            >
              <Plus className="w-4 h-4" />
            </button>
          </div>
        </div>

        {!isChatMinimized && (
          <>
            <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-white/30 dark:bg-zinc-900/30 custom-scrollbar">
              {messages.length === 0 && (
                <div className="h-full flex flex-col items-center justify-center opacity-40 gap-3 pb-10">
                  <div className="w-12 h-12 bg-black/5 dark:bg-white/5 rounded-full flex items-center justify-center">
                    <Terminal className="w-6 h-6 text-black dark:text-white" />
                  </div>
                  <p className="text-sm font-medium text-black dark:text-white">
                    How can I help you?
                  </p>
                </div>
              )}
              {messages.map((msg, i) => (
                <div
                  key={i}
                  className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`max-w-[85%] py-2.5 px-4 text-sm rounded-2xl shadow-sm backdrop-blur-sm leading-relaxed ${
                      msg.role === "user"
                        ? "bg-black dark:bg-white text-white dark:text-black"
                        : "bg-white/80 dark:bg-zinc-800/80 text-neutral-800 dark:text-zinc-200 border border-white/50 dark:border-zinc-700/50"
                    }`}
                  >
                    {msg.content}
                  </div>
                </div>
              ))}
              {isThinking && (
                <div className="flex justify-start animate-pulse">
                  <div className="bg-white/50 dark:bg-zinc-800/50 px-4 py-2 rounded-full text-xs text-neutral-500 dark:text-zinc-400">
                    Thinking...
                  </div>
                </div>
              )}
            </div>

            <div className="p-3 bg-white/60 dark:bg-zinc-900/60 border-t border-white/50 dark:border-zinc-700/50">
              <div className="relative">
                <input
                  className="w-full bg-white/50 dark:bg-zinc-800/50 hover:bg-white dark:hover:bg-zinc-800 focus:bg-white dark:focus:bg-zinc-800 border border-transparent focus:border-black/5 dark:focus:border-white/5 rounded-xl py-3 pl-4 pr-10 text-sm text-neutral-900 dark:text-zinc-100 outline-none transition-all placeholder:text-neutral-400 dark:placeholder:text-zinc-500 shadow-sm"
                  placeholder="Type a message..."
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleSend()}
                />
                <button
                  onClick={handleSend}
                  disabled={!input.trim()}
                  className="absolute right-2 top-1/2 -translate-y-1/2 p-1.5 bg-black dark:bg-white text-white dark:text-black rounded-lg hover:scale-110 active:scale-95 transition-all shadow-md disabled:opacity-20 disabled:hover:scale-100"
                >
                  <div className="w-2.5 h-2.5 bg-white dark:bg-black rounded-sm" />
                </button>
              </div>
            </div>
          </>
        )}
      </div>

      {/* EXECUTE BUTTON (Floating Bottom Right) */}
      {currentPlan && (
        <div className="absolute bottom-10 right-10 z-50 animate-in slide-in-from-bottom-10 fade-in duration-500">
          <button
            onClick={async () => {
              await fetch(`${API_BASE}/execute`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ plan: currentPlan }),
              });
            }}
            className="px-8 py-4 bg-black/90 dark:bg-white/90 hover:bg-black dark:hover:bg-white text-white dark:text-black rounded-full shadow-2xl backdrop-blur-xl border border-white/10 dark:border-black/10 flex items-center gap-3 font-semibold tracking-wide hover:scale-105 transition-all active:scale-95 group"
          >
            <Play className="w-5 h-5 fill-white dark:fill-black group-hover:scale-110 transition-transform" />
            START SEQUENCE
          </button>
        </div>
      )}
    </>
  );
}
