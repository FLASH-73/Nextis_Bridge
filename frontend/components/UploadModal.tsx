"use client";

import { X } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import FileUploader from "./modals/upload";
import { type Dataset } from "../lib/supabase";

interface UploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onUploadComplete?: (dataset: Dataset) => void;
}

export default function UploadModal({ isOpen, onClose, onUploadComplete }: UploadModalProps) {
  const handleUploadComplete = (dataset: Dataset) => {
    onUploadComplete?.(dataset);
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/20 backdrop-blur-sm z-50"
          />

          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 10 }}
            transition={{ duration: 0.2, ease: [0.25, 1, 0.5, 1] }}
            className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-50 w-full max-w-xl"
          >
            <div className="bg-white/90 dark:bg-zinc-900/90 backdrop-blur-xl rounded-3xl shadow-2xl border border-white/50 dark:border-zinc-700/50 overflow-hidden">
              {/* Header */}
              <div className="flex items-center justify-between px-6 py-4 border-b border-neutral-100 dark:border-zinc-800">
                <div>
                  <h2 className="text-lg font-semibold text-neutral-900 dark:text-zinc-100">New Upload</h2>
                  <p className="text-sm text-neutral-500 dark:text-zinc-400">Upload robot logs to your library</p>
                </div>
                <button
                  onClick={onClose}
                  className="p-2 hover:bg-neutral-100 dark:hover:bg-zinc-800 rounded-xl transition-colors"
                >
                  <X className="w-5 h-5 text-neutral-400 dark:text-zinc-500" />
                </button>
              </div>

              {/* Content */}
              <div className="p-6">
                <FileUploader
                  onUploadComplete={handleUploadComplete}
                  onClose={onClose}
                />
              </div>

              {/* Footer */}
              <div className="px-6 py-4 bg-neutral-50/50 dark:bg-zinc-800/50 border-t border-neutral-100 dark:border-zinc-800">
                <p className="text-xs text-neutral-400 dark:text-zinc-500 text-center">
                  Large files are uploaded in chunks and can be paused/resumed
                </p>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
