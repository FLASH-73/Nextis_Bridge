"use client";

import { useState } from "react";
import {
  Upload,
  Pause,
  Play,
  X,
  CheckCircle,
  AlertCircle,
  Folder,
  FileText,
  Film,
  ChevronRight,
  ChevronDown,
} from "lucide-react";

export type UploadState = "idle" | "uploading" | "paused" | "complete" | "error";

export interface FileEntry {
  file: File;
  relativePath: string;
}

export interface FolderStructure {
  name: string;
  type: "folder" | "file";
  size?: number;
  children?: FolderStructure[];
}

interface UploadProgressProps {
  selectedFiles: FileEntry[];
  datasetName: string;
  folderStructure: FolderStructure | null;
  uploadState: UploadState;
  progress: number;
  currentFileIndex: number;
  errorMessage: string | null;
  isLeRobotDataset: boolean;
  onStartUpload: () => void;
  onTogglePause: () => void;
  onCancelUpload: () => void;
  onResetUploader: () => void;
  onClose?: () => void;
}

export default function UploadProgress({
  selectedFiles,
  datasetName,
  folderStructure,
  uploadState,
  progress,
  currentFileIndex,
  errorMessage,
  isLeRobotDataset,
  onStartUpload,
  onTogglePause,
  onCancelUpload,
  onResetUploader,
  onClose,
}: UploadProgressProps) {
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set(["root"]));

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
  };

  const getTotalSize = () => {
    return selectedFiles.reduce((acc, entry) => acc + entry.file.size, 0);
  };

  const toggleFolder = (path: string) => {
    setExpandedFolders((prev) => {
      const next = new Set(prev);
      if (next.has(path)) {
        next.delete(path);
      } else {
        next.add(path);
      }
      return next;
    });
  };

  // Render folder tree
  const renderTree = (node: FolderStructure, path: string = "root", depth: number = 0) => {
    const isExpanded = expandedFolders.has(path);
    const isLeRobotFolder = ["meta", "data", "videos"].includes(node.name);

    if (node.type === "file") {
      const isVideo = /\.(mp4|webm|avi|mov)$/i.test(node.name);

      return (
        <div
          key={path}
          className="flex items-center gap-2 py-1 text-xs text-neutral-600 dark:text-zinc-400"
          style={{ paddingLeft: `${depth * 16 + 8}px` }}
        >
          {isVideo ? (
            <Film className="w-3.5 h-3.5 text-purple-500 flex-shrink-0" />
          ) : (
            <FileText className="w-3.5 h-3.5 text-neutral-400 dark:text-zinc-500 flex-shrink-0" />
          )}
          <span className="truncate">{node.name}</span>
          {node.size && (
            <span className="text-neutral-400 dark:text-zinc-500 ml-auto flex-shrink-0">
              {formatFileSize(node.size)}
            </span>
          )}
        </div>
      );
    }

    return (
      <div key={path}>
        <button
          onClick={() => toggleFolder(path)}
          className="w-full flex items-center gap-2 py-1 text-xs font-medium text-neutral-700 dark:text-zinc-300 hover:bg-neutral-50 dark:hover:bg-zinc-800 rounded transition-colors"
          style={{ paddingLeft: `${depth * 16}px` }}
        >
          {isExpanded ? (
            <ChevronDown className="w-4 h-4 text-neutral-400 dark:text-zinc-500" />
          ) : (
            <ChevronRight className="w-4 h-4 text-neutral-400 dark:text-zinc-500" />
          )}
          <Folder
            className={`w-4 h-4 flex-shrink-0 ${
              isLeRobotFolder ? "text-indigo-500" : "text-amber-500"
            }`}
          />
          <span>{node.name}</span>
          {node.children && (
            <span className="text-neutral-400 dark:text-zinc-500 ml-1">({node.children.length})</span>
          )}
        </button>
        {isExpanded && node.children && (
          <div>
            {node.children.map((child) =>
              renderTree(child, `${path}/${child.name}`, depth + 1)
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="bg-white dark:bg-zinc-800 rounded-2xl border border-neutral-100 dark:border-zinc-700 shadow-sm overflow-hidden">
      {/* Header */}
      <div className="p-4 flex items-center gap-3 border-b border-neutral-100 dark:border-zinc-700">
        <div className="w-10 h-10 bg-indigo-100 dark:bg-indigo-950 rounded-xl flex items-center justify-center flex-shrink-0">
          <Folder className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <p className="font-medium text-neutral-900 dark:text-zinc-100 truncate">{datasetName}</p>
            {isLeRobotDataset && (
              <span className="px-2 py-0.5 bg-indigo-100 dark:bg-indigo-950 text-indigo-700 dark:text-indigo-300 text-[10px] font-semibold rounded-full">
                LeRobot
              </span>
            )}
          </div>
          <p className="text-sm text-neutral-500 dark:text-zinc-400">
            {selectedFiles.length} files · {formatFileSize(getTotalSize())}
          </p>
        </div>
        {uploadState === "idle" && (
          <button
            onClick={onResetUploader}
            className="p-2 hover:bg-neutral-100 dark:hover:bg-zinc-700 rounded-lg transition-colors"
          >
            <X className="w-5 h-5 text-neutral-400 dark:text-zinc-500" />
          </button>
        )}
      </div>

      {/* Folder Structure Preview */}
      {uploadState === "idle" && folderStructure && (
        <div className="max-h-48 overflow-y-auto border-b border-neutral-100 dark:border-zinc-700 p-2">
          {renderTree(folderStructure)}
        </div>
      )}

      {/* Progress */}
      {(uploadState === "uploading" || uploadState === "paused") && (
        <div className="px-4 py-3">
          <div className="flex items-center justify-between text-xs text-neutral-500 dark:text-zinc-400 mb-2">
            <span>
              {uploadState === "paused" ? "Paused" : "Uploading"} ({currentFileIndex + 1}/{selectedFiles.length})
            </span>
            <span>{progress}%</span>
          </div>
          <div className="h-2 bg-neutral-100 dark:bg-zinc-700 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-300 ease-out ${
                uploadState === "paused" ? "bg-amber-500" : "bg-indigo-600"
              }`}
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="text-xs text-neutral-400 dark:text-zinc-500 mt-2 truncate">
            {selectedFiles[currentFileIndex]?.relativePath}
          </p>
        </div>
      )}

      {/* Complete */}
      {uploadState === "complete" && (
        <div className="px-4 py-3">
          <div className="flex items-center gap-2 text-green-600">
            <CheckCircle className="w-5 h-5" />
            <span className="text-sm font-medium">Dataset uploaded successfully</span>
          </div>
        </div>
      )}

      {/* Error */}
      {uploadState === "error" && (
        <div className="px-4 py-3">
          <div className="flex items-center gap-2 text-red-600">
            <AlertCircle className="w-5 h-5" />
            <span className="text-sm font-medium">{errorMessage || "Upload failed"}</span>
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="px-4 py-3 flex gap-2 bg-neutral-50 dark:bg-zinc-900">
        {uploadState === "idle" && (
          <button
            onClick={onStartUpload}
            className="flex-1 py-2.5 bg-black dark:bg-white hover:bg-neutral-800 dark:hover:bg-zinc-200 text-white dark:text-black font-medium rounded-xl transition-colors flex items-center justify-center gap-2"
          >
            <Upload className="w-4 h-4" />
            Upload Dataset
          </button>
        )}

        {(uploadState === "uploading" || uploadState === "paused") && (
          <>
            <button
              onClick={onTogglePause}
              className={`flex-1 py-2.5 font-medium rounded-xl transition-colors flex items-center justify-center gap-2 ${
                uploadState === "paused"
                  ? "bg-black dark:bg-white text-white dark:text-black hover:bg-neutral-800 dark:hover:bg-zinc-200"
                  : "bg-amber-500 text-white hover:bg-amber-600"
              }`}
            >
              {uploadState === "paused" ? (
                <>
                  <Play className="w-4 h-4" />
                  Resume
                </>
              ) : (
                <>
                  <Pause className="w-4 h-4" />
                  Pause
                </>
              )}
            </button>
            <button
              onClick={onCancelUpload}
              className="px-4 py-2.5 bg-neutral-200 dark:bg-zinc-700 hover:bg-neutral-300 dark:hover:bg-zinc-600 text-neutral-700 dark:text-zinc-300 font-medium rounded-xl transition-colors"
            >
              Cancel
            </button>
          </>
        )}

        {(uploadState === "complete" || uploadState === "error") && (
          <button
            onClick={uploadState === "complete" ? (onClose || onResetUploader) : onResetUploader}
            className="flex-1 py-2.5 bg-black dark:bg-white hover:bg-neutral-800 dark:hover:bg-zinc-200 text-white dark:text-black font-medium rounded-xl transition-colors"
          >
            {uploadState === "complete" ? "Done" : "Try Again"}
          </button>
        )}
      </div>
    </div>
  );
}
