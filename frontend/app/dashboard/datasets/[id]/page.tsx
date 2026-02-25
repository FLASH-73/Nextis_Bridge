"use client";

import { useState, useEffect } from "react";
import { useParams, useRouter } from "next/navigation";
import { motion } from "framer-motion";
import {
  ArrowLeft,
  Database,
  HardDrive,
  Clock,
  Folder,
  FileText,
  Film,
  FileJson,
  ChevronRight,
  ChevronDown,
  Download,
  Trash2,
  Loader2,
  ExternalLink,
  Copy,
  Check,
  Bot,
} from "lucide-react";
import { useAuth } from "../../../../lib/AuthContext";
import { formatRelativeTime, formatBytes, formatNumber, getStatusColor } from "../../../../lib/utils";
import {
  getDatasetById,
  getDatasetFiles,
  deleteDataset,
  CloudDataset,
  StorageFile,
} from "../../actions";

export default function DatasetDetailPage() {
  const params = useParams();
  const router = useRouter();
  const { user } = useAuth();
  const datasetId = params.id as string;

  const [dataset, setDataset] = useState<CloudDataset | null>(null);
  const [files, setFiles] = useState<StorageFile[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set());
  const [deleting, setDeleting] = useState(false);
  const [copiedPath, setCopiedPath] = useState<string | null>(null);

  useEffect(() => {
    if (user && datasetId) {
      fetchData();
    }
  }, [user, datasetId]);

  const fetchData = async () => {
    if (!user) return;
    setLoading(true);
    try {
      const [datasetData, filesData] = await Promise.all([
        getDatasetById(datasetId),
        getDatasetFiles(user.id, datasetId),
      ]);
      setDataset(datasetData);
      setFiles(filesData);

      // Auto-expand root folders
      const rootFolderPaths = filesData
        .filter((f) => f.isFolder)
        .map((f) => f.path);
      setExpandedFolders(new Set(rootFolderPaths));
    } catch (err) {
      console.error("Failed to fetch dataset:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async () => {
    if (!user || !dataset) return;
    if (!confirm("Are you sure you want to delete this dataset? This action cannot be undone.")) {
      return;
    }

    setDeleting(true);
    const result = await deleteDataset(datasetId, user.id);
    if (result.success) {
      router.push("/dashboard");
    } else {
      alert(`Failed to delete: ${result.error}`);
      setDeleting(false);
    }
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

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedPath(text);
    setTimeout(() => setCopiedPath(null), 2000);
  };

  const getFileIcon = (fileName: string, isFolder: boolean) => {
    if (isFolder) return <Folder className="w-4 h-4 text-amber-500" />;

    const ext = fileName.split(".").pop()?.toLowerCase();
    switch (ext) {
      case "mp4":
      case "webm":
      case "avi":
      case "mov":
        return <Film className="w-4 h-4 text-purple-500" />;
      case "json":
      case "jsonl":
        return <FileJson className="w-4 h-4 text-green-500" />;
      case "parquet":
        return <Database className="w-4 h-4 text-blue-500" />;
      default:
        return <FileText className="w-4 h-4 text-neutral-400" />;
    }
  };

  const renderFileTree = (items: StorageFile[], depth: number = 0) => {
    return items.map((item) => {
      const isExpanded = expandedFolders.has(item.path);

      return (
        <div key={item.path}>
          <div
            className={`flex items-center gap-2 py-2 px-3 hover:bg-neutral-50 rounded-lg transition-colors cursor-pointer group ${
              depth > 0 ? "ml-4" : ""
            }`}
            style={{ paddingLeft: `${depth * 16 + 12}px` }}
            onClick={() => item.isFolder && toggleFolder(item.path)}
          >
            {item.isFolder && (
              <button className="p-0.5">
                {isExpanded ? (
                  <ChevronDown className="w-3.5 h-3.5 text-neutral-400" />
                ) : (
                  <ChevronRight className="w-3.5 h-3.5 text-neutral-400" />
                )}
              </button>
            )}
            {!item.isFolder && <span className="w-4" />}

            {getFileIcon(item.name, item.isFolder)}

            <span className="flex-1 text-sm text-neutral-700 truncate">{item.name}</span>

            {!item.isFolder && item.size > 0 && (
              <span className="text-xs text-neutral-400 mr-2">{formatBytes(item.size)}</span>
            )}

            {item.isFolder && item.children && (
              <span className="text-xs text-neutral-400 mr-2">{item.children.length} items</span>
            )}

            <button
              onClick={(e) => {
                e.stopPropagation();
                copyToClipboard(item.path);
              }}
              className="p-1 hover:bg-neutral-200 rounded opacity-0 group-hover:opacity-100 transition-opacity"
              title="Copy path"
            >
              {copiedPath === item.path ? (
                <Check className="w-3.5 h-3.5 text-green-500" />
              ) : (
                <Copy className="w-3.5 h-3.5 text-neutral-400" />
              )}
            </button>
          </div>

          {item.isFolder && isExpanded && item.children && (
            <div>{renderFileTree(item.children, depth + 1)}</div>
          )}
        </div>
      );
    });
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-neutral-50 to-neutral-100 flex items-center justify-center">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex flex-col items-center gap-4"
        >
          <Loader2 className="w-8 h-8 text-blue-500 animate-spin" />
          <p className="text-neutral-500">Loading dataset...</p>
        </motion.div>
      </div>
    );
  }

  if (!dataset) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-neutral-50 to-neutral-100 flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-xl font-semibold text-neutral-900 mb-2">Dataset not found</h2>
          <p className="text-neutral-500 mb-4">
            The dataset you're looking for doesn't exist or you don't have access to it.
          </p>
          <button
            onClick={() => router.push("/dashboard")}
            className="px-4 py-2 bg-black text-white rounded-xl font-medium hover:bg-neutral-800 transition-colors"
          >
            Back to Dashboard
          </button>
        </div>
      </div>
    );
  }

  const statusColors = getStatusColor(dataset.status);

  return (
    <div className="min-h-screen bg-gradient-to-br from-neutral-50 to-neutral-100">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-xl border-b border-neutral-200/50 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center gap-4">
            <button
              onClick={() => router.push("/dashboard")}
              className="p-2 hover:bg-neutral-100 rounded-xl transition-colors"
            >
              <ArrowLeft className="w-5 h-5 text-neutral-600" />
            </button>
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
                <Bot className="w-5 h-5 text-white" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h1 className="text-xl font-bold text-neutral-900">{dataset.name}</h1>
                  <span
                    className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium ${statusColors.bg} ${statusColors.text}`}
                  >
                    <span className={`w-1.5 h-1.5 rounded-full ${statusColors.dot}`} />
                    {dataset.status.charAt(0).toUpperCase() + dataset.status.slice(1)}
                  </span>
                </div>
                <p className="text-sm text-neutral-500">Dataset Details</p>
              </div>
            </div>

            <div className="ml-auto flex items-center gap-2">
              <button
                onClick={handleDelete}
                disabled={deleting}
                className="flex items-center gap-2 px-4 py-2 border border-red-200 text-red-600 rounded-xl font-medium hover:bg-red-50 transition-colors disabled:opacity-50"
              >
                {deleting ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Trash2 className="w-4 h-4" />
                )}
                Delete
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Metadata Card */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="lg:col-span-1"
          >
            <div className="bg-white rounded-2xl border border-neutral-200/50 p-6 shadow-sm">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-14 h-14 bg-gradient-to-br from-emerald-50 to-emerald-100 rounded-2xl flex items-center justify-center">
                  <Database className="w-7 h-7 text-emerald-600" />
                </div>
                <div>
                  <h2 className="font-semibold text-neutral-900">Dataset Info</h2>
                  <p className="text-sm text-neutral-500">Metadata & statistics</p>
                </div>
              </div>

              <div className="space-y-4">
                {dataset.description && (
                  <div>
                    <p className="text-xs font-medium text-neutral-500 uppercase tracking-wider mb-1">
                      Description
                    </p>
                    <p className="text-sm text-neutral-700">{dataset.description}</p>
                  </div>
                )}

                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-neutral-50 rounded-xl p-4">
                    <div className="flex items-center gap-2 text-neutral-500 mb-1">
                      <HardDrive className="w-4 h-4" />
                      <span className="text-xs font-medium uppercase">Size</span>
                    </div>
                    <p className="text-lg font-semibold text-neutral-900">
                      {formatBytes(dataset.file_size)}
                    </p>
                  </div>

                  <div className="bg-neutral-50 rounded-xl p-4">
                    <div className="flex items-center gap-2 text-neutral-500 mb-1">
                      <Folder className="w-4 h-4" />
                      <span className="text-xs font-medium uppercase">Frames</span>
                    </div>
                    <p className="text-lg font-semibold text-neutral-900">
                      {dataset.frame_count ? formatNumber(dataset.frame_count) : "-"}
                    </p>
                  </div>
                </div>

                <div className="border-t border-neutral-100 pt-4 space-y-3">
                  <div className="flex justify-between">
                    <span className="text-sm text-neutral-500">Format</span>
                    <span className="text-sm font-medium text-neutral-700">
                      {dataset.file_format || "Unknown"}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-neutral-500">Uploaded</span>
                    <span className="text-sm font-medium text-neutral-700">
                      {formatRelativeTime(dataset.created_at)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-neutral-500">Last Updated</span>
                    <span className="text-sm font-medium text-neutral-700">
                      {formatRelativeTime(dataset.updated_at)}
                    </span>
                  </div>
                </div>

                <div className="border-t border-neutral-100 pt-4">
                  <p className="text-xs font-medium text-neutral-500 uppercase tracking-wider mb-2">
                    Storage Path
                  </p>
                  <div className="flex items-center gap-2">
                    <code className="flex-1 text-xs bg-neutral-100 px-3 py-2 rounded-lg text-neutral-600 truncate">
                      {dataset.storage_path}
                    </code>
                    <button
                      onClick={() => copyToClipboard(dataset.storage_path)}
                      className="p-2 hover:bg-neutral-100 rounded-lg transition-colors"
                    >
                      {copiedPath === dataset.storage_path ? (
                        <Check className="w-4 h-4 text-green-500" />
                      ) : (
                        <Copy className="w-4 h-4 text-neutral-400" />
                      )}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>

          {/* File Browser */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="lg:col-span-2"
          >
            <div className="bg-white rounded-2xl border border-neutral-200/50 shadow-sm overflow-hidden">
              <div className="px-6 py-4 border-b border-neutral-100">
                <h2 className="font-semibold text-neutral-900">Files</h2>
                <p className="text-sm text-neutral-500">Browse the dataset structure</p>
              </div>

              <div className="max-h-[600px] overflow-y-auto">
                {files.length === 0 ? (
                  <div className="p-8 text-center">
                    <Folder className="w-12 h-12 text-neutral-200 mx-auto mb-3" />
                    <p className="text-neutral-500">No files found</p>
                    <p className="text-sm text-neutral-400">
                      The file listing might still be loading or the dataset is empty.
                    </p>
                  </div>
                ) : (
                  <div className="py-2">{renderFileTree(files)}</div>
                )}
              </div>
            </div>
          </motion.div>
        </div>
      </main>
    </div>
  );
}
