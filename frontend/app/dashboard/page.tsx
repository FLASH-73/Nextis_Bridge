"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import {
  Database,
  Upload,
  HardDrive,
  Clock,
  MoreVertical,
  Trash2,
  ExternalLink,
  Sparkles,
  Cloud,
  Folder,
  Search,
  Grid3X3,
  List,
  CheckCircle,
  Loader2,
  AlertCircle,
  Bot,
} from "lucide-react";
import { useAuth } from "../../lib/AuthContext";
import { formatRelativeTime, formatBytes, formatNumber, getStatusColor } from "../../lib/utils";
import {
  getCloudDatasets,
  getStorageUsage,
  deleteDataset,
  CloudDataset,
  StorageUsage,
} from "./actions";
import UploadModal from "../../components/UploadModal";
import AuthModal from "../../components/AuthModal";

export default function DashboardPage() {
  const router = useRouter();
  const { user, loading: authLoading } = useAuth();

  const [datasets, setDatasets] = useState<CloudDataset[]>([]);
  const [storageUsage, setStorageUsage] = useState<StorageUsage | null>(null);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [menuOpenId, setMenuOpenId] = useState<string | null>(null);

  // Fetch data when user is authenticated
  useEffect(() => {
    if (user) {
      fetchData();
    } else if (!authLoading) {
      setLoading(false);
    }
  }, [user, authLoading]);

  const fetchData = async () => {
    if (!user) return;
    setLoading(true);
    try {
      const [datasetsData, usageData] = await Promise.all([
        getCloudDatasets(user.id),
        getStorageUsage(user.id),
      ]);
      setDatasets(datasetsData);
      setStorageUsage(usageData);
    } catch (err) {
      console.error("Failed to fetch data:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (datasetId: string) => {
    if (!user) return;
    if (!confirm("Are you sure you want to delete this dataset? This cannot be undone.")) return;

    setDeletingId(datasetId);
    const result = await deleteDataset(datasetId, user.id);
    if (result.success) {
      setDatasets((prev) => prev.filter((d) => d.id !== datasetId));
      // Refresh storage usage
      const usageData = await getStorageUsage(user.id);
      setStorageUsage(usageData);
    } else {
      alert(`Failed to delete: ${result.error}`);
    }
    setDeletingId(null);
    setMenuOpenId(null);
  };

  // Filter datasets by search query
  const filteredDatasets = datasets.filter(
    (d) =>
      d.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      d.description?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // If not authenticated
  if (!authLoading && !user) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-neutral-50 to-neutral-100 flex items-center justify-center p-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center max-w-md"
        >
          <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-purple-600 rounded-3xl flex items-center justify-center mx-auto mb-6 shadow-xl">
            <Cloud className="w-10 h-10 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-neutral-900 mb-3">Welcome to RoboCloud</h1>
          <p className="text-neutral-600 mb-8">
            Sign in to access your robotics datasets, track training progress, and collaborate with
            your team.
          </p>
          <button
            onClick={() => setShowAuthModal(true)}
            className="px-8 py-3 bg-black text-white rounded-xl font-medium hover:bg-neutral-800 transition-colors"
          >
            Sign In to Continue
          </button>
        </motion.div>
        <AuthModal isOpen={showAuthModal} onClose={() => setShowAuthModal(false)} />
      </div>
    );
  }

  // Loading state
  if (loading || authLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-neutral-50 to-neutral-100 flex items-center justify-center">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex flex-col items-center gap-4"
        >
          <Loader2 className="w-8 h-8 text-blue-500 animate-spin" />
          <p className="text-neutral-500">Loading your workspace...</p>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-neutral-50 to-neutral-100">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-xl border-b border-neutral-200/50 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
                <Bot className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-neutral-900">RoboCloud</h1>
                <p className="text-sm text-neutral-500">
                  Welcome back, {user?.email?.split("@")[0]}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <button
                onClick={() => setShowUploadModal(true)}
                className="flex items-center gap-2 px-4 py-2 bg-black text-white rounded-xl font-medium hover:bg-neutral-800 transition-colors"
              >
                <Upload className="w-4 h-4" />
                Upload Dataset
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Storage Usage Card */}
        {storageUsage && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white rounded-2xl border border-neutral-200/50 p-6 mb-8 shadow-sm"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="p-2.5 bg-blue-50 rounded-xl">
                  <HardDrive className="w-5 h-5 text-blue-600" />
                </div>
                <div>
                  <h2 className="font-semibold text-neutral-900">Storage Usage</h2>
                  <p className="text-sm text-neutral-500">Free Tier - 10 GB included</p>
                </div>
              </div>
              <div className="text-right">
                <p className="text-2xl font-bold text-neutral-900">
                  {formatBytes(storageUsage.used)}
                </p>
                <p className="text-sm text-neutral-500">of {formatBytes(storageUsage.limit)}</p>
              </div>
            </div>

            <div className="relative">
              <div className="w-full bg-neutral-100 rounded-full h-3 overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${storageUsage.percentage}%` }}
                  transition={{ duration: 0.8, ease: "easeOut" }}
                  className={`h-full rounded-full ${
                    storageUsage.percentage > 90
                      ? "bg-red-500"
                      : storageUsage.percentage > 70
                      ? "bg-amber-500"
                      : "bg-gradient-to-r from-blue-500 to-purple-500"
                  }`}
                />
              </div>
              <p className="text-xs text-neutral-400 mt-2">
                {storageUsage.percentage.toFixed(1)}% used
                {storageUsage.percentage > 80 && (
                  <span className="text-amber-600 ml-2">Consider upgrading for more space</span>
                )}
              </p>
            </div>
          </motion.div>
        )}

        {/* Toolbar */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-neutral-400" />
              <input
                type="text"
                placeholder="Search datasets..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 pr-4 py-2 bg-white border border-neutral-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 w-64"
              />
            </div>
          </div>

          <div className="flex items-center gap-2 bg-white border border-neutral-200 rounded-xl p-1">
            <button
              onClick={() => setViewMode("grid")}
              className={`p-2 rounded-lg transition-colors ${
                viewMode === "grid"
                  ? "bg-neutral-100 text-neutral-900"
                  : "text-neutral-400 hover:text-neutral-600"
              }`}
            >
              <Grid3X3 className="w-4 h-4" />
            </button>
            <button
              onClick={() => setViewMode("list")}
              className={`p-2 rounded-lg transition-colors ${
                viewMode === "list"
                  ? "bg-neutral-100 text-neutral-900"
                  : "text-neutral-400 hover:text-neutral-600"
              }`}
            >
              <List className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Empty State */}
        {filteredDatasets.length === 0 && !loading && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-white rounded-3xl border border-neutral-200/50 p-12 text-center shadow-sm"
          >
            <div className="w-24 h-24 bg-gradient-to-br from-blue-100 to-purple-100 rounded-3xl flex items-center justify-center mx-auto mb-6">
              <Sparkles className="w-12 h-12 text-blue-500" />
            </div>
            <h2 className="text-2xl font-bold text-neutral-900 mb-3">
              {searchQuery ? "No datasets found" : "Ready to get started?"}
            </h2>
            <p className="text-neutral-500 mb-8 max-w-md mx-auto">
              {searchQuery
                ? `No datasets match "${searchQuery}". Try a different search term.`
                : "Upload your first robotics dataset to start building better robots with RoboCloud."}
            </p>
            {!searchQuery && (
              <button
                onClick={() => setShowUploadModal(true)}
                className="inline-flex items-center gap-2 px-6 py-3 bg-black text-white rounded-xl font-medium hover:bg-neutral-800 transition-colors"
              >
                <Upload className="w-5 h-5" />
                Upload Your First Dataset
              </button>
            )}
          </motion.div>
        )}

        {/* Dataset Grid */}
        {filteredDatasets.length > 0 && viewMode === "grid" && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4"
          >
            <AnimatePresence>
              {filteredDatasets.map((dataset, index) => {
                const statusColors = getStatusColor(dataset.status);
                return (
                  <motion.div
                    key={dataset.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0.95 }}
                    transition={{ delay: index * 0.05 }}
                    className="bg-white rounded-2xl border border-neutral-200/50 p-5 hover:shadow-lg hover:border-neutral-300/50 transition-all cursor-pointer group relative"
                    onClick={() => router.push(`/dashboard/datasets/${dataset.id}`)}
                  >
                    {/* Menu Button */}
                    <div className="absolute top-4 right-4">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setMenuOpenId(menuOpenId === dataset.id ? null : dataset.id);
                        }}
                        className="p-1.5 hover:bg-neutral-100 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity"
                      >
                        <MoreVertical className="w-4 h-4 text-neutral-400" />
                      </button>

                      {/* Dropdown Menu */}
                      <AnimatePresence>
                        {menuOpenId === dataset.id && (
                          <motion.div
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.95 }}
                            className="absolute right-0 top-8 bg-white border border-neutral-200 rounded-xl shadow-lg py-1 min-w-[140px] z-10"
                          >
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                router.push(`/dashboard/datasets/${dataset.id}`);
                              }}
                              className="w-full px-4 py-2 text-left text-sm hover:bg-neutral-50 flex items-center gap-2"
                            >
                              <ExternalLink className="w-4 h-4" />
                              View Details
                            </button>
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                handleDelete(dataset.id);
                              }}
                              disabled={deletingId === dataset.id}
                              className="w-full px-4 py-2 text-left text-sm hover:bg-red-50 text-red-600 flex items-center gap-2"
                            >
                              {deletingId === dataset.id ? (
                                <Loader2 className="w-4 h-4 animate-spin" />
                              ) : (
                                <Trash2 className="w-4 h-4" />
                              )}
                              Delete
                            </button>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>

                    {/* Dataset Icon */}
                    <div className="w-12 h-12 bg-gradient-to-br from-emerald-50 to-emerald-100 rounded-xl flex items-center justify-center mb-4">
                      <Database className="w-6 h-6 text-emerald-600" />
                    </div>

                    {/* Dataset Info */}
                    <h3 className="font-semibold text-neutral-900 mb-1 truncate pr-8">
                      {dataset.name}
                    </h3>
                    {dataset.description && (
                      <p className="text-sm text-neutral-500 mb-3 line-clamp-2">
                        {dataset.description}
                      </p>
                    )}

                    {/* Status Badge */}
                    <div
                      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${statusColors.bg} ${statusColors.text} mb-4`}
                    >
                      <span className={`w-1.5 h-1.5 rounded-full ${statusColors.dot}`} />
                      {dataset.status.charAt(0).toUpperCase() + dataset.status.slice(1)}
                    </div>

                    {/* Stats */}
                    <div className="flex items-center gap-4 text-xs text-neutral-500 border-t border-neutral-100 pt-4">
                      <div className="flex items-center gap-1">
                        <HardDrive className="w-3.5 h-3.5" />
                        {formatBytes(dataset.file_size)}
                      </div>
                      {dataset.frame_count && (
                        <div className="flex items-center gap-1">
                          <Folder className="w-3.5 h-3.5" />
                          {formatNumber(dataset.frame_count)} frames
                        </div>
                      )}
                      <div className="flex items-center gap-1 ml-auto">
                        <Clock className="w-3.5 h-3.5" />
                        {formatRelativeTime(dataset.created_at)}
                      </div>
                    </div>
                  </motion.div>
                );
              })}
            </AnimatePresence>
          </motion.div>
        )}

        {/* Dataset List */}
        {filteredDatasets.length > 0 && viewMode === "list" && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="bg-white rounded-2xl border border-neutral-200/50 overflow-hidden shadow-sm"
          >
            <table className="w-full">
              <thead>
                <tr className="bg-neutral-50 border-b border-neutral-200">
                  <th className="text-left px-6 py-3 text-xs font-semibold text-neutral-500 uppercase tracking-wider">
                    Dataset
                  </th>
                  <th className="text-left px-6 py-3 text-xs font-semibold text-neutral-500 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="text-left px-6 py-3 text-xs font-semibold text-neutral-500 uppercase tracking-wider">
                    Size
                  </th>
                  <th className="text-left px-6 py-3 text-xs font-semibold text-neutral-500 uppercase tracking-wider">
                    Frames
                  </th>
                  <th className="text-left px-6 py-3 text-xs font-semibold text-neutral-500 uppercase tracking-wider">
                    Uploaded
                  </th>
                  <th className="w-10"></th>
                </tr>
              </thead>
              <tbody className="divide-y divide-neutral-100">
                {filteredDatasets.map((dataset) => {
                  const statusColors = getStatusColor(dataset.status);
                  return (
                    <tr
                      key={dataset.id}
                      className="hover:bg-neutral-50 cursor-pointer transition-colors"
                      onClick={() => router.push(`/dashboard/datasets/${dataset.id}`)}
                    >
                      <td className="px-6 py-4">
                        <div className="flex items-center gap-3">
                          <div className="w-10 h-10 bg-gradient-to-br from-emerald-50 to-emerald-100 rounded-xl flex items-center justify-center">
                            <Database className="w-5 h-5 text-emerald-600" />
                          </div>
                          <div>
                            <p className="font-medium text-neutral-900">{dataset.name}</p>
                            {dataset.description && (
                              <p className="text-sm text-neutral-500 truncate max-w-xs">
                                {dataset.description}
                              </p>
                            )}
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4">
                        <span
                          className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${statusColors.bg} ${statusColors.text}`}
                        >
                          <span className={`w-1.5 h-1.5 rounded-full ${statusColors.dot}`} />
                          {dataset.status.charAt(0).toUpperCase() + dataset.status.slice(1)}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-sm text-neutral-600">
                        {formatBytes(dataset.file_size)}
                      </td>
                      <td className="px-6 py-4 text-sm text-neutral-600">
                        {dataset.frame_count ? formatNumber(dataset.frame_count) : "-"}
                      </td>
                      <td className="px-6 py-4 text-sm text-neutral-500">
                        {formatRelativeTime(dataset.created_at)}
                      </td>
                      <td className="px-4 py-4">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDelete(dataset.id);
                          }}
                          disabled={deletingId === dataset.id}
                          className="p-2 hover:bg-red-50 rounded-lg text-neutral-400 hover:text-red-500 transition-colors"
                        >
                          {deletingId === dataset.id ? (
                            <Loader2 className="w-4 h-4 animate-spin" />
                          ) : (
                            <Trash2 className="w-4 h-4" />
                          )}
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </motion.div>
        )}
      </main>

      {/* Upload Modal */}
      <UploadModal isOpen={showUploadModal} onClose={() => setShowUploadModal(false)} />
    </div>
  );
}
