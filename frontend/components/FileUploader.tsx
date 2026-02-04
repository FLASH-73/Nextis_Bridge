"use client";

import { useState, useCallback, useRef } from "react";
import { Upload, Pause, Play, X, CheckCircle, AlertCircle, Folder, FileText, Film, ChevronRight, ChevronDown } from "lucide-react";
import * as tus from "tus-js-client";
import { supabase, type Dataset } from "../lib/supabase";

interface FileUploaderProps {
  onUploadComplete?: (dataset: Dataset) => void;
  onClose?: () => void;
}

type UploadState = "idle" | "uploading" | "paused" | "complete" | "error";

interface FileEntry {
  file: File;
  relativePath: string;
}

interface FolderStructure {
  name: string;
  type: "folder" | "file";
  size?: number;
  children?: FolderStructure[];
}

export default function FileUploader({ onUploadComplete, onClose }: FileUploaderProps) {
  const [dragActive, setDragActive] = useState(false);
  const [uploadState, setUploadState] = useState<UploadState>("idle");
  const [progress, setProgress] = useState(0);
  const [currentFileIndex, setCurrentFileIndex] = useState(0);
  const [selectedFiles, setSelectedFiles] = useState<FileEntry[]>([]);
  const [datasetName, setDatasetName] = useState<string>("");
  const [folderStructure, setFolderStructure] = useState<FolderStructure | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set(["root"]));

  const uploadRef = useRef<tus.Upload | null>(null);
  const isPausedRef = useRef(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  // Process dropped items (files or folders)
  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const items = e.dataTransfer.items;
    if (!items) return;

    const fileEntries: FileEntry[] = [];
    let rootName = "";

    // Process all dropped items
    for (let i = 0; i < items.length; i++) {
      const item = items[i].webkitGetAsEntry();
      if (item) {
        if (item.isDirectory && !rootName) {
          rootName = item.name;
        }
        await traverseFileTree(item, "", fileEntries);
      }
    }

    if (fileEntries.length > 0) {
      processFiles(fileEntries, rootName || "dataset");
    }
  }, []);

  // Recursively traverse folder structure
  const traverseFileTree = async (
    item: FileSystemEntry,
    path: string,
    fileEntries: FileEntry[]
  ): Promise<void> => {
    return new Promise((resolve) => {
      if (item.isFile) {
        (item as FileSystemFileEntry).file((file) => {
          const relativePath = path ? `${path}/${file.name}` : file.name;
          fileEntries.push({ file, relativePath });
          resolve();
        });
      } else if (item.isDirectory) {
        const dirReader = (item as FileSystemDirectoryEntry).createReader();
        const newPath = path ? `${path}/${item.name}` : item.name;

        const readEntries = () => {
          dirReader.readEntries(async (entries) => {
            if (entries.length === 0) {
              resolve();
              return;
            }
            for (const entry of entries) {
              await traverseFileTree(entry, newPath, fileEntries);
            }
            // Continue reading (directories may have many entries)
            readEntries();
          });
        };
        readEntries();
      }
    });
  };

  // Handle folder input change
  const handleFolderInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const fileEntries: FileEntry[] = [];
    let rootName = "";

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const relativePath = file.webkitRelativePath;

      if (!rootName && relativePath) {
        rootName = relativePath.split("/")[0];
      }

      fileEntries.push({
        file,
        relativePath: relativePath || file.name,
      });
    }

    if (fileEntries.length > 0) {
      processFiles(fileEntries, rootName || "dataset");
    }
  };

  // Handle single file input
  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      processFiles([{ file, relativePath: file.name }], file.name);
    }
  };

  // Process selected files and build folder structure
  const processFiles = (entries: FileEntry[], name: string) => {
    setSelectedFiles(entries);
    setDatasetName(name);
    setUploadState("idle");
    setProgress(0);
    setCurrentFileIndex(0);
    setErrorMessage(null);

    // Build folder structure for preview
    const structure = buildFolderStructure(entries, name);
    setFolderStructure(structure);
  };

  // Build tree structure from flat file list
  const buildFolderStructure = (entries: FileEntry[], rootName: string): FolderStructure => {
    const root: FolderStructure = { name: rootName, type: "folder", children: [] };

    for (const entry of entries) {
      const parts = entry.relativePath.split("/");
      let current = root;

      // Skip the root folder name if it matches
      const startIndex = parts[0] === rootName ? 1 : 0;

      for (let i = startIndex; i < parts.length; i++) {
        const part = parts[i];
        const isFile = i === parts.length - 1;

        if (isFile) {
          current.children!.push({
            name: part,
            type: "file",
            size: entry.file.size,
          });
        } else {
          let child = current.children!.find((c) => c.name === part && c.type === "folder");
          if (!child) {
            child = { name: part, type: "folder", children: [] };
            current.children!.push(child);
          }
          current = child;
        }
      }
    }

    // Sort: folders first, then files
    const sortChildren = (node: FolderStructure) => {
      if (node.children) {
        node.children.sort((a, b) => {
          if (a.type !== b.type) return a.type === "folder" ? -1 : 1;
          return a.name.localeCompare(b.name);
        });
        node.children.forEach(sortChildren);
      }
    };
    sortChildren(root);

    return root;
  };

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

  const startUpload = async () => {
    if (selectedFiles.length === 0) return;

    const { data: { session } } = await supabase.auth.getSession();
    if (!session) {
      setErrorMessage("Please sign in to upload files");
      setUploadState("error");
      return;
    }

    isPausedRef.current = false;
    setUploadState("uploading");

    const userId = session.user.id;
    const datasetId = `${Date.now()}-${datasetName}`;
    const basePath = `${userId}/${datasetId}`;

    let completedBytes = 0;
    const totalBytes = getTotalSize();

    // Upload each file sequentially
    for (let i = 0; i < selectedFiles.length; i++) {
      if (isPausedRef.current) {
        return; // Stop if paused
      }

      setCurrentFileIndex(i);
      const entry = selectedFiles[i];
      const storagePath = `${basePath}/${entry.relativePath}`;

      try {
        await uploadSingleFile(entry.file, storagePath, session.access_token, (uploaded) => {
          const totalProgress = ((completedBytes + uploaded) / totalBytes) * 100;
          setProgress(Math.round(totalProgress));
        });
        completedBytes += entry.file.size;
      } catch (error) {
        console.error(`Error uploading ${entry.relativePath}:`, error);
        setErrorMessage(`Failed to upload ${entry.relativePath}`);
        setUploadState("error");
        return;
      }
    }

    // All files uploaded successfully
    setUploadState("complete");
    setProgress(100);

    // Insert dataset record
    const { data, error } = await supabase
      .from("datasets")
      .insert({
        owner_id: userId,
        name: datasetName,
        storage_path: basePath,
        file_size: totalBytes,
        file_format: "lerobot",
        status: "processing",
      })
      .select()
      .single();

    if (error) {
      console.error("Database insert error:", error);
      setErrorMessage("Files uploaded but failed to save metadata");
    } else if (data && onUploadComplete) {
      onUploadComplete(data as Dataset);
    }
  };

  const uploadSingleFile = (
    file: File,
    storagePath: string,
    accessToken: string,
    onProgress: (uploaded: number) => void
  ): Promise<void> => {
    return new Promise((resolve, reject) => {
      const upload = new tus.Upload(file, {
        endpoint: `${process.env.NEXT_PUBLIC_SUPABASE_URL}/storage/v1/upload/resumable`,
        retryDelays: [0, 3000, 5000, 10000, 20000],
        headers: {
          authorization: `Bearer ${accessToken}`,
          "x-upsert": "true",
        },
        uploadDataDuringCreation: true,
        removeFingerprintOnSuccess: true,
        metadata: {
          bucketName: "datasets",
          objectName: storagePath,
          contentType: file.type || "application/octet-stream",
          cacheControl: "3600",
        },
        chunkSize: 6 * 1024 * 1024,
        onError: (error) => {
          reject(error);
        },
        onProgress: (bytesUploaded) => {
          onProgress(bytesUploaded);
        },
        onSuccess: () => {
          resolve();
        },
      });

      uploadRef.current = upload;
      upload.start();
    });
  };

  const togglePause = () => {
    if (uploadState === "uploading") {
      isPausedRef.current = true;
      if (uploadRef.current) {
        uploadRef.current.abort();
      }
      setUploadState("paused");
    } else if (uploadState === "paused") {
      isPausedRef.current = false;
      startUpload(); // Resume from current file
    }
  };

  const cancelUpload = () => {
    isPausedRef.current = true;
    if (uploadRef.current) {
      uploadRef.current.abort();
    }
    resetUploader();
  };

  const resetUploader = () => {
    setSelectedFiles([]);
    setDatasetName("");
    setFolderStructure(null);
    setUploadState("idle");
    setProgress(0);
    setCurrentFileIndex(0);
    setErrorMessage(null);
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

  const isLeRobotDataset = folderStructure?.children?.some(
    (child) => child.type === "folder" && ["meta", "data", "videos"].includes(child.name)
  );

  return (
    <div className="w-full max-w-lg mx-auto">
      {/* Drop Zone */}
      {selectedFiles.length === 0 && (
        <div
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          className={`
            relative border-2 border-dashed rounded-2xl p-8 text-center
            transition-all duration-200 ease-out
            ${dragActive
              ? "border-black dark:border-white bg-black/5 dark:bg-white/5 scale-[1.02]"
              : "border-neutral-200 dark:border-zinc-700 hover:border-neutral-300 dark:hover:border-zinc-600"
            }
          `}
        >
          <input
            ref={fileInputRef}
            type="file"
            onChange={handleFileInput}
            accept=".mcap,.db3,.bag,.hdf5,.h5,.zarr,.zip"
            className="hidden"
          />
          <input
            ref={folderInputRef}
            type="file"
            onChange={handleFolderInput}
            // @ts-ignore - webkitdirectory is not in types
            webkitdirectory=""
            directory=""
            multiple
            className="hidden"
          />

          <div className={`
            w-14 h-14 mx-auto mb-4 rounded-2xl flex items-center justify-center
            transition-colors duration-200
            ${dragActive ? "bg-black dark:bg-white text-white dark:text-black" : "bg-neutral-100 dark:bg-zinc-800 text-neutral-400 dark:text-zinc-500"}
          `}>
            <Folder className="w-7 h-7" />
          </div>

          <p className="text-base font-medium text-neutral-800 dark:text-zinc-200 mb-1">
            {dragActive ? "Drop your dataset here" : "Drag & drop LeRobot dataset"}
          </p>
          <p className="text-sm text-neutral-500 dark:text-zinc-400 mb-4">
            or select manually below
          </p>

          <div className="flex gap-2 justify-center">
            <button
              onClick={() => folderInputRef.current?.click()}
              className="px-4 py-2 bg-black dark:bg-white text-white dark:text-black text-sm font-medium rounded-xl hover:bg-neutral-800 dark:hover:bg-zinc-200 transition-colors"
            >
              Select Folder
            </button>
            <button
              onClick={() => fileInputRef.current?.click()}
              className="px-4 py-2 bg-neutral-100 dark:bg-zinc-800 text-neutral-700 dark:text-zinc-300 text-sm font-medium rounded-xl hover:bg-neutral-200 dark:hover:bg-zinc-700 transition-colors"
            >
              Single File
            </button>
          </div>

          <p className="text-xs text-neutral-400 dark:text-zinc-500 mt-4">
            Supports LeRobot v2 format, MCAP, ROS 2 Bags, HDF5
          </p>
        </div>
      )}

      {/* Dataset Selected */}
      {selectedFiles.length > 0 && (
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
                onClick={resetUploader}
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
                onClick={startUpload}
                className="flex-1 py-2.5 bg-black dark:bg-white hover:bg-neutral-800 dark:hover:bg-zinc-200 text-white dark:text-black font-medium rounded-xl transition-colors flex items-center justify-center gap-2"
              >
                <Upload className="w-4 h-4" />
                Upload Dataset
              </button>
            )}

            {(uploadState === "uploading" || uploadState === "paused") && (
              <>
                <button
                  onClick={togglePause}
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
                  onClick={cancelUpload}
                  className="px-4 py-2.5 bg-neutral-200 dark:bg-zinc-700 hover:bg-neutral-300 dark:hover:bg-zinc-600 text-neutral-700 dark:text-zinc-300 font-medium rounded-xl transition-colors"
                >
                  Cancel
                </button>
              </>
            )}

            {(uploadState === "complete" || uploadState === "error") && (
              <button
                onClick={uploadState === "complete" ? (onClose || resetUploader) : resetUploader}
                className="flex-1 py-2.5 bg-black dark:bg-white hover:bg-neutral-800 dark:hover:bg-zinc-200 text-white dark:text-black font-medium rounded-xl transition-colors"
              >
                {uploadState === "complete" ? "Done" : "Try Again"}
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
