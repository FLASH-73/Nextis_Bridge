"use client";

import { useState, useCallback, useRef } from "react";
import { Folder } from "lucide-react";
import * as tus from "tus-js-client";
import { supabase, type Dataset } from "../../../lib/supabase";
import UploadProgress, {
  type UploadState,
  type FileEntry,
  type FolderStructure,
} from "./UploadProgress";

interface FileUploaderProps {
  onUploadComplete?: (dataset: Dataset) => void;
  onClose?: () => void;
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
        <UploadProgress
          selectedFiles={selectedFiles}
          datasetName={datasetName}
          folderStructure={folderStructure}
          uploadState={uploadState}
          progress={progress}
          currentFileIndex={currentFileIndex}
          errorMessage={errorMessage}
          isLeRobotDataset={!!isLeRobotDataset}
          onStartUpload={startUpload}
          onTogglePause={togglePause}
          onCancelUpload={cancelUpload}
          onResetUploader={resetUploader}
          onClose={onClose}
        />
      )}
    </div>
  );
}
