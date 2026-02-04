"use server";

import { createClient } from "@supabase/supabase-js";
import { cookies } from "next/headers";

// Server-side Supabase client with service role for admin operations
// or use the user's session for RLS-protected queries
function getSupabaseClient() {
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
  const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;
  return createClient(supabaseUrl, supabaseAnonKey);
}

export interface CloudDataset {
  id: string;
  owner_id: string;
  name: string;
  description: string | null;
  status: "uploading" | "processing" | "ready" | "cleaning" | "training" | "error";
  storage_path: string;
  file_size: number;
  file_format: string | null;
  duration_seconds: number | null;
  frame_count: number | null;
  recorded_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface StorageUsage {
  used: number;
  limit: number;
  percentage: number;
}

// Free tier limit: 10GB
const FREE_TIER_LIMIT = 10 * 1024 * 1024 * 1024; // 10GB in bytes

export async function getCloudDatasets(userId: string): Promise<CloudDataset[]> {
  const supabase = getSupabaseClient();

  const { data, error } = await supabase
    .from("datasets")
    .select("*")
    .eq("owner_id", userId)
    .order("created_at", { ascending: false });

  if (error) {
    console.error("Error fetching datasets:", error);
    return [];
  }

  return data || [];
}

export async function getStorageUsage(userId: string): Promise<StorageUsage> {
  const supabase = getSupabaseClient();

  const { data, error } = await supabase
    .from("datasets")
    .select("file_size")
    .eq("owner_id", userId);

  if (error) {
    console.error("Error fetching storage usage:", error);
    return { used: 0, limit: FREE_TIER_LIMIT, percentage: 0 };
  }

  const totalUsed = (data || []).reduce((sum, d) => sum + (d.file_size || 0), 0);
  const percentage = Math.min((totalUsed / FREE_TIER_LIMIT) * 100, 100);

  return {
    used: totalUsed,
    limit: FREE_TIER_LIMIT,
    percentage,
  };
}

export async function getDatasetById(datasetId: string): Promise<CloudDataset | null> {
  const supabase = getSupabaseClient();

  const { data, error } = await supabase
    .from("datasets")
    .select("*")
    .eq("id", datasetId)
    .single();

  if (error) {
    console.error("Error fetching dataset:", error);
    return null;
  }

  return data;
}

export interface StorageFile {
  name: string;
  path: string;
  size: number;
  isFolder: boolean;
  children?: StorageFile[];
}

export async function getDatasetFiles(userId: string, datasetId: string): Promise<StorageFile[]> {
  const supabase = getSupabaseClient();
  const basePath = `${userId}/${datasetId}`;

  // List files recursively from storage
  const { data, error } = await supabase.storage
    .from("datasets")
    .list(basePath, {
      limit: 1000,
      sortBy: { column: "name", order: "asc" },
    });

  if (error) {
    console.error("Error listing files:", error);
    return [];
  }

  // Build folder structure recursively
  const buildStructure = async (path: string): Promise<StorageFile[]> => {
    const { data: items, error } = await supabase.storage
      .from("datasets")
      .list(path, { limit: 1000 });

    if (error || !items) return [];

    const files: StorageFile[] = [];
    for (const item of items) {
      const fullPath = `${path}/${item.name}`;

      if (item.id === null) {
        // It's a folder
        const children = await buildStructure(fullPath);
        files.push({
          name: item.name,
          path: fullPath,
          size: 0,
          isFolder: true,
          children,
        });
      } else {
        // It's a file
        files.push({
          name: item.name,
          path: fullPath,
          size: item.metadata?.size || 0,
          isFolder: false,
        });
      }
    }
    return files;
  };

  return buildStructure(basePath);
}

export async function deleteDataset(datasetId: string, userId: string): Promise<{ success: boolean; error?: string }> {
  const supabase = getSupabaseClient();
  const storagePath = `${userId}/${datasetId}`;

  try {
    // First, list all files in the dataset folder
    const { data: files, error: listError } = await supabase.storage
      .from("datasets")
      .list(storagePath, { limit: 1000 });

    if (listError) {
      throw new Error(`Failed to list files: ${listError.message}`);
    }

    // Delete all files recursively
    if (files && files.length > 0) {
      const filePaths = files.map((f) => `${storagePath}/${f.name}`);
      const { error: deleteStorageError } = await supabase.storage
        .from("datasets")
        .remove(filePaths);

      if (deleteStorageError) {
        throw new Error(`Failed to delete files: ${deleteStorageError.message}`);
      }
    }

    // Delete the database record
    const { error: deleteDbError } = await supabase
      .from("datasets")
      .delete()
      .eq("id", datasetId)
      .eq("owner_id", userId);

    if (deleteDbError) {
      throw new Error(`Failed to delete database record: ${deleteDbError.message}`);
    }

    return { success: true };
  } catch (err) {
    console.error("Delete dataset error:", err);
    return { success: false, error: err instanceof Error ? err.message : "Unknown error" };
  }
}
