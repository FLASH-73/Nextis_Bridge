import { createClient, SupabaseClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || '';
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || '';

// Create client only if credentials are available
export const supabase: SupabaseClient = createClient(
  supabaseUrl || 'https://placeholder.supabase.co',
  supabaseAnonKey || 'placeholder-key'
);

export const isSupabaseConfigured = Boolean(supabaseUrl && supabaseAnonKey);

export type Dataset = {
  id: string;
  owner_id: string;
  name: string;
  description: string | null;
  status: 'uploading' | 'processing' | 'ready' | 'cleaning' | 'training' | 'error';
  storage_path: string;
  file_size: number;
  file_format: string | null;
  duration_seconds: number | null;
  frame_count: number | null;
  recorded_at: string | null;
  created_at: string;
  updated_at: string;
};
