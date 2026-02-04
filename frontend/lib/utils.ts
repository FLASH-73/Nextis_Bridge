/**
 * Format a date string into a human-readable relative time
 * e.g., "Just now", "5 minutes ago", "2 days ago", "Jan 15, 2024"
 */
export function formatRelativeTime(dateString: string): string {
  const date = new Date(dateString);
  const now = new Date();
  const diffInSeconds = Math.floor((now.getTime() - date.getTime()) / 1000);

  if (diffInSeconds < 60) {
    return "Just now";
  }

  const diffInMinutes = Math.floor(diffInSeconds / 60);
  if (diffInMinutes < 60) {
    return diffInMinutes === 1 ? "1 minute ago" : `${diffInMinutes} minutes ago`;
  }

  const diffInHours = Math.floor(diffInMinutes / 60);
  if (diffInHours < 24) {
    return diffInHours === 1 ? "1 hour ago" : `${diffInHours} hours ago`;
  }

  const diffInDays = Math.floor(diffInHours / 24);
  if (diffInDays < 7) {
    return diffInDays === 1 ? "Yesterday" : `${diffInDays} days ago`;
  }

  if (diffInDays < 30) {
    const weeks = Math.floor(diffInDays / 7);
    return weeks === 1 ? "1 week ago" : `${weeks} weeks ago`;
  }

  // For older dates, show the actual date
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: date.getFullYear() !== now.getFullYear() ? "numeric" : undefined,
  });
}

/**
 * Format bytes into human-readable size
 * e.g., "1.5 GB", "256 MB", "1.2 KB"
 */
export function formatBytes(bytes: number, decimals: number = 1): string {
  if (bytes === 0) return "0 B";

  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(decimals))} ${sizes[i]}`;
}

/**
 * Format a number with commas for thousands
 * e.g., 1234567 -> "1,234,567"
 */
export function formatNumber(num: number): string {
  return num.toLocaleString();
}

/**
 * Get status color classes for dataset status badges
 */
export function getStatusColor(status: string): { bg: string; text: string; dot: string } {
  switch (status) {
    case "ready":
      return { bg: "bg-emerald-50", text: "text-emerald-700", dot: "bg-emerald-500" };
    case "processing":
      return { bg: "bg-blue-50", text: "text-blue-700", dot: "bg-blue-500" };
    case "uploading":
      return { bg: "bg-amber-50", text: "text-amber-700", dot: "bg-amber-500" };
    case "training":
      return { bg: "bg-purple-50", text: "text-purple-700", dot: "bg-purple-500" };
    case "cleaning":
      return { bg: "bg-cyan-50", text: "text-cyan-700", dot: "bg-cyan-500" };
    case "error":
      return { bg: "bg-red-50", text: "text-red-700", dot: "bg-red-500" };
    default:
      return { bg: "bg-neutral-50", text: "text-neutral-700", dot: "bg-neutral-500" };
  }
}
