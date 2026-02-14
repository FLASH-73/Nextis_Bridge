// Canonical API base URL — replaces 3 inconsistent patterns across modals
export function getApiBase(): string {
  if (typeof window === "undefined") return "http://127.0.0.1:8000";
  return `${window.location.protocol}//${window.location.hostname}:8000`;
}

export const API_BASE = getApiBase();

export async function apiFetch<T>(
  path: string,
  options?: RequestInit
): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(error.detail || `API error ${res.status}`);
  }
  return res.json();
}

// Fire-and-forget fetch that doesn't throw on non-2xx (for status polling etc.)
export async function apiFetchSafe<T>(
  path: string,
  options?: RequestInit
): Promise<T | null> {
  try {
    return await apiFetch<T>(path, options);
  } catch {
    return null;
  }
}

export const api = {
  get: <T>(path: string) => apiFetch<T>(path),
  post: <T>(path: string, body?: unknown) =>
    apiFetch<T>(path, {
      method: "POST",
      body: body ? JSON.stringify(body) : undefined,
    }),
  put: <T>(path: string, body: unknown) =>
    apiFetch<T>(path, {
      method: "PUT",
      body: JSON.stringify(body),
    }),
  patch: <T>(path: string, body: unknown) =>
    apiFetch<T>(path, {
      method: "PATCH",
      body: JSON.stringify(body),
    }),
  delete: <T>(path: string, body?: unknown) =>
    apiFetch<T>(path, {
      method: "DELETE",
      body: body ? JSON.stringify(body) : undefined,
    }),
};
