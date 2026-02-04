"use client";

import { AuthProvider } from "../lib/AuthContext";
import { ThemeProvider } from "../lib/ThemeContext";

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <ThemeProvider>
      <AuthProvider>{children}</AuthProvider>
    </ThemeProvider>
  );
}
