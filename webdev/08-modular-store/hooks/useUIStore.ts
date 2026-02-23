/**
 * useUIStore — selector hooks for the UI settings slice.
 */
import { useAppStore } from "../store";
import type { ThemeMode, SidebarState, TaskFilter } from "../types";

// ─── Data hooks ───────────────────────────────────────────────────────────────

export function useTheme(): ThemeMode {
  return useAppStore((s) => s.ui.theme);
}

export function useSidebarState(): SidebarState {
  return useAppStore((s) => s.ui.sidebar);
}

export function useActiveTaskFilter(): TaskFilter {
  return useAppStore((s) => s.ui.activeTaskFilter);
}

export function usePageSize() {
  return useAppStore((s) => s.ui.pageSize);
}

export function useShowCompleted() {
  return useAppStore((s) => s.ui.showCompleted);
}

/** Full UI settings object — use only when you need all fields at once. */
export function useUISettings() {
  return useAppStore((s) => s.ui);
}

// ─── Action hooks ─────────────────────────────────────────────────────────────

export function useUIActions() {
  return useAppStore((s) => ({
    setTheme: s.setTheme,
    toggleTheme: s.toggleTheme,
    setSidebar: s.setSidebar,
    toggleSidebar: s.toggleSidebar,
    setActiveTaskFilter: s.setActiveTaskFilter,
    setPageSize: s.setPageSize,
    setShowCompleted: s.setShowCompleted,
    resetUI: s.resetUI,
  }));
}

// ─── Derived booleans ─────────────────────────────────────────────────────────

/** True when the resolved theme is dark (not system). */
export function useIsDarkMode(): boolean {
  return useAppStore((s) => s.ui.theme === "dark");
}

export function useIsSidebarOpen(): boolean {
  return useAppStore((s) => s.ui.sidebar === "expanded");
}
