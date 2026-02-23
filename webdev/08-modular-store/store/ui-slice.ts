/**
 * UI Settings slice for the modular Zustand store.
 *
 * Owns display preferences that should survive page refreshes:
 *   - Theme (light / dark / system)
 *   - Sidebar state
 *   - Active task filter
 *   - Page size
 *   - Show-completed toggle
 */
import type { StateCreator } from "zustand";
import type { UISettings, ThemeMode, SidebarState, TaskFilter } from "../types";

// ─── Slice state + actions ────────────────────────────────────────────────────

export interface UISlice {
  ui: UISettings;

  setTheme: (theme: ThemeMode) => void;
  toggleTheme: () => void; // cycles light → dark → system → light

  setSidebar: (state: SidebarState) => void;
  toggleSidebar: () => void; // collapsed ↔ expanded

  setActiveTaskFilter: (filter: TaskFilter) => void;
  setPageSize: (size: UISettings["pageSize"]) => void;
  setShowCompleted: (show: boolean) => void;

  /** Reset all UI settings to defaults. */
  resetUI: () => void;
}

// ─── Defaults ─────────────────────────────────────────────────────────────────

const DEFAULT_UI: UISettings = {
  theme: "system",
  sidebar: "expanded",
  activeTaskFilter: "all",
  pageSize: 25,
  showCompleted: true,
};

const THEME_CYCLE: Record<ThemeMode, ThemeMode> = {
  light: "dark",
  dark: "system",
  system: "light",
};

// ─── Factory ──────────────────────────────────────────────────────────────────

export const createUISlice: StateCreator<
  UISlice,
  [["zustand/devtools", never], ["zustand/persist", unknown]],
  [],
  UISlice
> = (set) => ({
  ui: DEFAULT_UI,

  setTheme: (theme) =>
    set((state) => ({ ui: { ...state.ui, theme } }), false, "ui/setTheme"),

  toggleTheme: () =>
    set(
      (state) => ({ ui: { ...state.ui, theme: THEME_CYCLE[state.ui.theme] } }),
      false,
      "ui/toggleTheme",
    ),

  setSidebar: (sidebar) =>
    set((state) => ({ ui: { ...state.ui, sidebar } }), false, "ui/setSidebar"),

  toggleSidebar: () =>
    set(
      (state) => ({
        ui: {
          ...state.ui,
          sidebar: state.ui.sidebar === "expanded" ? "collapsed" : "expanded",
        },
      }),
      false,
      "ui/toggleSidebar",
    ),

  setActiveTaskFilter: (activeTaskFilter) =>
    set(
      (state) => ({ ui: { ...state.ui, activeTaskFilter } }),
      false,
      "ui/setActiveTaskFilter",
    ),

  setPageSize: (pageSize) =>
    set(
      (state) => ({ ui: { ...state.ui, pageSize } }),
      false,
      "ui/setPageSize",
    ),

  setShowCompleted: (showCompleted) =>
    set(
      (state) => ({ ui: { ...state.ui, showCompleted } }),
      false,
      "ui/setShowCompleted",
    ),

  resetUI: () => set(() => ({ ui: DEFAULT_UI }), false, "ui/resetUI"),
});
