/**
 * Root Zustand store — merges TasksSlice and UISlice.
 *
 * Middleware stack (outer-to-inner):
 *   devtools → persist → base store
 *
 * Install: npm install zustand nanoid
 *
 * Persistence:
 *   - Stored in localStorage under the key "app-store".
 *   - Only the tasks array and ui object are persisted (all other
 *     action functions are excluded automatically by Zustand persist).
 *   - Version 1; add `migrate` if the shape changes in a future release.
 *
 * DevTools:
 *   - Action labels use the "slice/action" convention, visible in the
 *     Redux DevTools browser extension.
 */
import { create } from "zustand";
import { devtools, persist } from "zustand/middleware";
import { createTasksSlice } from "./tasks-slice";
import { createUISlice } from "./ui-slice";
import type { TasksSlice } from "./tasks-slice";
import type { UISlice } from "./ui-slice";

// ─── Combined store type ──────────────────────────────────────────────────────

export type AppStore = TasksSlice & UISlice;

// ─── Store creation ───────────────────────────────────────────────────────────

export const useAppStore = create<AppStore>()(
  devtools(
    persist(
      (...args) => ({
        ...createTasksSlice(...args),
        ...createUISlice(...args),
      }),
      {
        name: "app-store",
        version: 1,
        // Only persist data — Zustand persist automatically skips functions,
        // but being explicit here makes it clear what ends up in localStorage.
        partialize: (state) => ({
          tasks: state.tasks,
          ui: state.ui,
        }),
      },
    ),
    {
      name: "AppStore",
      // Anonymise actions that don't carry an explicit name string.
      anonymousActionType: "anonymous",
    },
  ),
);

// ─── Pre-bound typed hooks (preferred over raw useAppStore) ───────────────────
// Import from the dedicated hooks/ files for cleaner dependency tracking.
// These re-exports are convenience aliases only.

export type { TasksSlice, UISlice };
