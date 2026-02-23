/**
 * Shared types for the modular Zustand store.
 *
 * Two slices are provided:
 *   TasksSlice — task list management (CRUD + filter)
 *   UISettingsSlice — display preferences (dark mode, sidebar, active view)
 *
 * Both slices are merged into a single store that is persisted to localStorage
 * and wired to Redux DevTools.
 */

// ─── Tasks ─────────────────────────────────────────────────────────────────────

export type TaskPriority = "low" | "medium" | "high";
export type TaskStatus = "todo" | "in-progress" | "done";

export interface Task {
  id: string;
  title: string;
  description?: string;
  status: TaskStatus;
  priority: TaskPriority;
  tags: string[];
  createdAt: string; // ISO 8601
  updatedAt: string;
  dueDate?: string;
}

export type NewTask = Omit<Task, "id" | "createdAt" | "updatedAt" | "status"> &
  Partial<Pick<Task, "status">>;

export type TaskFilter = "all" | TaskStatus;

// ─── UI Settings ───────────────────────────────────────────────────────────────

export type ThemeMode = "light" | "dark" | "system";
export type SidebarState = "expanded" | "collapsed" | "hidden";

export interface UISettings {
  theme: ThemeMode;
  sidebar: SidebarState;
  /** Which task filter is currently active in the UI */
  activeTaskFilter: TaskFilter;
  /** Number of tasks to display per page */
  pageSize: 10 | 25 | 50;
  /** Whether to show completed tasks in lists */
  showCompleted: boolean;
}
