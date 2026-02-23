/**
 * useTaskStore — selector hooks for the tasks slice.
 *
 * Each export is a focused hook that subscribes only to the slice of state
 * it needs. This means components re-render only when their relevant data
 * changes, not on every store update.
 *
 * Usage:
 *   const tasks = useAllTasks();
 *   const { addTask, removeTask } = useTaskActions();
 */
import { useAppStore } from "../store";
import type { TaskFilter, TaskStatus } from "../types";

// ─── Data hooks ───────────────────────────────────────────────────────────────

/** All tasks (unfiltered). Prefer useFilteredTasks for rendering lists. */
export function useAllTasks() {
  return useAppStore((s) => s.tasks);
}

/** Tasks filtered and sorted by the given filter. */
export function useFilteredTasks(filter: TaskFilter) {
  return useAppStore((s) => s.getFilteredTasks(filter));
}

/** A single task by id. Returns undefined if not found. */
export function useTask(id: string) {
  return useAppStore((s) => s.getTaskById(id));
}

/** Count of tasks per status. */
export function useTaskCounts() {
  return useAppStore((s) => {
    const counts = {
      todo: 0,
      "in-progress": 0,
      done: 0,
      total: s.tasks.length,
    };
    for (const task of s.tasks) {
      counts[task.status]++;
    }
    return counts;
  });
}

/** True when there is at least one completed task. */
export function useHasCompletedTasks() {
  return useAppStore((s) => s.tasks.some((t) => t.status === "done"));
}

// ─── Action hooks ─────────────────────────────────────────────────────────────

/** Stable action references — safe to use in dependency arrays. */
export function useTaskActions() {
  return useAppStore((s) => ({
    addTask: s.addTask,
    updateTask: s.updateTask,
    removeTask: s.removeTask,
    toggleTaskStatus: s.toggleTaskStatus,
    setTaskStatus: s.setTaskStatus,
    clearCompleted: s.clearCompleted,
    reorderTasks: s.reorderTasks,
  }));
}

// ─── Convenience combined hook ────────────────────────────────────────────────

/**
 * Returns filtered tasks + the filter setter from the UI slice in one call.
 * Useful for task list components that also render the filter bar.
 */
export function useTaskList() {
  const activeFilter = useAppStore((s) => s.ui.activeTaskFilter);
  const tasks = useAppStore((s) => s.getFilteredTasks(activeFilter));
  const setActiveTaskFilter = useAppStore((s) => s.setActiveTaskFilter);
  const showCompleted = useAppStore((s) => s.ui.showCompleted);

  const visibleTasks = showCompleted
    ? tasks
    : tasks.filter((t) => t.status !== "done");

  return { tasks: visibleTasks, activeFilter, setActiveTaskFilter } as const;
}

// ─── Single-task status hook ──────────────────────────────────────────────────

export function useTaskStatus(id: string): TaskStatus | undefined {
  return useAppStore((s) => s.tasks.find((t) => t.id === id)?.status);
}
