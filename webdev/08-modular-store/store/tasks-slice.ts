/**
 * Tasks slice for the modular Zustand store.
 *
 * This slice owns:
 *   - The tasks array
 *   - All CRUD operations (add, update, remove, toggle status)
 *   - Bulk helpers (clearCompleted, reorder)
 *
 * Import into the root store via the slice pattern:
 *   create<TasksSlice & UISlice>()((...args) => ({
 *     ...createTasksSlice(...args),
 *     ...createUISlice(...args),
 *   }))
 */
import { nanoid } from "nanoid";
import type { StateCreator } from "zustand";
import type { Task, NewTask, TaskStatus, TaskFilter } from "../types";

// ─── Slice state + actions ────────────────────────────────────────────────────

export interface TasksSlice {
  tasks: Task[];

  /** Add a new task. Returns the generated id. */
  addTask: (task: NewTask) => string;

  /** Replace fields on an existing task. No-op if the id is not found. */
  updateTask: (
    id: string,
    patch: Partial<Omit<Task, "id" | "createdAt">>,
  ) => void;

  /** Remove a task by id. */
  removeTask: (id: string) => void;

  /** Cycle task through todo → in-progress → done → todo */
  toggleTaskStatus: (id: string) => void;

  /** Atomically set a task's status. */
  setTaskStatus: (id: string, status: TaskStatus) => void;

  /** Remove all tasks with status === "done". */
  clearCompleted: () => void;

  /** Reorder tasks array (e.g. after a drag-and-drop). */
  reorderTasks: (orderedIds: string[]) => void;

  // ── Derived selectors (pure functions, no Zustand subscription) ──────────────

  /** Returns tasks matching the given filter, sorted by createdAt desc. */
  getFilteredTasks: (filter: TaskFilter) => Task[];

  /** Returns a single task by id, or undefined. */
  getTaskById: (id: string) => Task | undefined;
}

// ─── Status cycle ─────────────────────────────────────────────────────────────

const STATUS_CYCLE: Record<TaskStatus, TaskStatus> = {
  todo: "in-progress",
  "in-progress": "done",
  done: "todo",
};

// ─── Factory ──────────────────────────────────────────────────────────────────

export const createTasksSlice: StateCreator<
  TasksSlice,
  [["zustand/devtools", never], ["zustand/persist", unknown]],
  [],
  TasksSlice
> = (set, get) => ({
  tasks: [],

  addTask: (task) => {
    const id = nanoid();
    const now = new Date().toISOString();
    const newTask: Task = {
      id,
      status: task.status ?? "todo",
      createdAt: now,
      updatedAt: now,
      tags: task.tags ?? [],
      title: task.title,
      description: task.description,
      priority: task.priority,
      dueDate: task.dueDate,
    };
    set(
      (state) => ({ tasks: [newTask, ...state.tasks] }),
      false,
      "tasks/addTask",
    );
    return id;
  },

  updateTask: (id, patch) => {
    set(
      (state) => ({
        tasks: state.tasks.map((t) =>
          t.id === id
            ? {
                ...t,
                ...patch,
                id,
                createdAt: t.createdAt,
                updatedAt: new Date().toISOString(),
              }
            : t,
        ),
      }),
      false,
      "tasks/updateTask",
    );
  },

  removeTask: (id) => {
    set(
      (state) => ({ tasks: state.tasks.filter((t) => t.id !== id) }),
      false,
      "tasks/removeTask",
    );
  },

  toggleTaskStatus: (id) => {
    set(
      (state) => ({
        tasks: state.tasks.map((t) =>
          t.id === id
            ? {
                ...t,
                status: STATUS_CYCLE[t.status],
                updatedAt: new Date().toISOString(),
              }
            : t,
        ),
      }),
      false,
      "tasks/toggleTaskStatus",
    );
  },

  setTaskStatus: (id, status) => {
    set(
      (state) => ({
        tasks: state.tasks.map((t) =>
          t.id === id
            ? { ...t, status, updatedAt: new Date().toISOString() }
            : t,
        ),
      }),
      false,
      "tasks/setTaskStatus",
    );
  },

  clearCompleted: () => {
    set(
      (state) => ({ tasks: state.tasks.filter((t) => t.status !== "done") }),
      false,
      "tasks/clearCompleted",
    );
  },

  reorderTasks: (orderedIds) => {
    set(
      (state) => {
        const map = new Map(state.tasks.map((t) => [t.id, t]));
        const reordered = orderedIds.flatMap((id) => {
          const task = map.get(id);
          return task ? [task] : [];
        });
        // Append any tasks not in orderedIds (safety guard)
        const includedIds = new Set(orderedIds);
        const remainder = state.tasks.filter((t) => !includedIds.has(t.id));
        return { tasks: [...reordered, ...remainder] };
      },
      false,
      "tasks/reorderTasks",
    );
  },

  // Selectors — read from get() so they always reflect current state.
  getFilteredTasks: (filter) => {
    const { tasks } = get();
    const filtered =
      filter === "all" ? tasks : tasks.filter((t) => t.status === filter);
    return [...filtered].sort(
      (a, b) =>
        new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime(),
    );
  },

  getTaskById: (id) => get().tasks.find((t) => t.id === id),
});
