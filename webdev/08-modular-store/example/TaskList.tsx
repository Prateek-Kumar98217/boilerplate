/**
 * TaskList.tsx — example component using the modular Zustand store.
 *
 * Demonstrates:
 *  - useTaskList() → filtered tasks + filter setter
 *  - useTaskActions() → CRUD actions
 *  - useTaskCounts() → badge counters
 *  - useUIActions().toggleTheme() → theme switcher
 *  - useIsDarkMode() → conditional class
 */
"use client";

import React, { useState } from "react";
import {
  useTaskList,
  useTaskActions,
  useTaskCounts,
  useHasCompletedTasks,
} from "../hooks/useTaskStore";
import { useUIActions, useIsDarkMode } from "../hooks/useUIStore";
import type { TaskFilter, TaskPriority } from "../types";

const FILTERS: TaskFilter[] = ["all", "todo", "in-progress", "done"];
const PRIORITIES: TaskPriority[] = ["low", "medium", "high"];

const PRIORITY_CLASSES: Record<TaskPriority, string> = {
  low: "bg-blue-100 text-blue-800",
  medium: "bg-yellow-100 text-yellow-800",
  high: "bg-red-100 text-red-800",
};

// ─── Add Task Form ─────────────────────────────────────────────────────────────

function AddTaskForm() {
  const { addTask } = useTaskActions();
  const [title, setTitle] = useState("");
  const [priority, setPriority] = useState<TaskPriority>("medium");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = title.trim();
    if (!trimmed) return;
    addTask({ title: trimmed, priority, tags: [] });
    setTitle("");
  };

  return (
    <form onSubmit={handleSubmit} className="flex gap-2 mb-4">
      <input
        type="text"
        value={title}
        onChange={(e) => setTitle(e.target.value)}
        placeholder="New task…"
        className="flex-1 border rounded px-3 py-2 text-sm"
        aria-label="New task title"
      />
      <select
        value={priority}
        onChange={(e) => setPriority(e.target.value as TaskPriority)}
        className="border rounded px-2 py-2 text-sm"
        aria-label="Task priority"
      >
        {PRIORITIES.map((p) => (
          <option key={p} value={p}>
            {p}
          </option>
        ))}
      </select>
      <button
        type="submit"
        disabled={!title.trim()}
        className="bg-indigo-600 text-white px-4 py-2 rounded text-sm disabled:opacity-50"
      >
        Add
      </button>
    </form>
  );
}

// ─── Task Item ─────────────────────────────────────────────────────────────────

function TaskItem({
  id,
  title,
  status,
  priority,
}: {
  id: string;
  title: string;
  status: string;
  priority: TaskPriority;
}) {
  const { removeTask, toggleTaskStatus } = useTaskActions();

  return (
    <li className="flex items-center justify-between gap-2 py-2 border-b last:border-0">
      <button
        onClick={() => toggleTaskStatus(id)}
        className="text-left flex-1 text-sm hover:text-indigo-600"
        aria-label={`Toggle status of "${title}"`}
      >
        <span className={status === "done" ? "line-through text-gray-400" : ""}>
          {title}
        </span>
      </button>
      <span
        className={`text-xs px-2 py-0.5 rounded-full font-medium ${PRIORITY_CLASSES[priority]}`}
      >
        {priority}
      </span>
      <span className="text-xs text-gray-500 w-20 text-center">{status}</span>
      <button
        onClick={() => removeTask(id)}
        aria-label={`Delete "${title}"`}
        className="text-red-400 hover:text-red-600 text-xs px-1"
      >
        ✕
      </button>
    </li>
  );
}

// ─── Task List ─────────────────────────────────────────────────────────────────

export function TaskList() {
  const { tasks, activeFilter, setActiveTaskFilter } = useTaskList();
  const { clearCompleted } = useTaskActions();
  const counts = useTaskCounts();
  const hasCompleted = useHasCompletedTasks();
  const { toggleTheme } = useUIActions();
  const isDark = useIsDarkMode();

  return (
    <div
      className={`max-w-lg mx-auto p-4 ${isDark ? "bg-gray-900 text-white" : "bg-white text-gray-900"}`}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-xl font-semibold">Tasks ({counts.total})</h1>
        <button
          onClick={toggleTheme}
          className="text-xs border px-2 py-1 rounded"
          aria-label="Toggle theme"
        >
          {isDark ? "☀ Light" : "☾ Dark"}
        </button>
      </div>

      {/* Add form */}
      <AddTaskForm />

      {/* Filter tabs */}
      <div className="flex gap-1 mb-3 border-b">
        {FILTERS.map((f) => (
          <button
            key={f}
            onClick={() => setActiveTaskFilter(f)}
            className={`px-3 py-1 text-sm rounded-t ${
              activeFilter === f
                ? "bg-indigo-600 text-white"
                : "text-gray-500 hover:text-gray-900"
            }`}
          >
            {f}{" "}
            {f !== "all" && (
              <span className="ml-1 text-xs opacity-70">
                ({counts[f as keyof typeof counts] ?? ""})
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Task list */}
      {tasks.length === 0 ? (
        <p className="text-sm text-gray-400 text-center py-6">No tasks here.</p>
      ) : (
        <ul>
          {tasks.map((t) => (
            <TaskItem
              key={t.id}
              id={t.id}
              title={t.title}
              status={t.status}
              priority={t.priority}
            />
          ))}
        </ul>
      )}

      {/* Clear completed */}
      {hasCompleted && (
        <button
          onClick={clearCompleted}
          className="mt-3 text-xs text-red-500 hover:text-red-700"
        >
          Clear completed ({counts.done})
        </button>
      )}
    </div>
  );
}
