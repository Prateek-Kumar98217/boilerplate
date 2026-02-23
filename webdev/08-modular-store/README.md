# 08 · Modular Zustand Store

Slice-based Zustand store with DevTools and localStorage persistence.
Two slices — **Tasks** and **UI Settings** — are merged into one typed store.

---

## File map

```
08-modular-store/
├── types.ts                     Task, UISettings, filter types
├── store/
│   ├── tasks-slice.ts           TasksSlice: CRUD, filter, reorder
│   ├── ui-slice.ts              UISlice: theme, sidebar, activeFilter, pageSize
│   └── index.ts                 Merged store with devtools + persist
├── hooks/
│   ├── useTaskStore.ts          Focused task selector hooks
│   └── useUIStore.ts            Focused UI setting hooks
└── example/
    └── TaskList.tsx             Full working component using all hooks
```

---

## Architecture

The **slice pattern** keeps concerns separate while sharing one store:

```ts
// store/index.ts
export const useAppStore = create<AppStore>()(
  devtools(
    persist(
      (...args) => ({
        ...createTasksSlice(...args),
        ...createUISlice(...args),
      }),
      { name: "app-store", version: 1 },
    ),
  ),
);
```

Each slice file exports a `StateCreator` factory:

```ts
export const createTasksSlice: StateCreator<TasksSlice, [...], [], TasksSlice> =
  (set, get) => ({ ... });
```

---

## Usage

### Install

```bash
npm install zustand nanoid
```

### Read data (subscribe to minimal slice)

```tsx
// Re-renders only when tasks change, not when UI settings change
const tasks = useAllTasks();
const { addTask, removeTask } = useTaskActions();
```

### Combined hook for task list pages

```tsx
const { tasks, activeFilter, setActiveTaskFilter } = useTaskList();
```

### UI settings

```tsx
const { toggleTheme } = useUIActions();
const isDark = useIsDarkMode();
```

---

## Selector hooks at a glance

| Hook                       | What it returns                                              |
| -------------------------- | ------------------------------------------------------------ |
| `useAllTasks()`            | All tasks (unsorted)                                         |
| `useFilteredTasks(filter)` | Tasks for a specific status filter, sorted by createdAt desc |
| `useTask(id)`              | Single task or `undefined`                                   |
| `useTaskCounts()`          | `{ todo, "in-progress", done, total }`                       |
| `useTaskList()`            | Filtered tasks + filter setter (reads `ui.activeTaskFilter`) |
| `useTaskActions()`         | All mutation functions — stable refs                         |
| `useTheme()`               | `"light" \| "dark" \| "system"`                              |
| `useSidebarState()`        | `"expanded" \| "collapsed" \| "hidden"`                      |
| `useUIActions()`           | All UI setters — stable refs                                 |

---

## Persistence

Stored in `localStorage["app-store"]` as JSON. Only `tasks[]` and `ui` are
persisted (`partialize`). Version is set to `1`; add a `migrate` function if
the state shape changes.

---

## What can go wrong

| Issue                                    | Cause                                              | Fix                                                                       |
| ---------------------------------------- | -------------------------------------------------- | ------------------------------------------------------------------------- |
| Store not hydrated on first render (SSR) | `localStorage` is not available server-side        | Wrap in `<ClientOnly>` or use `skipHydration` option in persist           |
| DevTools shows anonymous actions         | `set()` was called without an action label         | Pass the label as the third arg: `set(updater, false, "tasks/addTask")`   |
| Selector causes too many re-renders      | Inline object/array returned from selector         | Use `useShallow` from `zustand/shallow` or select primitives individually |
| `nanoid` not found                       | Package not installed                              | `npm install nanoid`                                                      |
| Reorder doesn't work                     | `orderedIds` doesn't include all existing task ids | Remainder tasks are appended automatically as a safety fallback           |
