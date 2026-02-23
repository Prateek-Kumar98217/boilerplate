/**
 * Example Client Component — calling a safe server action.
 *
 * Uses React 19 useActionState. For React 18, replace with a plain
 * async onClick handler or useTransition.
 */
"use client";

import { useActionState } from "react";
import { createPost } from "./example-actions";
import type { ActionResult } from "./types";
import type { Post } from "./schema";

// ─── Initial state ────────────────────────────────────────────────────────────

const INITIAL_STATE: ActionResult<Post> = {
  success: false,
  error: "",
  code: "INTERNAL_ERROR",
};

// ─── Component ────────────────────────────────────────────────────────────────

export function CreatePostForm() {
  const [state, formAction, isPending] = useActionState(
    async (_prevState: ActionResult<Post>, formData: FormData) => {
      return createPost({
        title: formData.get("title"),
        content: formData.get("content"),
      });
    },
    INITIAL_STATE,
  );

  return (
    <form action={formAction} className="flex flex-col gap-4 max-w-lg">
      <h2 className="text-xl font-semibold">Create Post</h2>

      {/* Global error */}
      {!state.success && state.error && (
        <p className="text-sm text-red-600 bg-red-50 p-3 rounded">
          {state.error}
        </p>
      )}

      {/* Success */}
      {state.success && (
        <p className="text-sm text-green-600 bg-green-50 p-3 rounded">
          Post "{state.data.title}" created!
        </p>
      )}

      {/* Title */}
      <div className="flex flex-col gap-1">
        <label htmlFor="title" className="text-sm font-medium">
          Title
        </label>
        <input
          id="title"
          name="title"
          type="text"
          className="border rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          disabled={isPending}
        />
        {!state.success && state.fieldErrors?.title && (
          <p className="text-xs text-red-500">{state.fieldErrors.title[0]}</p>
        )}
      </div>

      {/* Content */}
      <div className="flex flex-col gap-1">
        <label htmlFor="content" className="text-sm font-medium">
          Content
        </label>
        <textarea
          id="content"
          name="content"
          rows={5}
          className="border rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
          disabled={isPending}
        />
        {!state.success && state.fieldErrors?.content && (
          <p className="text-xs text-red-500">{state.fieldErrors.content[0]}</p>
        )}
      </div>

      <button
        type="submit"
        disabled={isPending}
        className="bg-blue-600 text-white rounded px-4 py-2 text-sm font-medium hover:bg-blue-700 disabled:opacity-50 transition-colors"
      >
        {isPending ? "Creating…" : "Create Post"}
      </button>
    </form>
  );
}
