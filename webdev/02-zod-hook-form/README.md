# 02 — Zod + React Hook Form

A fully typed, reusable form system built on **react-hook-form** and **Zod**.

- One `GenericForm` component accepts a Zod schema + a field config array — zero repeated `register()` calls.
- `FormField` renders the correct input variant (text, email, password, number, textarea, select, checkbox, radio) and wires up all error messages.
- `useZodForm` hook for when you need direct access to form methods.

---

## Files

| File                         | Purpose                                                         |
| ---------------------------- | --------------------------------------------------------------- |
| `types/index.ts`             | `FieldConfig<TSchema>` discriminated union + `GenericFormProps` |
| `components/FormField.tsx`   | Renders the right input for each `FieldConfig.type`             |
| `components/GenericForm.tsx` | `FormProvider` wrapper + submit handler + error display         |
| `components/index.ts`        | Barrel export                                                   |
| `hooks/useZodForm.ts`        | Low-level hook (pre-wires `zodResolver`)                        |
| `example/ExampleForms.tsx`   | Full registration form + minimal login form                     |

---

## Flow

```
<GenericForm schema={…} fields={[…]} onSubmit={…}>
    │
    ├─ useForm({ resolver: zodResolver(schema) })   — react-hook-form
    ├─ FormProvider (propagates form context to all FormField children)
    │
    ▼
<FormField config={fieldConfig}>
    │
    ├─ switch (config.type) → renders <input> / <textarea> / <select> etc.
    ├─ useFormContext()      → reads register, errors
    └─ wires aria-invalid, aria-describedby for accessibility
    │
    ▼
handleSubmit(onSubmit)
    │
    ├─ Zod validates synchronously — per-field errors shown inline
    ├─ onSubmit(data) called with fully typed, validated data
    └─ Return string from onSubmit → shown as global error banner
```

---

## Usage

```tsx
import { z } from "zod";
import { GenericForm } from "./components";
import type { FieldConfig } from "./components";

const LoginSchema = z.object({
  email: z.string().email("Enter a valid email."),
  password: z.string().min(8, "At least 8 characters."),
});

const fields: FieldConfig<typeof LoginSchema>[] = [
  {
    name: "email",
    label: "Email",
    type: "email",
    placeholder: "you@example.com",
  },
  { name: "password", label: "Password", type: "password" },
];

export function LoginForm() {
  return (
    <GenericForm
      schema={LoginSchema}
      fields={fields}
      onSubmit={async (data) => {
        const result = await signIn(data);
        if (!result.ok) return result.message; // string → global error
      }}
      submitLabel="Sign In"
    />
  );
}
```

### Using `useZodForm` directly

```tsx
import { useZodForm } from "./hooks/useZodForm";
import { FormProvider } from "react-hook-form";
import { FormField } from "./components/FormField";

const form = useZodForm({ schema: MySchema });

<FormProvider {...form}>
  <form onSubmit={form.handleSubmit(onSubmit)}>
    <FormField config={{ name: "title", label: "Title", type: "text" }} />
  </form>
</FormProvider>;
```

---

## Supported Field Types

| `type`     | Renders                      | Extra options                            |
| ---------- | ---------------------------- | ---------------------------------------- |
| `text`     | `<input type="text">`        | `placeholder`                            |
| `email`    | `<input type="email">`       | `placeholder`                            |
| `password` | `<input type="password">`    | –                                        |
| `url`      | `<input type="url">`         | `placeholder`                            |
| `tel`      | `<input type="tel">`         | `placeholder`                            |
| `number`   | `<input type="number">`      | `min`, `max`, `step`                     |
| `textarea` | `<textarea>`                 | `rows`                                   |
| `select`   | `<select>`                   | `options: SelectOption[]`, `placeholder` |
| `checkbox` | `<input type="checkbox">`    | `checkboxLabel`                          |
| `radio`    | `<input type="radio">` group | `options: SelectOption[]`                |

---

## What Can Go Wrong

| Scenario                              | How it's handled                                                                                                                                          |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Field name not in schema              | TypeScript error at compile time — `name` is `Path<z.infer<TSchema>>`                                                                                     |
| Nested field paths (`"address.city"`) | `getNestedError` walks the error object by splitting on `.`                                                                                               |
| `number` field coercion               | `register` uses `{ valueAsNumber: true }` — Zod `z.number()` receives a number, not a string                                                              |
| `onSubmit` throws unexpectedly        | Caught in `handleFormSubmit`, displayed as global error                                                                                                   |
| `onSubmit` returns a string           | Shown as global error banner without resetting form state                                                                                                 |
| Multiple field types on the same form | Each variant is a separate discriminated union branch — TypeScript ensures correct extra props                                                            |
| `any` type usage                      | None — `useFormContext<FieldValues>()` is used; nested error traversal uses `Record<string, any>` internally but is isolated to a single utility function |

---

## Installation

```bash
npm install react-hook-form @hookform/resolvers zod
```
