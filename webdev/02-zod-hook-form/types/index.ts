/**
 * Type definitions for the generic form system.
 *
 * FieldConfig<TSchema> is a discriminated union on `type`. This ensures
 * that type-specific options (e.g. `options` for select) are only available
 * on the correct variant, and that `name` is always a valid key of TSchema.
 */
import type { Path } from "react-hook-form";
import type { z } from "zod";

// ─── Shared field base ────────────────────────────────────────────────────────

type BaseFieldConfig<TSchema extends z.ZodTypeAny> = {
  /** Must be a key path in the Zod schema's inferred type. */
  name: Path<z.infer<TSchema>>;
  label: string;
  description?: string;
  placeholder?: string;
  disabled?: boolean;
  className?: string;
};

// ─── Discriminated variants ───────────────────────────────────────────────────

export type TextFieldConfig<TSchema extends z.ZodTypeAny> =
  BaseFieldConfig<TSchema> & {
    type: "text" | "email" | "password" | "url" | "tel";
  };

export type NumberFieldConfig<TSchema extends z.ZodTypeAny> =
  BaseFieldConfig<TSchema> & {
    type: "number";
    min?: number;
    max?: number;
    step?: number;
  };

export type TextareaFieldConfig<TSchema extends z.ZodTypeAny> =
  BaseFieldConfig<TSchema> & {
    type: "textarea";
    rows?: number;
  };

export type SelectOption = {
  value: string;
  label: string;
  disabled?: boolean;
};

export type SelectFieldConfig<TSchema extends z.ZodTypeAny> =
  BaseFieldConfig<TSchema> & {
    type: "select";
    options: SelectOption[];
    /** Show an empty "choose…" option at the top. Default: true. */
    placeholder?: string;
  };

export type CheckboxFieldConfig<TSchema extends z.ZodTypeAny> =
  BaseFieldConfig<TSchema> & {
    type: "checkbox";
    /** Label shown next to the checkbox (separate from the group label). */
    checkboxLabel?: string;
  };

export type RadioFieldConfig<TSchema extends z.ZodTypeAny> =
  BaseFieldConfig<TSchema> & {
    type: "radio";
    options: SelectOption[];
  };

// ─── Union ────────────────────────────────────────────────────────────────────

export type FieldConfig<TSchema extends z.ZodTypeAny> =
  | TextFieldConfig<TSchema>
  | NumberFieldConfig<TSchema>
  | TextareaFieldConfig<TSchema>
  | SelectFieldConfig<TSchema>
  | CheckboxFieldConfig<TSchema>
  | RadioFieldConfig<TSchema>;

// ─── Generic form props ───────────────────────────────────────────────────────

export type GenericFormProps<TSchema extends z.ZodTypeAny> = {
  schema: TSchema;
  fields: FieldConfig<TSchema>[];
  /**
   * Called with the validated, typed form data.
   * Return a string to display as a global error.
   * Return void/undefined on success.
   */
  onSubmit: (data: z.infer<TSchema>) => Promise<string | void> | string | void;
  defaultValues?: Partial<z.infer<TSchema>>;
  submitLabel?: string;
  /** Shown when the form is submitting. Default: "Submitting…" */
  submittingLabel?: string;
  /** Called after a successful submission. */
  onSuccess?: () => void;
  className?: string;
};
