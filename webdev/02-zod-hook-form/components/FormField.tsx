/**
 * FormField — renders the correct input element based on `config.type`.
 *
 * Must be rendered inside a react-hook-form <FormProvider>.
 * Reads errors and register from the parent form context.
 */
"use client";

import { useFormContext } from "react-hook-form";
import type { FieldValues } from "react-hook-form";
import type { FieldConfig } from "../types";
import type { z } from "zod";

// ─── Utility ──────────────────────────────────────────────────────────────────

function cn(...classes: (string | undefined | false | null)[]) {
  return classes.filter(Boolean).join(" ");
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function FieldError({ message }: { message: string | undefined }) {
  if (!message) return null;
  return (
    <p role="alert" className="text-xs text-red-500 mt-1">
      {message}
    </p>
  );
}

function FieldLabel({
  htmlFor,
  children,
  required,
}: {
  htmlFor: string;
  children: React.ReactNode;
  required?: boolean;
}) {
  return (
    <label
      htmlFor={htmlFor}
      className="block text-sm font-medium text-gray-700 mb-1"
    >
      {children}
      {required && <span className="text-red-500 ml-1">*</span>}
    </label>
  );
}

function FieldDescription({ text }: { text: string | undefined }) {
  if (!text) return null;
  return <p className="text-xs text-gray-500 mt-1">{text}</p>;
}

const inputBaseClasses =
  "w-full rounded-md border border-gray-300 px-3 py-2 text-sm " +
  "focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent " +
  "disabled:cursor-not-allowed disabled:bg-gray-50 disabled:text-gray-500 " +
  "placeholder:text-gray-400";

const errorInputClasses = "border-red-400 focus:ring-red-400";

// ─── Main component ───────────────────────────────────────────────────────────

type FormFieldProps<TSchema extends z.ZodTypeAny> = {
  config: FieldConfig<TSchema>;
};

export function FormField<TSchema extends z.ZodTypeAny>({
  config,
}: FormFieldProps<TSchema>) {
  const {
    register,
    formState: { errors },
  } = useFormContext<FieldValues>();

  // Navigate nested error paths like "address.city"
  const getNestedError = (path: string): string | undefined => {
    const parts = path.split(".");
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let current: Record<string, any> = errors;
    for (const part of parts) {
      if (current === undefined || current === null) return undefined;
      current = current[part] as Record<string, unknown>;
    }
    return typeof current?.message === "string" ? current.message : undefined;
  };

  const errorMessage = getNestedError(config.name as string);
  const inputId = `field-${config.name as string}`;
  const hasError = Boolean(errorMessage);

  const sharedInputClass = cn(
    inputBaseClasses,
    hasError && errorInputClasses,
    config.className,
  );

  // ── text / email / password / url / tel ──────────────────────────────────
  if (
    config.type === "text" ||
    config.type === "email" ||
    config.type === "password" ||
    config.type === "url" ||
    config.type === "tel"
  ) {
    return (
      <div>
        <FieldLabel htmlFor={inputId}>{config.label}</FieldLabel>
        <input
          id={inputId}
          type={config.type}
          placeholder={config.placeholder}
          disabled={config.disabled}
          aria-invalid={hasError}
          aria-describedby={hasError ? `${inputId}-error` : undefined}
          className={sharedInputClass}
          {...register(config.name as string)}
        />
        <FieldDescription text={config.description} />
        <FieldError message={errorMessage} />
      </div>
    );
  }

  // ── number ───────────────────────────────────────────────────────────────
  if (config.type === "number") {
    return (
      <div>
        <FieldLabel htmlFor={inputId}>{config.label}</FieldLabel>
        <input
          id={inputId}
          type="number"
          placeholder={config.placeholder}
          disabled={config.disabled}
          min={config.min}
          max={config.max}
          step={config.step}
          aria-invalid={hasError}
          className={sharedInputClass}
          {...register(config.name as string, { valueAsNumber: true })}
        />
        <FieldDescription text={config.description} />
        <FieldError message={errorMessage} />
      </div>
    );
  }

  // ── textarea ─────────────────────────────────────────────────────────────
  if (config.type === "textarea") {
    return (
      <div>
        <FieldLabel htmlFor={inputId}>{config.label}</FieldLabel>
        <textarea
          id={inputId}
          rows={config.rows ?? 4}
          placeholder={config.placeholder}
          disabled={config.disabled}
          aria-invalid={hasError}
          className={cn(sharedInputClass, "resize-y")}
          {...register(config.name as string)}
        />
        <FieldDescription text={config.description} />
        <FieldError message={errorMessage} />
      </div>
    );
  }

  // ── select ───────────────────────────────────────────────────────────────
  if (config.type === "select") {
    return (
      <div>
        <FieldLabel htmlFor={inputId}>{config.label}</FieldLabel>
        <select
          id={inputId}
          disabled={config.disabled}
          aria-invalid={hasError}
          className={cn(sharedInputClass, "bg-white")}
          {...register(config.name as string)}
        >
          {config.placeholder !== undefined ? (
            <option value="" disabled>
              {config.placeholder || "Select an option…"}
            </option>
          ) : (
            <option value="" disabled>
              Select an option…
            </option>
          )}
          {config.options.map((opt) => (
            <option key={opt.value} value={opt.value} disabled={opt.disabled}>
              {opt.label}
            </option>
          ))}
        </select>
        <FieldDescription text={config.description} />
        <FieldError message={errorMessage} />
      </div>
    );
  }

  // ── checkbox ─────────────────────────────────────────────────────────────
  if (config.type === "checkbox") {
    return (
      <div>
        <div className="flex items-start gap-2">
          <input
            id={inputId}
            type="checkbox"
            disabled={config.disabled}
            aria-invalid={hasError}
            className={cn(
              "mt-0.5 h-4 w-4 rounded border-gray-300 text-blue-600",
              "focus:ring-2 focus:ring-blue-500",
              "disabled:cursor-not-allowed disabled:opacity-50",
              config.className,
            )}
            {...register(config.name as string)}
          />
          <label htmlFor={inputId} className="text-sm text-gray-700">
            {config.checkboxLabel ?? config.label}
          </label>
        </div>
        <FieldDescription text={config.description} />
        <FieldError message={errorMessage} />
      </div>
    );
  }

  // ── radio ────────────────────────────────────────────────────────────────
  if (config.type === "radio") {
    return (
      <fieldset>
        <legend className="block text-sm font-medium text-gray-700 mb-1">
          {config.label}
        </legend>
        <div className="space-y-2">
          {config.options.map((opt) => {
            const radioId = `${inputId}-${opt.value}`;
            return (
              <div key={opt.value} className="flex items-center gap-2">
                <input
                  id={radioId}
                  type="radio"
                  value={opt.value}
                  disabled={config.disabled ?? opt.disabled}
                  className={cn(
                    "h-4 w-4 border-gray-300 text-blue-600",
                    "focus:ring-2 focus:ring-blue-500",
                    config.className,
                  )}
                  {...register(config.name as string)}
                />
                <label htmlFor={radioId} className="text-sm text-gray-700">
                  {opt.label}
                </label>
              </div>
            );
          })}
        </div>
        <FieldDescription text={config.description} />
        <FieldError message={errorMessage} />
      </fieldset>
    );
  }

  // TypeScript exhaustive check — this line should never be reached.
  const _exhaustive: never = config;
  console.error("Unknown field type encountered:", _exhaustive);
  return null;
}
