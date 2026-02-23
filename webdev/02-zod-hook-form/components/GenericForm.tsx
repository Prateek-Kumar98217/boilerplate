/**
 * GenericForm — a type-safe, reusable form powered by react-hook-form + Zod.
 *
 * Accepts a Zod schema, a field config array, and an onSubmit handler.
 * No manual register() calls needed in consuming components.
 *
 * Usage:
 *
 *   const LoginSchema = z.object({
 *     email: z.string().email("Invalid email."),
 *     password: z.string().min(8, "At least 8 characters."),
 *   });
 *
 *   <GenericForm
 *     schema={LoginSchema}
 *     fields={[
 *       { name: "email",    label: "Email",    type: "email" },
 *       { name: "password", label: "Password", type: "password" },
 *     ]}
 *     onSubmit={async (data) => {
 *       const result = await signIn(data);
 *       if (!result.success) return result.error; // return string = global error
 *     }}
 *     submitLabel="Sign In"
 *   />
 */
"use client";

import { useState } from "react";
import { useForm, FormProvider } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import type { z } from "zod";
import type { GenericFormProps } from "../types";
import { FormField } from "./FormField";

export function GenericForm<TSchema extends z.ZodTypeAny>({
  schema,
  fields,
  onSubmit,
  defaultValues,
  submitLabel = "Submit",
  submittingLabel = "Submitting…",
  onSuccess,
  className,
}: GenericFormProps<TSchema>) {
  const [globalError, setGlobalError] = useState<string | null>(null);
  const [isSuccess, setIsSuccess] = useState(false);

  const methods = useForm<z.infer<TSchema>>({
    resolver: zodResolver(schema),
    defaultValues: defaultValues as z.infer<TSchema>,
    mode: "onTouched", // validate on blur, then revalidate on change
  });

  const {
    handleSubmit,
    formState: { isSubmitting },
  } = methods;

  const handleFormSubmit = async (data: z.infer<TSchema>) => {
    setGlobalError(null);
    setIsSuccess(false);

    try {
      const result = await onSubmit(data);

      if (typeof result === "string") {
        // Handler returned an error string — show it globally.
        setGlobalError(result);
        return;
      }

      setIsSuccess(true);
      onSuccess?.();
    } catch (err) {
      // Catch unexpected throws from the onSubmit handler.
      const message =
        err instanceof Error
          ? err.message
          : "An unexpected error occurred. Please try again.";
      setGlobalError(message);
    }
  };

  return (
    <FormProvider {...methods}>
      <form
        onSubmit={handleSubmit(handleFormSubmit)}
        noValidate
        className={className}
      >
        {/* Global error banner */}
        {globalError && (
          <div
            role="alert"
            className="mb-4 rounded-md bg-red-50 border border-red-200 px-4 py-3 text-sm text-red-600"
          >
            {globalError}
          </div>
        )}

        {/* Success banner */}
        {isSuccess && (
          <div
            role="status"
            className="mb-4 rounded-md bg-green-50 border border-green-200 px-4 py-3 text-sm text-green-700"
          >
            Submitted successfully.
          </div>
        )}

        {/* Fields */}
        <div className="space-y-4">
          {fields.map((fieldConfig) => (
            <FormField key={fieldConfig.name as string} config={fieldConfig} />
          ))}
        </div>

        {/* Submit button */}
        <button
          type="submit"
          disabled={isSubmitting}
          className="mt-6 w-full rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 transition-colors"
        >
          {isSubmitting ? submittingLabel : submitLabel}
        </button>
      </form>
    </FormProvider>
  );
}
