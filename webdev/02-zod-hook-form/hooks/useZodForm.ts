/**
 * useZodForm — thin wrapper around useForm that pre-wires zodResolver.
 *
 * Prefer GenericForm for most cases. Use this hook when you need direct
 * access to the form methods (setValue, watch, reset, etc.) while still
 * benefiting from Zod validation.
 *
 * Usage:
 *
 *   const form = useZodForm({ schema: MySchema, defaultValues: { name: "" } });
 *   <FormProvider {...form}>
 *     <form onSubmit={form.handleSubmit(onSubmit)}>
 *       <FormField config={{ name: "name", label: "Name", type: "text" }} />
 *     </form>
 *   </FormProvider>
 */
"use client";

import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import type { UseFormProps } from "react-hook-form";
import type { z } from "zod";

type UseZodFormOptions<TSchema extends z.ZodTypeAny> = Omit<
  UseFormProps<z.infer<TSchema>>,
  "resolver"
> & {
  schema: TSchema;
};

export function useZodForm<TSchema extends z.ZodTypeAny>({
  schema,
  mode = "onTouched",
  ...rest
}: UseZodFormOptions<TSchema>) {
  return useForm<z.infer<TSchema>>({
    resolver: zodResolver(schema),
    mode,
    ...rest,
  });
}
