/**
 * Example: Registration form showcasing all field types.
 *
 * Demonstrates:
 * - Text, email, password, number, textarea, select, checkbox, radio
 * - Complex Zod validation (cross-field, custom messages)
 * - onSubmit returning an error string for server-side errors
 */
"use client";

import { z } from "zod";
import { GenericForm } from "../components/GenericForm";
import type { FieldConfig } from "../types";

// ─── Schema ───────────────────────────────────────────────────────────────────

const RegistrationSchema = z
  .object({
    name: z.string().min(2, "Name must be at least 2 characters."),
    email: z.string().email("Enter a valid email address."),
    password: z
      .string()
      .min(8, "Password must be at least 8 characters.")
      .regex(/[A-Z]/, "Must contain at least one uppercase letter.")
      .regex(/[0-9]/, "Must contain at least one number."),
    confirmPassword: z.string(),
    age: z
      .number({ invalid_type_error: "Age must be a number." })
      .int("Age must be a whole number.")
      .min(13, "You must be at least 13.")
      .max(120, "Invalid age."),
    bio: z.string().max(500, "Bio cannot exceed 500 characters.").optional(),
    role: z.enum(["developer", "designer", "manager"], {
      required_error: "Please select a role.",
    }),
    experience: z.enum(["junior", "mid", "senior"], {
      required_error: "Please select your experience level.",
    }),
    acceptTerms: z.literal(true, {
      errorMap: () => ({ message: "You must accept the terms." }),
    }),
  })
  .refine((data) => data.password === data.confirmPassword, {
    message: "Passwords do not match.",
    path: ["confirmPassword"],
  });

type RegistrationInput = z.infer<typeof RegistrationSchema>;

// ─── Field config ─────────────────────────────────────────────────────────────

const fields: FieldConfig<typeof RegistrationSchema>[] = [
  {
    name: "name",
    label: "Full Name",
    type: "text",
    placeholder: "Jane Doe",
  },
  {
    name: "email",
    label: "Email Address",
    type: "email",
    placeholder: "jane@example.com",
  },
  {
    name: "password",
    label: "Password",
    type: "password",
    placeholder: "Min. 8 chars, 1 uppercase, 1 number",
  },
  {
    name: "confirmPassword",
    label: "Confirm Password",
    type: "password",
  },
  {
    name: "age",
    label: "Age",
    type: "number",
    min: 13,
    max: 120,
  },
  {
    name: "bio",
    label: "Short Bio",
    type: "textarea",
    placeholder: "Tell us about yourself…",
    rows: 3,
    description: "Optional. Max 500 characters.",
  },
  {
    name: "role",
    label: "Role",
    type: "select",
    placeholder: "Select your role…",
    options: [
      { value: "developer", label: "Developer" },
      { value: "designer", label: "Designer" },
      { value: "manager", label: "Manager" },
    ],
  },
  {
    name: "experience",
    label: "Experience Level",
    type: "radio",
    options: [
      { value: "junior", label: "Junior (0–2 years)" },
      { value: "mid", label: "Mid (2–5 years)" },
      { value: "senior", label: "Senior (5+ years)" },
    ],
  },
  {
    name: "acceptTerms",
    label: "Terms",
    type: "checkbox",
    checkboxLabel: "I accept the Terms of Service and Privacy Policy.",
  },
];

// ─── Component ────────────────────────────────────────────────────────────────

export function RegistrationForm() {
  const handleSubmit = async (
    data: RegistrationInput,
  ): Promise<string | void> => {
    // Simulate a server request.
    await new Promise<void>((resolve) => setTimeout(resolve, 1000));

    // Example: server says the email is already taken.
    if (data.email === "taken@example.com") {
      return "This email address is already in use.";
    }

    console.log("Registered:", data);
    // Return nothing on success.
  };

  return (
    <div className="max-w-lg mx-auto p-8">
      <h1 className="text-2xl font-bold mb-6">Create an Account</h1>
      <GenericForm
        schema={RegistrationSchema}
        fields={fields}
        onSubmit={handleSubmit}
        submitLabel="Register"
        submittingLabel="Creating account…"
        onSuccess={() => {
          alert("Account created! Redirecting…");
        }}
        defaultValues={{ bio: "" }}
        className="space-y-4"
      />
    </div>
  );
}

// ─── Minimal login form example ───────────────────────────────────────────────

const LoginSchema = z.object({
  email: z.string().email("Enter a valid email."),
  password: z.string().min(1, "Password is required."),
  rememberMe: z.boolean().optional(),
});

const loginFields: FieldConfig<typeof LoginSchema>[] = [
  {
    name: "email",
    label: "Email",
    type: "email",
    placeholder: "you@example.com",
  },
  { name: "password", label: "Password", type: "password" },
  {
    name: "rememberMe",
    label: "Remember me",
    type: "checkbox",
    checkboxLabel: "Keep me signed in",
  },
];

export function LoginForm() {
  return (
    <div className="max-w-sm mx-auto p-8">
      <h1 className="text-2xl font-bold mb-6">Sign In</h1>
      <GenericForm
        schema={LoginSchema}
        fields={loginFields}
        onSubmit={async (data) => {
          console.log("Login:", data);
        }}
        submitLabel="Sign In"
        className="space-y-4"
      />
    </div>
  );
}
