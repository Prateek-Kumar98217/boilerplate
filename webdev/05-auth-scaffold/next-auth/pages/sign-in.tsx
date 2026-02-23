/**
 * Sign-in page — app/auth/sign-in/page.tsx
 *
 * Supports:
 * - OAuth providers (GitHub, Google) via server action
 * - Credentials form with validation
 * - Redirects back to the page the user came from via callbackUrl
 */
import { signIn, auth } from "../auth";
import { redirect } from "next/navigation";
import { AuthError } from "next-auth";

// ─── Server actions ───────────────────────────────────────────────────────────

async function signInWithProvider(provider: string, callbackUrl: string) {
  "use server";
  await signIn(provider, { redirectTo: callbackUrl });
}

async function signInWithCredentials(formData: FormData) {
  "use server";
  const callbackUrl =
    (formData.get("callbackUrl") as string | null) ?? "/dashboard";

  try {
    await signIn("credentials", {
      email: formData.get("email"),
      password: formData.get("password"),
      redirectTo: callbackUrl,
    });
  } catch (err) {
    if (err instanceof AuthError) {
      switch (err.type) {
        case "CredentialsSignin":
          redirect(
            `/auth/sign-in?error=invalid_credentials&callbackUrl=${encodeURIComponent(callbackUrl)}`,
          );
        default:
          redirect(
            `/auth/sign-in?error=unknown&callbackUrl=${encodeURIComponent(callbackUrl)}`,
          );
      }
    }
    // Re-throw redirect errors so Next.js can handle them.
    throw err;
  }
}

// ─── Error messages ───────────────────────────────────────────────────────────

const ERROR_MESSAGES: Record<string, string> = {
  invalid_credentials: "Invalid email or password. Please try again.",
  unknown: "Something went wrong. Please try again.",
  OAuthAccountNotLinked: "This email is linked to a different sign-in method.",
};

// ─── Page ─────────────────────────────────────────────────────────────────────

type SignInPageProps = {
  searchParams: Promise<{ error?: string; callbackUrl?: string }>;
};

export default async function SignInPage({ searchParams }: SignInPageProps) {
  const session = await auth();
  const { error, callbackUrl = "/dashboard" } = await searchParams;

  // Already signed in — redirect.
  if (session?.user?.id) redirect(callbackUrl);

  const errorMessage = error
    ? (ERROR_MESSAGES[error] ?? ERROR_MESSAGES["unknown"])
    : null;

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 px-4">
      <div className="w-full max-w-sm bg-white rounded-xl shadow-sm border p-8 space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Sign in</h1>
          <p className="mt-1 text-sm text-gray-500">Welcome back</p>
        </div>

        {errorMessage && (
          <div
            role="alert"
            className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-md p-3"
          >
            {errorMessage}
          </div>
        )}

        {/* OAuth Providers */}
        <div className="space-y-3">
          <form action={signInWithProvider.bind(null, "github", callbackUrl)}>
            <button
              type="submit"
              className="w-full flex items-center justify-center gap-2 border border-gray-300 rounded-lg px-4 py-2.5 text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors"
            >
              Continue with GitHub
            </button>
          </form>
          <form action={signInWithProvider.bind(null, "google", callbackUrl)}>
            <button
              type="submit"
              className="w-full flex items-center justify-center gap-2 border border-gray-300 rounded-lg px-4 py-2.5 text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors"
            >
              Continue with Google
            </button>
          </form>
        </div>

        <div className="relative">
          <div className="absolute inset-0 flex items-center">
            <div className="w-full border-t border-gray-200" />
          </div>
          <div className="relative flex justify-center text-xs text-gray-500">
            <span className="bg-white px-2">or continue with email</span>
          </div>
        </div>

        {/* Credentials form */}
        <form action={signInWithCredentials} className="space-y-4">
          <input type="hidden" name="callbackUrl" value={callbackUrl} />

          <div>
            <label
              htmlFor="email"
              className="block text-sm font-medium text-gray-700 mb-1"
            >
              Email
            </label>
            <input
              id="email"
              name="email"
              type="email"
              autoComplete="email"
              required
              className="w-full border border-gray-300 rounded-lg px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="you@example.com"
            />
          </div>

          <div>
            <label
              htmlFor="password"
              className="block text-sm font-medium text-gray-700 mb-1"
            >
              Password
            </label>
            <input
              id="password"
              name="password"
              type="password"
              autoComplete="current-password"
              required
              className="w-full border border-gray-300 rounded-lg px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Enter your password"
            />
          </div>

          <button
            type="submit"
            className="w-full bg-blue-600 text-white rounded-lg px-4 py-2.5 text-sm font-medium hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors"
          >
            Sign in
          </button>
        </form>
      </div>
    </div>
  );
}
