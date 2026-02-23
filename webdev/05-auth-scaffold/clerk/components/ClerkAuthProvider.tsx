/**
 * Clerk root provider — place in your root layout.
 *
 * Wraps your app in ClerkProvider so all Client/Server Components
 * can access auth state via the Clerk hooks and helpers.
 *
 * Usage in app/layout.tsx:
 *
 *   import { ClerkAuthProvider } from "@/components/ClerkAuthProvider";
 *
 *   export default function RootLayout({ children }) {
 *     return (
 *       <html>
 *         <body>
 *           <ClerkAuthProvider>{children}</ClerkAuthProvider>
 *         </body>
 *       </html>
 *     );
 *   }
 */
import { ClerkProvider } from "@clerk/nextjs";
import type { ReactNode } from "react";

type ClerkAuthProviderProps = {
  children: ReactNode;
};

export function ClerkAuthProvider({ children }: ClerkAuthProviderProps) {
  return (
    <ClerkProvider
      appearance={{
        // Customize Clerk's built-in UI components.
        // See https://clerk.com/docs/customization/overview
        variables: {
          colorPrimary: "#2563eb", // blue-600
        },
        elements: {
          formButtonPrimary:
            "bg-blue-600 hover:bg-blue-700 text-sm normal-case font-medium",
        },
      }}
    >
      {children}
    </ClerkProvider>
  );
}
