/**
 * SessionProvider wrapper — place in your root layout.
 *
 * Wraps next-auth/react's SessionProvider so Client Components
 * (and hooks like useAuth, useUser) can access the session.
 *
 * Usage in app/layout.tsx:
 *
 *   import { AuthSessionProvider } from "@/components/AuthSessionProvider";
 *
 *   export default async function RootLayout({ children }) {
 *     const session = await auth();
 *     return (
 *       <html>
 *         <body>
 *           <AuthSessionProvider session={session}>
 *             {children}
 *           </AuthSessionProvider>
 *         </body>
 *       </html>
 *     );
 *   }
 */
"use client";

import { SessionProvider } from "next-auth/react";
import type { Session } from "next-auth";
import type { ReactNode } from "react";

type AuthSessionProviderProps = {
  children: ReactNode;
  session: Session | null;
};

export function AuthSessionProvider({
  children,
  session,
}: AuthSessionProviderProps) {
  return <SessionProvider session={session}>{children}</SessionProvider>;
}
