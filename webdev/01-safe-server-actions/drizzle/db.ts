/**
 * Drizzle ORM + node-postgres database client setup.
 *
 * Environment: DATABASE_URL must be set in .env.local
 * e.g.: postgresql://user:password@localhost:5432/mydb
 *
 * This singleton pattern prevents connection pool exhaustion in Next.js
 * dev mode where modules are hot-reloaded.
 */
import { drizzle } from "drizzle-orm/node-postgres";
import { Pool } from "pg";
import * as schema from "./schema";

// ─── Validate environment variable ───────────────────────────────────────────

const connectionString = process.env.DATABASE_URL;

if (!connectionString) {
  throw new Error(
    "DATABASE_URL environment variable is not set. " +
      "Add it to your .env.local file.",
  );
}

// ─── Singleton pool (prevents hot-reload exhaustion in dev) ──────────────────

declare global {
  // eslint-disable-next-line no-var
  var __pgPool: Pool | undefined;
}

const pool: Pool =
  global.__pgPool ??
  new Pool({
    connectionString,
    max: 10,
    idleTimeoutMillis: 30_000,
    connectionTimeoutMillis: 5_000,
  });

if (process.env.NODE_ENV === "development") {
  global.__pgPool = pool;
}

// ─── Drizzle instance ─────────────────────────────────────────────────────────

export const db = drizzle(pool, { schema });
export type Db = typeof db;
