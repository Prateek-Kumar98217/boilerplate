/**
 * Supabase auto-generated types stub.
 *
 * Generate the real file with:
 *   npx supabase gen types typescript --project-id <your-project-id> > database.types.ts
 *
 * The stub below is the minimum required shape so TypeScript is satisfied
 * before you run the generator.
 */

export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[];

export type Database = {
  public: {
    Tables: Record<
      string,
      {
        Row: Record<string, Json>;
        Insert: Record<string, Json>;
        Update: Partial<Record<string, Json>>;
        Relationships: unknown[];
      }
    >;
    Views: Record<string, never>;
    Functions: Record<string, never>;
    Enums: Record<string, never>;
    CompositeTypes: Record<string, never>;
  };
};
