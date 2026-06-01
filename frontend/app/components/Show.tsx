"use client";

import { atLeast, useLevel, type Level } from "@/app/lib/level";

// Declarative level gate. `<Show min="expert">` renders children only at that
// level or above; `<Show max="standard">` renders only at that level or below.
export function Show({
  min,
  max,
  children,
}: {
  min?: Level;
  max?: Level;
  children: React.ReactNode;
}) {
  const { level } = useLevel();
  if (min && !atLeast(level, min)) return null;
  if (max && !atLeast(max, level)) return null;
  return <>{children}</>;
}
