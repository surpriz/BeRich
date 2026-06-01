"use client";

import { useTranslate } from "@/app/lib/i18n";

// Inline glossary affordances. Definitions live in the i18n catalogs under the
// flat `glossary.<id>` key, so they stay bilingual. CSS-only popover (see the
// .berich-term rules in globals.css) — keyboard reachable + screen-reader safe.

export function Term({ id, children }: { id: string; children?: React.ReactNode }) {
  const t = useTranslate();
  const def = t(`glossary.${id}`);
  const label = children ?? t(`glossary.${id}.label`);
  return (
    <span className="berich-term" tabIndex={0} role="note" aria-label={def}>
      {label}
      <span className="berich-tip" role="tooltip">
        {def}
      </span>
    </span>
  );
}

// Bare "?" badge for placement next to an existing label (e.g. a table header).
export function Info({ id }: { id: string }) {
  const t = useTranslate();
  const def = t(`glossary.${id}`);
  return (
    <span className="berich-term berich-info" tabIndex={0} role="note" aria-label={def}>
      <span aria-hidden="true">?</span>
      <span className="berich-tip" role="tooltip">
        {def}
      </span>
    </span>
  );
}
