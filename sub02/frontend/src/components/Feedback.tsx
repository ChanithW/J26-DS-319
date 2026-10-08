import type { ReactNode } from "react";

export function ErrorNotice({ title = "Something went wrong", message, onRetry }: {
  title?: string;
  message: string;
  onRetry?: () => void;
}) {
  return (
    <div role="alert" className="rounded-md border border-red-300 bg-red-50 p-3 text-sm text-red-900 dark:border-red-800 dark:bg-red-950 dark:text-red-100">
      <p className="font-semibold">{title}</p>
      <p className="mt-1">{message}</p>
      {onRetry && (
        <button type="button" className="btn mt-2" onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  );
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="card text-center">
      <p className="text-base font-semibold">{title}</p>
      {children && <div className="mt-2 text-sm text-slate-600 dark:text-slate-400">{children}</div>}
    </div>
  );
}

export function Skeleton({ lines = 3 }: { lines?: number }) {
  return (
    <div aria-busy="true" aria-label="Loading" className="space-y-2">
      {Array.from({ length: lines }, (_, i) => (
        <div key={i} className="h-5 animate-pulse rounded bg-slate-200 dark:bg-slate-800" style={{ width: `${90 - i * 12}%` }} />
      ))}
    </div>
  );
}
