import { Link } from "react-router-dom";

/** Always visible, not dismissible. Shown on every page. */
export function LimitationsBanner() {
  return (
    <div
      role="note"
      aria-label="Limitations"
      data-testid="limitations-banner"
      className="border-b border-amber-300 bg-amber-50 px-4 py-2 text-sm text-amber-950 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-100"
    >
      <p className="mx-auto max-w-6xl">
        <strong>Trained on solar flares.</strong> Accuracy on stars has not been validated. The probability is a ranking
        score, not a chance of a CME.{" "}
        <Link to="/model" className="font-medium underline">
          Read the limitations
        </Link>
      </p>
    </div>
  );
}
