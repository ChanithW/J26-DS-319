import { NavLink, Outlet, Route, Routes } from "react-router-dom";

import { LimitationsBanner } from "./components/LimitationsBanner";
import { ModelInfoPage } from "./pages/ModelInfoPage";
import { PredictPage } from "./pages/PredictPage";
import { RunPage } from "./pages/RunPage";
import { RunsPage } from "./pages/RunsPage";

const link = ({ isActive }: { isActive: boolean }) =>
  `rounded-md px-3 py-1.5 text-sm font-medium ${isActive ? "bg-slate-200 dark:bg-slate-800" : "hover:bg-slate-100 dark:hover:bg-slate-900"}`;

function Layout() {
  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-2 px-4 py-3">
          <p className="text-base font-semibold">Flare CME predictor</p>
          <nav aria-label="Main" className="flex gap-1">
            <NavLink to="/" end className={link}>Predict</NavLink>
            <NavLink to="/runs" className={link}>Runs</NavLink>
            <NavLink to="/model" className={link}>Model info</NavLink>
          </nav>
        </div>
      </header>
      <div className="sticky top-0 z-30">
        <LimitationsBanner />
      </div>
      <main className="mx-auto max-w-6xl px-4 py-6">
        <Outlet />
      </main>
    </div>
  );
}

/** Routes only (providers live in main.tsx), so pages can be tested in a memory router. */
export function AppRoutes() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<PredictPage />} />
        <Route path="runs" element={<RunsPage />} />
        <Route path="runs/:runId" element={<RunPage />} />
        <Route path="model" element={<ModelInfoPage />} />
        <Route path="*" element={<p>Page not found.</p>} />
      </Route>
    </Routes>
  );
}
