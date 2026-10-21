import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterAll, afterEach, beforeAll } from "vitest";

import { server } from "./mocks";

class RO {
  observe() {}
  unobserve() {}
  disconnect() {}
}
// Recharts needs ResizeObserver, which jsdom does not provide.
(globalThis as unknown as { ResizeObserver: typeof RO }).ResizeObserver = RO;

beforeAll(() => server.listen({ onUnhandledRequest: "error" }));
afterEach(() => {
  server.resetHandlers();
  cleanup();
});
afterAll(() => server.close());
