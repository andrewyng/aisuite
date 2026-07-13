# E2E tests (Playwright)

End-to-end regression tests for the GUI. They drive the real app in Chromium but are **hermetic**:
every `/v1` request and the event WebSocket are mocked at the network layer, so tests need **no
Python backend**, run deterministically, and never mutate real state.

## Run

```bash
npm run e2e          # headless
npm run e2e:ui       # Playwright UI mode (watch/inspect)
npx playwright test e2e/settings.spec.ts   # a single spec
```

## Live smoke (not CI)

`npm run e2e:live` runs `e2e-live/` (separate `playwright.live.config.ts`) against the **real**
backend + a **real** model: it asks a fresh Cowork session to produce `fib.md` and verifies the file
lands on disk. It needs `coworker-server` up on :8765 with a model configured (skips cleanly
otherwise), is nondeterministic, and costs a few tokens per run — so it lives in its own dir/config
and the default `e2e` (and CI) never picks it up. It exercises the vertical the specs below mock:
model wiring, the tool/approval loop, file I/O, and WebSocket event streaming.

The config (`playwright.config.ts`) starts the Vite dev server on port **5199** (dedicated, so it
won't clash with a running `npm run dev` on 5173) and reuses it if already up.

## How the mock works

`e2e/fixtures.ts` exports a `test` whose `page` has `mockApi()` installed before navigation:

- `page.route("**/v1/**", …)` dispatches by pathname + method to fixtures whose shapes mirror the
  real backend (captured from a live server). Unknown endpoints return an empty-but-valid body.
- Channel subscribe/unsubscribe mutate an in-memory list, so add/remove reflect through the real UI
  on re-fetch.
- The event WebSocket is stubbed (`routeWebSocket`) so the app's live channel doesn't error.

## Adding a spec

```ts
import { test, expect } from "./fixtures";

test("…", async ({ page }) => {
  await page.goto("/");
  // interact + assert
});
```

If a flow reads a new endpoint, add its fixture + a route branch in `fixtures.ts` — the catch-all
returns `{}`, which will crash components that expect arrays (e.g. persona `recommends`). Prefer
`getByRole`, but note some controls (the Sources bar, the ✕ remove) take their accessible name from
inner content — target those with `getByTitle`/`getByLabel`.
```
