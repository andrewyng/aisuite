# UX Decisions — OpenCoworker GUI

**Owner:** Rohit. **Status of this doc:** living spec — the source of truth for *why the UX is the
way it is*. Every entry is a deliberate decision with a rationale.

> **For agents/contributors:** Do **not** restructure or "improve" any UX listed here without the
> owner's explicit sign-off. If a change seems warranted, propose it (add a `↪ Proposed` note under
> the relevant entry) and ask — don't silently override a decision. New UX requests get appended
> here with their rationale before/while they're built.

Mocks that realize these decisions live in `platform/ui-mocks/` (`redesign.html`). The mock is the
visual spec; this doc is the reasoning + data-model spec. **Implementation** is specced in
[`UI-REFRESH-SPEC.md`](UI-REFRESH-SPEC.md) (+ [`UI-REFRESH-VERIFICATION.md`](UI-REFRESH-VERIFICATION.md),
[`FAKE-SLACK-SPEC.md`](FAKE-SLACK-SPEC.md)).

Status legend: **Decided** (settled), **Proposed** (endorsed, not finalized), **Mocked** (shown in
the HTML mock), **Built** (in the real React/Python app).

---

## 1. Connector-agnostic source model
- **Decision (Decided / Mocked):** Every external source renders through one connector registry:
  `{ id → label, brand color, logo }`, with a **default fallback** (neutral plug glyph + gray) so an
  unknown/custom/not-yet-shipped connector still renders cleanly.
- **Rationale:** We're going multi-connector (Slack now; Salesforce, HubSpot, GitHub, Datadog,
  PagerDuty, Telegram, custom/MCP next). UI must never hardcode "Slack"; adding a connector should be
  a registry entry, not new components.
- **Data model:** the registry lives on the connector descriptor (brand color + logo id). Recommended
  connectors that we don't ship yet still get a label + fallback badge.

## 2. Session conversation
### 2a. Connector inbound-message card
- **Decision (Decided / Mocked):** Inbound messages from a connector get a dedicated card (not the raw
  `💬 New message on slack:C…` bubble): a subtle brand-tinted header with **connector logo +
  channel/entity name + person name + time**, body below, brand-tinted left edge.
- **IDs on hover:** the header shows friendly names by default; hovering reveals the raw IDs
  (`C0BD7KZ1AH5 · U07JK68S4BH`). Names come from connector resolution (e.g. Slack `users.info`).
- **Rationale:** IDs are noise in the common case but essential when debugging routing; hover keeps
  both without clutter. The card generalizes to Salesforce "Case #123", etc.

### 2b. Collapsible tool/approval steps
- **Decision (Decided / Mocked):** Tool calls + approvals (e.g. `send_message`, `write_file` approval)
  collapse into a single `<details>` disclosure (`"2 actions · 1 approval ✓"`), **collapsed by
  default**. Expand for the per-step detail (args, who approved, where).
- **Rationale:** The thread should read message → reasoning → reply. Mechanics are available on demand,
  not in your face. (Re-introduces the earlier collapsed-steps pattern.)

## 3. "Sources" bar + per-session connections drawer
- **Decision (Decided / Mocked):** Below the session title sits a **Sources** bar showing **connector
  icons only** (overlapping avatars) + a short "needs attention" badge. Clicking it opens a
  **per-session connections drawer** (right slide-over).
- **Attention badge:** use a compact **⚠ N** (yellow exclamation + count) to denote *N connections
  recommended but not yet connected* — short, not the verbose "N recommended". *(Req 2026-06-29)*
- **Drawer content (persona-aware):**
  1. A **why-connect blurb** for the session's persona ("Ops works best wired into GitHub, Slack, a
     monitoring dashboard…") + a small progress indicator (`2 of 4 connected`).
  2. **Connected** sources for this session (with live status **and an enable/disable toggle** — see §4).
  3. **Recommended connectors** and **recommended MCP servers**, each with **the value it unlocks**
     ("so I can pull the firing alerts"), tiered **core vs optional**, with Connect/Add.
  4. A link out to **global Integrations**.
- **Rationale:** Turns an empty integration state into a capability story; progressive setup the user
  can finish later (explicitly *not* gating onboarding). Per-session scope keeps it contextual; the
  drawer reuses the global Integrations visual language so it's one mental model.
- ↪ **Superseded (2026-07-11):** the always-visible bar + `⚠ N` badge is replaced by the
  session-settings row (rest = one icon; the nudge moves into the drawer) — see §23. The drawer
  itself lives on, renamed "Session settings" and extended with working directories.

## 4. Connection hierarchy: persona-level vs session-level enablement  *(Decided — data-model change)*
- **Decision (Decided 2026-06-29 by owner):** Today connectors are enabled **per persona**. We're
  introducing **per-session enable/disable**: a connection can be enabled for the persona but toggled
  **off for a specific session** (Connect *and* an Enable toggle in the drawer).
- **Hierarchy (confirmed):** `account-connected connector` → `persona enables it` → `session may
  disable it`. Effective state for a session = connected AND persona-enabled AND not session-disabled.
- **Rationale:** A session may want a narrower surface than its persona's default (e.g. mute Slack for
  one focused session) without disconnecting the account or editing the persona.
- **Open:** storage shape (a per-session override set, mirroring how Unattended/subscriptions are
  stored). To design before building.

## 5. Richer persona manifests — `recommends`  *(Decided → Built: data model)*
- **Decision:** Persona manifests gain a `recommends` list: each item references a `connector:` or
  `mcp:`, with a `reason` (the value it unlocks) and a `tier` (`core` | `optional`). This drives the
  drawer's persona-aware recommendations (§3) — data, not hardcoded UI.
- **Rationale:** "Good evolution before we go live." Recommendations must travel with the persona
  (incl. third-party personas), so they belong in the manifest, not the frontend.
- **Validation:** `recommends` is **not** strictly validated against shipped connector descriptors (a
  persona may legitimately recommend a connector we don't ship yet, or an external one). Only the
  structure (kind/ref/tier) is validated. *(Contrast: `tools` ARE validated against the catalog.)*

## 6. Global Integrations page
- **Decision (Decided / Mocked):** Replace the single long scroll with a **left sub-nav**:
  **Connectors · Messaging routing · Activity · MCP servers**.
  - *Connectors*: card grid; a connected two-way connector expands to its allow-list + recent senders.
  - *Messaging routing*: groups the three previously-scattered controls — channel subscriptions, DM
    routing, Unattended-approvals routing — into one panel.
  - *Activity*: the Unrouted/failed dead-letter table (with a count badge in the nav).
- **Rationale:** Things were "dumped one after another," hard to find. Grouping by intent fixes
  discoverability. (Global = account-wide; the per-session drawer in §3 is the scoped counterpart.)

## 7. Left navigation — dual layout  *(Decided / Mocked)*
- **Decision:** Support **two session layouts**, user-toggleable via a small icon next to the
  "OpenCoworker" wordmark:
  1. **Flat** — Pinned + Recent (current).
  2. **Grouped by persona** — sessions clustered under their persona.
- **Boundaries (Decided 2026-06-29):** the grouped layout's per-persona sections must have **clear
  visual boundaries** — render each persona group as a **bounded card** (faint surface + border +
  header), not just a text header (the first cut had indistinct boundaries).
- **Per-group gear → Persona page (§9):** each persona group header carries a **settings gear** that
  opens the persona's detail page.
- **Rationale:** Different mental models — time/recency vs. persona/role. Don't force one. Pins remain
  pure accessibility in both.
- **Open:** persist the choice (prefs); default = Flat.

## 8. "New session" — persona picker  *(Decided / Mocked)*
- **Decision (Decided 2026-06-29):** "New session" is a **split button** — the main action starts a
  session with the **last-used / default persona**; the **▾** opens a dropdown to pick from the
  **enabled personas** (icon + name + tagline) + "Manage personas…" (→ §9). Not a modal, not an
  always-dropdown.
- **Rationale:** Fast path for the common case + discoverable choice without a heavyweight picker.
  Personas are the product's organizing concept, so surfacing them at session creation is right.

## 9. Persona detail page  *(Decided / Mocked)*
- **Decision:** A per-persona detail page (opened from the grouped-nav gear, or "Manage personas…")
  shows: **identity** (icon, name, tagline) + an **Enable** toggle; **About**; **built-in
  capabilities** (tools); **Connections for full benefit** (the manifest `recommends`, §5 — core/
  optional + the value each unlocks + connect state); **"New sessions get by default"** (the
  persona→session default connections, each toggleable — the middle layer of the §4 hierarchy); and
  **defaults** (models, mode, workspace).
- **Rationale:** The owner needs one place to answer "what *is* this persona, what does it need
  connected, and what does a session inherit by default?" It's also where the persona-level layer of
  the §4 hierarchy is configured.

## 10. Persona enablement / onboarding  *(Proposed — tabled)*
- **Question (owner):** Enabling a persona is just a toggle today, but it implies setup (connecting
  the recommended sources). Should that onboarding happen **inside the first session**, or be a more
  **explicit upfront** step? Onboarding can be heavy if it requires connecting several sources.
- **Recommendation (mine):** Keep it **lightweight and progressive, never gating**. Enabling a
  persona flips it on and surfaces its recommended connections (reuse the §9 page / a slim "set up
  Ops" panel) where the user connects what they want **now** and the rest stay as `⚠ N` nudges in the
  session's Sources bar (§3) — completable later. The first session works immediately with whatever's
  connected; it just shows the nudge. Avoid a blocking multi-step wizard.
- **Status:** **tabled** for later by owner; design intent recorded so we don't accidentally build a
  heavy gated wizard.

## 11. Session top bar — no model / mode chips  *(Decided 2026-06-29 by owner)*
- **Decision:** The session top bar shows **only** the title · persona · ⋯ menu (left) and the
  persona/panel icons (right). The read-only **model** chip (`anthropic:claude-opus-4-8`) and
  **permission-mode** chip (`Interactive`) are **removed**.
- **Rationale:** Both were non-interactive `<span>`s that *duplicated* controls already in the
  composer — the model dropdown and the **"Ask for approval"** permission-mode dropdown — where the
  user actually changes them. Worse, the mode chip showed the raw enum **"Interactive"** while the
  composer labels the identical setting **"Ask for approval"**, so the two names read as two different
  things. The composer is the single source of truth; the chips were clutter. (The persona detail
  page §9 still documents the persona's *default* model + mode — that's a different surface.)
- **Note:** The three permission modes remain unchanged — **Plan** (read-only, propose first),
  **Interactive** = "Ask for approval" (asks before edits/commands), **Full access** (runs without
  asking). This is a separate axis from the composer's **Unattended** toggle (routes approvals to the
  Inbox for hands-off runs). Mock (`redesign.html`) updated to match.

## 12. Sidebar bottom — Inbox + a single ⚙ menu  *(Decided 2026-06-30 by owner)*
- **Decision:** The sidebar bottom was a stack of 5 rows (Integrations · Automations · Inbox ·
  Activity + a path/gear footer) — "very busy." Collapse it to **two rows**: **Inbox** stays visible
  (its attention badge must be glanceable), and **Settings · Integrations · Automations · Activity**
  move into one **⚙ "Settings & more"** click-to-open menu that opens upward. The
  current workspace path becomes the menu's header (was the standalone footer).
- **Rationale:** Inbox is the only high-frequency destination; the rest are occasional (Integrations
  is set-up-once, Activity/Settings are rare). Similar products use a bottom menu for *account* items —
  we have no account (local, BYO-key), so ours holds *app* destinations instead. Net: 5 rows → 2,
  primary nav (New session, Search) stays clean at top.
- **Not chosen:** moving Inbox to the top cluster (kept it bottom-adjacent to its badge); keeping
  Integrations/Automations visible (folded them in for a cleaner bottom — promotable later if usage
  warrants).
- ↪ **Superseded (2026-07-11):** the no-account premise went stale when Phase 3 shipped cloud
  sign-in — the bottom is now a single account row with a state-driven inbox chip, and the ⚙
  menu became the account menu. See §26.

---

## 14. Per-session Slack channels — Sources drill-down  *(Decided 2026-07-01 by owner)*
- **Decision:** Managing which channels a session listens to belongs in the **Sources drawer**, not
  the composer's `+` menu. On a connected **two-way** messaging connector's row (Slack/Telegram), a
  **"Channels · N ›"** control opens a **child panel with a ‹ back button** — subscribed channels
  (× to stop listening) + an add picker (recent-channels datalist). The `+` menu stays about
  attachments only.
- **Rationale:** Sources is *the* per-session connection surface — it already owns "is Slack on for
  this session" (the mute toggle). Channels are the same category (per-session, connector-scoped
  standing config); the `+` menu is per-*message* attachment — a different mental model. Placing it
  on the Slack row makes it discoverable in context and gates itself (only shows when the connector
  is connected+enabled and two-way). The child panel gives the picker room and generalizes into a
  per-connector settings drill-down (allow-list, DM routing could live there later).
- **Reuse:** pure GUI — existing `subscribeChannel`/`unsubscribeChannel`/`getSubscriptions`/
  `getRecentChannels`; `two_way` read from the connector index the drawer already loads. No backend.
- **Not chosen:** the composer `+` menu (wrong mental model, needs conditional gating in the
  composer); inline row-accordion (cramped in the 420px drawer, clutters the Connected list).

---

## 13. Settings as a full page; Activity re-shelled  *(Decided 2026-07-01 by owner)*
- **Decision:** Retire the top-tab **ManageModal** and make **Settings** a full-page surface that
  reuses the Integrations shell (208px left sub-nav + centered panel + `PanelHead`). Split by scope
  (**Option 2**): Settings holds the *local/app* concerns — **Appearance · Files · Models · Personas**;
  anything *external* (Connectors · Messaging · MCP · Activity) stays under **Integrations**. The
  **Activity** page (old `AuditView`) moves onto the same page shell, dropping the legacy `page-view`
  layout and its duplicate header.
- **Rationale:** The modal was the last top-tab surface and the last `page-view` straggler, and it
  *duplicated* MCP/Connectors that already live under Integrations. One page idiom everywhere; no
  duplicated homes; more room than a modal. Models + Personas field bodies were re-skinned to the
  Tailwind card idiom to match Appearance/Files (left-aligned segmented control, carded config).
- **Wiring:** the ⚙ menu's *Settings*, the desktop tray's Settings event, the composer's "no model"
  chip, and the persona-card gear all route to `surface: "settings"` with an initial section. The
  shared tab bodies (`ModelsTab`, `ConnectorsTab`, `McpTab`) moved to `ManageTabs.tsx`; dead
  `SettingsTab`/`AuditTab`/`ManageModal`/`UnattendedToggle` removed.
- **Not chosen:** Option 1 (one unified Settings+Integrations hub) — cleaner long-term but a bigger
  rebuild and a very long single nav; the Settings/Integrations split reads more naturally.
- **Follow-up:** dead-CSS pass for now-unused modal/legacy classes (`manage-*`, `mtab`, `page-view`,
  `sa-view-*`, `persona-row`, `persona-install`, `consent-card`, `audit-*`); sync `redesign.html`.

---

## 15. Persona Gallery as a modal; delete; team-sharing-ready  *(Decided 2026-07-03 by owner)*
- **Decision:** The Gallery leaves the inline Settings ▸ Personas section and becomes a
  **screen-sized modal** opened from a "Browse the Persona Gallery" link on the Personas page
  (modal, not a route — installs finish back on Personas, disabled pending consent). Modal anatomy:
  header (title · search · close) + source chips (**All · From OpenCoworker · From your team**),
  a **featured carousel** (publisher-flagged `featured` cards, user-scrolled — never auto-rotating),
  and the catalog **list**; every card opens the in-modal **solo page** (hero + pitch + locally
  derived capabilities) — install only happens there. Personas page rows gain a **delete**
  affordance (non-builtin only, inline confirm, works signed out).
- **Not chosen:** loading cloud-served HTML in an iframe — it would reopen the
  "cloud describes capabilities" channel the solo-page design deliberately closed, need a spoofable
  postMessage install bridge, and break offline/theming/e2e. Rich showcase comes from publisher
  markdown + images rendered by our own components instead.
- **Visuals:** connector chips show hand-drawn SVG **brand marks** (`brandIcons.tsx`, neutral plug
  fallback); personas without publisher imagery get a deterministic **generated hero**
  (`PersonaHero.tsx`, hue from slug) so the carousel/solo pages never look empty.
- **Team-sharing-ready (design-only for now):** the "From your team" chip + empty-state teaser ship
  now; publish-to-tenant later reuses the gallery's existing `tenant_only` visibility. Team personas
  get **zero extra trust** — same validation, local capability derivation, disabled-until-approved.
- **Updates (design-only):** installed personas will record source (slug+version+hash); update
  re-runs consent **iff the capability surface changed** (never a silent permission expansion).

---

## 16. Workspace enum collapsed into family  *(Decided 2026-07-03 by owner)*
- **Decision:** The persona `workspace` enum (`git | project | deliverable | none`) is retired as a
  behavioral axis. `family` alone decides: **knowledge → transparent per-conversation scratch**
  (real folders added as session roots when needed; no folder gate, ever); **code → explicit
  directory picked by the user** (gate + project-grouped sidebar). Manifests may still carry the
  key (parsed + typo-checked for back-compat) but it's inert — the effective value derives from
  family. Built-in Ops moves to scratch + multi-root; Chat keeps `none` via its builder.
- **Rationale:** the only combo the enum enabled — knowledge+`project` (Ops) — predates multi-root
  knowledge sessions, which cover "work against a real folder" strictly better (progressive, no
  modal wall). The two-axis model split behavior across code paths (engine branched on family, the
  gate/sidebar on workspace) and produced the 2026-07-03 smoke-test contradictions: a gate demanding
  a folder while a scratch-backed chat ran behind it, and grouped-sidebar sessions with no home.
- **Future:** a `git: true` refinement of code-family may later force "start from a git repo", with
  clone-into-scratch as the safe execution mode. Personas may also *suggest* a root ("works best
  with a folder attached") — a banner, never a wall.

---

## 17. The model is fixed per session  *(Decided 2026-07-04 by owner)*
- **Decision:** a session's model is chosen up to the first turn, then **locked for the session's
  life**. The composer's model picker is interactive on a fresh session and becomes a read-only
  pill afterwards ("start a new session to switch"). Enforced **server-side** — the first
  user_message's `model` binds the engine; later message models and `set_model` are ignored —
  not just in the GUI, so API callers and socket races can't rebind a running conversation.
- **Mechanism:** every user_message carries the composer's visible model (the first one binds).
  This replaced a separate `set_model` handoff that could race the socket lifecycle (silent no-op
  before the socket exists; losable during the reconnect every new cowork session does to adopt
  its scratch dir; `ready` overwriting the visible selection) — the owner's repro was picking
  Opus and getting Kimi.
- **Rationale:** sessions are task-scoped; mixed-model transcripts invite provider-quirk breakage
  (tool-call replay, vision content), thrash prompt caches, and make behavior impossible to
  reason about. Mid-session switching was inherited chat-app convention, never a designed feature.
- **Future (owner, 2026-07-04, not built):** mid-session switching may return as a *designed*
  feature — a compatibility-matched switch, not a free dropdown: only models whose capabilities
  cover what the transcript already uses (tool-calling always; vision iff images are in history;
  reseller/vendor quirks vetted via the model matrix), with an explicit affordance + warning.
  Also covers the provider-died-mid-session case. Until then: start a new session.

---

## 18. Disable a persona = archive its conversations  *(Decided 2026-07-04 by owner)*
- **Decision:** disabling a persona **archives all of its real sessions** (unarchived,
  non-internal) in the same server-side action. Its sidebar section then disappears naturally —
  the grouped layout's never-orphan rule ("a persona with unarchived sessions always gets a
  section") stays untouched, because after the archive there is nothing left to orphan. No
  greyed-out sections, no time-based ("inactive for N hours") heuristics: every sidebar
  visibility change traces to an explicit user action.
- **Confirm:** unchecking *Enabled* on Settings ▸ Personas only **arms an inline confirm** when
  the persona has conversations — "Disabling archives its N conversations — they stay available
  under 'Show archived'" with Disable / Keep enabled (the same two-step idiom as row delete).
  With zero conversations the checkbox flips instantly; the confirm exists for the side effect,
  not for ceremony.
- **Re-enable never unarchives.** That would overwrite the user's archive state; history returns
  one click at a time via the Show-archived disclosure. The archive step lives in the manager
  (`set_persona_enabled`), so both persona routes — and any future client — share the semantic.

## 19. Unauthorized senders: park, don't drop; connector card is the config surface  *(Decided 2026-07-04 by owner)*
- **Problem:** first contact on a two-way connector took a double-send — the allow-list
  (correctly closed by default) silently dropped the first message; the sender only appeared
  under "Recent senders", and after being allowed had to message AGAIN.
- **Decision:** an allow-list drop **parks the message** instead of losing it. The connector's
  expanded card shows "Messages from senders you haven't allowed" with three resolutions:
  **Allow & deliver** (allow-list the sender AND re-inject the original message through the
  normal inbound path — buffer + subscriptions — no re-send), **Allow only** (future messages
  flow; this one is discarded), **Dismiss** (throw away, nothing else changes). Parked items
  are capped and persisted (`parked.json`).
- **The expanded connector card is the one-stop config surface** for a two-way connector:
  tools, allow-list + recent senders, parked messages, and "Sessions listening" — the
  per-connector cut of the global Channel-subscriptions table (which stays under
  Integrations ▸ Messaging routing; the owner looked for it on the connector and couldn't
  find it there).
- **Tokens hot-reload (no restarts):** a platform socket authenticates at connect time, so new
  creds require reopening that socket — and nothing else. Connect/disconnect of a messaging
  connector now refreshes the gateway listeners in-process; the sidecar never restarts. (Found
  when pasted Slack tokens "did nothing": the listener only started in the app lifespan.)
- ↪ **Partially superseded (2026-07-08):** the parked/allow-list semantics stand, but the config
  surface moves from the expanded card to the connector's **detail subpage** — see §21.

## 20. Collapsible left nav + RECENT header group/filter  *(Decided 2026-07-05 by owner)*
- **Decision:** The left nav gains three refinements (extends §7):
  1. **Collapse (⌘B) with hover-peek.** The nav can collapse so the content reclaims the width
     (grid → single column; the sidebar is taken out of flow). Collapsed, hovering the left edge
     (`.nav-hover-zone`) *peeks* it back as a floating overlay (shadow, over content, auto-hides on
     leave); ⌘B, the brand pin button, or the floating reveal button (`.nav-reveal-btn`, cleared
     past the traffic lights on desktop) dock it. Collapse is persisted per-device (localStorage).
  2. **RECENT header owns grouping + filtering.** The old brand-bar layout toggle is gone; the
     brand bar now holds only the wordmark + the collapse/pin control. A **RECENT** section header
     (like PINNED) carries a sliders control that opens one popover with **Group by** (Persona =
     the accordion ↔ Chronological = the flat list) and **Filter by coworker** (persona checkboxes;
     none checked = all shown). The popover stays open so you can group AND filter in one visit.
  3. **Artifact preview auto-collapses the nav.** Opening a full artifact preview (PDF/webpage/
     sheet) collapses the nav for max width and restores it on close — unless you manually toggled
     the nav meanwhile (the manual action takes control; the auto-collapse never overwrites the
     saved pref).
- **Inspiration (not copied):** Similar products collapsible sidebars + their group/filter menus.
- **Tests:** `e2e/nav-collapse.spec.ts` (collapse/dock, ⌘B, popover grouping); `Sidebar.test.tsx`
  updated to the new group/filter control.

## 21. Connectors redesign: connected-first list, detail subpages, add-modal, privacy filters  *(Decided 2026-07-08 by owner; mock: `ui-mocks/connectors-redesign.html`)*

Driven by managed-Slack going multi-workspace (M3.5): the connector card had outgrown itself, and
the interim "Slack workspaces" Integrations tab was the wrong shape. One pattern for ALL connectors:

- **Navigation:** the Integrations sub-nav stays fixed (Connectors · Messaging routing · Activity ·
  MCP servers) — **no per-connector nav items**. Each connected connector opens a **detail subpage**
  under Connectors (breadcrumb `‹ Connectors`). Slack manages workspaces there; Gmail accounts;
  HubSpot portals; Teams orgs. Supersedes both the expanded-card config surface (§19) and the
  short-lived "Slack workspaces" tab.
- **Connectors list:** **Connected first**, in its own section — single-column rows with brand
  badge, one-line status, a **health chip** (● Live / ⚠ Reauthorize / ● Ready) and a chevron; the
  row itself navigates. "Available" below (row list + Connect pills), long tail behind "show all".
  Cloud sign-in shrinks to a slim strip. Problems must surface **in the list**, not after a click.
- **Visual grammar:** macOS System-Settings style — grouped inset lists on gray paper, hairline
  separators, 44px rows, pill buttons, sentence-case group headers, minimal copy (footnotes, not
  paragraphs; IDs on hover). Owner: "I really like Apple for their aesthetics"; v1 was too verbose.
- **One entry point to add a connection:** the header button (`＋ Add workspace` / `＋ Add account`)
  opens a **modal** with a One click | Manual pill switcher, each pane carrying its instructions
  (Slack: relay vs Socket-Mode tokens; HubSpot: OAuth vs private-app token). **Modal only when a
  connector has ≥2 connect modes** — single-mode connectors (Gmail, Teams) launch directly. The
  bottom-of-page "Add a …" sections are gone (duplicate CTA).
- **Multi-account everywhere:** Slack workspaces, Gmail accounts, HubSpot portals, Teams orgs —
  same skeleton: one group per account with per-account auth state and Disconnect. A **Default**
  badge answers "which account do tools use?"; sessions can override in Sources; **every approval
  card names the account/portal it will act on** (a sandbox test can never quietly hit production).
  Profile keying generalizes the Slack pattern: `gmail:account:<email>`, `hubspot:portal:<hub_id>`.
- **Tools are a collapsed disclosure on every detail page** ("Tools · 3 of 5 enabled") — the lever
  exists everywhere but stays quiet; write tools carry "asks first" tags; destructive tools (e.g.
  HubSpot delete) are **never offered**.
- **Connection health = three honest layers** (Slack page): desktop↔relay WS (Live/Reconnecting/
  Offline + last event), cloud sign-in (required for relay), per-workspace/account token health.
  We never claim "Slack↔cloud is down" — event silence is indistinguishable from a quiet workspace.
- **Privacy filters, enforced at the DESKTOP, silent to agents:** the invariant is **"the cloud
  knows routing; the desktop knows content and policy."** Gmail: "Never show agents" senders/
  domains/labels — filtered mail **does not exist** from the model's perspective (no tombstone:
  `<truncated>` markers leak sender/subject and invite the model to reason around policy); the
  user sees "N hidden by filters" out-of-band on the tool card (MessageSource-sidecar pattern) +
  audit entries. HubSpot: **hidden fields** (property denylist stripped before model context) +
  **read-only vs read & write chosen at consent time** (scope minimization is the real ACL) +
  "agents see what the connecting HubSpot user sees" (record ACLs belong server-side in HubSpot
  permission sets — client-side per-record filters are theater and are deliberately NOT built).
- **Teams (concept):** relay-only (bots need a public endpoint — no manual mode). Consent is
  self-serve: chats = user installs it; channels = any **team owner** (RSC, per-team consent);
  org-blocked = Teams' native "Request approval" asks IT once. No tenant-admin Graph permissions
  by design — same per-actor privacy posture as the Slack relay.

## 22. Session screen cleanup: contextual top-left cluster, facts subtitle, three-control composer  *(Decided 2026-07-11 by owner; mocks: `ocw-context/docs/ux-improvements/mocks/UX-002-session-screen.html`)*

Owner sketch (hand-drawn) → discussed and resolved in the UX ledger (ocw-context, UX-002). Both the
fresh-session and in-progress screens shed chrome:

- **Top-left cluster `[sidebar] [+ new session] [search]` renders only when the sidebar is
  collapsed** — the expanded sidebar already owns those actions; never duplicate them. Build note:
  the cluster sits inside §20's hover-peek zone — the peek must not fire while the cursor is on
  these icons (start the zone below the icon row or add a short delay; the pin/reveal co-location
  logic already handles this corner). Keyboard shortcuts stay global regardless of sidebar state.
- **Centered title with a facts subtitle** beneath: `(Coworker · Opus 4.8)`; code sessions include
  the workspace folder. The subtitle is the session's **fixed facts, not controls** — it replaces
  the locked-model pill (§17's lock expressed spatially) and the topbar "About this persona"
  sliders button (subtitle click → the coworker page). Topbar right: the **panel toggle only**
  (mirrored variant of the left nav's sidebar glyph, one glyph both states); the right rail
  absorbs **artifacts only**. *(Revised at visual review 2026-07-11: the topbar
  session-settings icon was dropped as redundant — §23's row owns the drawer.)* *(Revised
  again 2026-07-11: the ⋮ conversation menu is REMOVED — the nav row's hover cluster owns
  pin/rename/archive/delete, so the topbar menu was a strict subset; the title STAYS (with
  the sidebar collapsed it is the only session identifier, and the subtitle orphans without
  it). The topbar goes edgeless: no bottom border, paper-tinted glass — invisible at rest,
  frosts only when the transcript scrolls under it.)* *(Revised 2026-07-11, third pass —
  ledger UX-008, mock `UX-008-merged-topbar.html`: the §23 session-settings row DOCKS into
  the bar's left region — one bar, not two strips. With the nav expanded the settings icon
  is the first element after the panel edge; collapsed, it follows the [sidebar][+][search]
  cluster. The §23 contract is unchanged.)*
- **Composer = `[+ attach] [Mode ⌄] [send]`.** The **Mode menu** carries the five permission
  options (Discuss / Plan / Ask for approval / Full access / Custom) **plus the
  Unattended/send-approvals-to-Inbox toggle** at the bottom — "who approves, and when" is one
  mental model; the separate InboxControl leaves the row. *(Revised 2026-07-11, competitor
  composer comparison: the trigger is borderless and names the CHOSEN mode — "Ask for
  approval ⌄", not a generic bordered "Mode ⌄" pill.)*
- **The model picker appears only on a fresh session** (quiet chip on the composer's right);
  after the first turn the fact moves up to the subtitle. No interactive-then-disabled control.
- **Folder/roots control and branch chip leave the composer** → the session-settings drawer
  (§23). Folder access is standing session config, not per-message attachment — the same
  reasoning §14 used to keep channels out of the `+` menu. The `+` menu stays attachments-only.
- **Open at build time:** fresh-session greeting copy; whether suggestion chips survive.

## 23. Session settings row: hover to glance, click to manage  *(Decided 2026-07-11 by owner; mock: `ocw-context/docs/ux-improvements/mocks/UX-003-session-row.html`)*

Replaces §3's always-visible SourcesBar (ledger UX-003). One sub-header row above the conversation
whose contract is **rest = icon · hover/focus = glance · click = manage**:

> **↪ Geometry revised 2026-07-11 (ledger UX-008):** the row DOCKS into the topbar's left
> region — the standalone strip under the bar is gone (one 48px bar, ~36px returned to the
> conversation). Everything below — the rest/glance/click contract, gray-is-the-nudge, zero
> reflow, deep links — is unchanged; only where the row renders moved.

- **Rest:** a single quiet icon, constant row height. **No nudge text at rest, ever** — the
  "recommended source not connected" nudge lives only in the drawer.
- **Hover/focus** (~150–200ms reveal delay so mouse-crossing doesn't flicker; reveals on keyboard
  focus too): a glance strip — **connected source icons in brand color**, persona-**recommended
  but unavailable ones in grayscale** (the gray icon IS the nudge, wordless), a **folder count**
  ("2 folders"; code sessions show the folder *name* — the workspace is the session's identity),
  and a trailing **"Configure ›"**. Icons only, no labels or chips. **No reflow** — row height is
  identical resting and hovered.
- **Gray covers both** "not connected" and "connected but muted for this session" (§4 override):
  the strip answers "what can *this session* touch right now"; tooltips disambiguate. Only
  persona-recommended connectors ever appear gray — never the whole catalog.
- **Everything in the glance is a shortcut:** icons and the folder count click straight into the
  matching drawer section. Tooltips are load-bearing once labels are gone (per-icon name + state;
  folder paths on the count).
- **Click:** opens the drawer, renamed **"Session settings"** — sources (connect-in-context,
  channels child panel §14, mute toggles §4) plus a new **working directories** section (roots
  list, add/remove, branch). ⚠ recommendations render here.
- **Rejected:** the icon morphing to "Click to configure" on hover (self-narrating UI, layout
  shift, permanent noise — affordance comes from the glance's content being clickable); at-rest
  nudge text (owner call: drawer only); placing the icon in the topbar with an anchored popover
  (max-clean but spatially disconnected — in-place morph keeps discoverability; cf. the
  cloud-sign-in placement regression, 2026-07-09).

## 24. First-run onboarding: model → recipe → tips  *(Decided 2026-07-11 by owner; mock: `ocw-context/docs/ux-improvements/mocks/UX-004-onboarding.html`; ledger UX-004)*

Replaces the settings-shaped first run. Three steps; **only step 1 gates** (§10's
progressive-never-gating rule):

- **1 — Connect a model.** Fields adapt per provider from the existing descriptors: key-only
  (Anthropic/OpenAI/Gemini, each with a create-a-key deep-link), endpoint-only (Ollama, prefilled
  — the free/local escape hatch), endpoint+key (Fireworks etc.). One **default-model dropdown**
  per provider (curated matrix, recommended pre-selected) — deliberately *not* an
  enable-checklist; curation stays in Settings ▸ Models. Inline Test/verify.
  *(Revised 2026-07-11 at owner's Mac-app walkthrough: headline = **"Welcome to
  OpenCoworker"** with "connect a model to get started" as the sub-line; native `<select>`s
  → the Settings-Models **SelectMenu** (sectioned Ready-to-use / Needs-setup, key-set dots);
  the **default-model dropdown is DROPPED** — the model is per-session (§17) and the old
  select never persisted anything — replaced by one pointer line to Settings ▸ Models; an
  optional endpoint (base_url with a default, on a keyed provider) collapses behind a
  **"Configure custom endpoint ›"** link (keyless providers keep it visible — the endpoint IS
  the connection); Test joins the action row: Skip setup … [Test] [Continue], status line
  fixed-height above it.)*
- **2 — Get your first automation running** (skippable). **Role tabs — Engineering · Sales ·
  Everyday** — each with a recipe one-liner, two connect rows, then the recipe card (source ·
  channel/time · cadence · consent). **One Cloud sign-in, lazily triggered by the first Connect**
  — never per-integration; tokens-stay-local copy on the pane. **Connections persist across
  tabs.** Channel fields carry the invite-@ocw hint. **Everyday** = morning brief (Calendar +
  Gmail) delivered in-app, Slack DM secondary. **Create automation enables only when the tab's
  connectors are connected.** The consent line — "post the digest to #X without asking each time;
  anything else still asks first" — is a **standing scoped approval: ledger UX-005, design
  pending — this section's BUILD IS BLOCKED on it.**
- **3 — You're set up.** Recap card of the created automation (absent if skipped) → Specialist
  coworkers tip (Show me → gallery) → one quiet session-control line, with **"Start working"
  opening the first session with the session-settings panel (§23) open** — teaching by landing,
  not telling.
- **Deferred (owner):** tab choice seeding which specialists the gallery features — revisit after
  the UI cleanup lands.

## 25. Standing scoped approvals for automations  *(Decided 2026-07-11 by owner; ledger UX-005 — unblocks §24's build)*

Recurring automations using gated tools (e.g. `send_message`) park an approval **every run** —
a weekly summary needing a weekly click isn't an automation. The fix is a remembered,
narrowly-scoped rule: *"this automation may call this tool against this exact target without
asking"* — **tool + target + owner (task id)**, none optional, no wildcards in v1. Rules live on
the `ScheduledTask` record (revocation on the task detail page; deleted with the task).

- **Minted on exactly two human-only surfaces — the model can never mint a rule:**
  1. **The creation consent card** (already exists — `create_scheduled_task` is approval-gated).
     Every automation surfaces its needs at creation: reads render as *disclosure* lines
     (read-only, no gating), writes are the *grants*, pre-set to allow. The agent path proposes
     the set via a new `permissions` field on the create-tool schema; the existing card renders
     it. **Rejected:** the agent writing `config.toml` — model-authored permission expansion in a
     global file, invisible in UI, outlives the automation.
  2. **"Allow every time"** on a recurring run's approval card — persists to the task record
     (unlike the session-scoped Always-allow). In-app only; not offered on Slack-mirrored
     buttons. The retrofit path.
- **Graceful degradation:** no grants → the automation still works; runs park approvals in the
  Inbox as today.
- **Invariants:** never offered for `risk=exec`/destructive tools (shell asks forever); additive
  on top of the run's permission mode — never a silent full-access upgrade; every auto-allowed
  call audits the rule; cards/audit name the account acted on (§21); persona install consent
  stays availability-only — installs never ship pre-approved writes.
- **Infra note:** `always_allowed_tools` + creation gating + run-time wiring already exist
  (`automation/models.py`, `manager.py`); the build is target-shaping those entries, the
  `permissions` create-field, the task-persistent Allow-every-time, and lifting `permissions.py`'s
  connector exclusion **for task-scoped, target-matched rules only**.

## 26. Sidebar bottom: one account row; Connectors renamed; Activity dedup  *(Decided 2026-07-11 by owner; mock: `ocw-context/docs/ux-improvements/mocks/UX-006-sidebar-account.html`; ledger UX-006)*

Supersedes §12's bottom cluster. §12's rationale ("we have no account, so the bottom menu holds
app destinations") went stale when Phase 3 shipped cloud sign-in two days later; the slot becomes
the **account anchor** and the bottom is **exactly one row**:

- **Account row:** avatar/initials + first name (email lives in the menu header, never on the
  row) + a green dot when signed in to OpenCoworker Cloud; signed out = "Not signed in" and the
  menu leads with the sign-in CTA. No workspace-path header (the path lives in Settings ▸ Files).
  Telemetry toggle moves to Settings. "Settings & more", the Inbox row, and any Connectors row
  are retired.
- **State-driven inbox chip with a sticky unlock** (owner: many users never park an item or use
  Unattended — a permanent Inbox row is dead chrome for them). Absent until the first item ever
  parks or Unattended is first enabled, then permanent: quiet icon when empty, accent + count
  when pending — §12's glanceability requirement, paid only when there's something to glance.
  Auth-independent (Inbox is local). **Two click targets:** the chip → Inbox directly; anywhere
  else on the row → the menu.
- **Menu (fixed):** email header · **Inbox** (with count) · **Connectors** · Settings (⌘,) ·
  Automations · Activity · Sign out. Inbox + Connectors are always listed — the permanent
  discoverable path regardless of chip state. **"Integrations" is renamed "Connectors"**
  everywhere (the findability complaint was the "& more" label; users think "connect Slack";
  MCP servers sit fine under the Connectors roof). "Inbox" keeps its name — "Approvals" was
  considered and rejected: the queue also holds questions, plan reviews, and folder-grant
  requests.
- **Activity dedup:** two unrelated pages were both named "Activity". The Integrations
  dead-letter page dissolves into **Messaging routing** as an "Unrouted" section (badge kept);
  the one remaining Activity = the audit log, in the account menu.
- **Automations** stays in the menu for now (owner: deserves a visible row, deferred). The
  Connectors-page sign-in strip retires — the account row supersedes it; connect-modal inline
  sign-in panes stay (§24's lazy trigger). Designed-not-built: Messaging routing keeps
  shrinking (§19/§21 moved per-connector cuts onto detail pages; the residual global table may
  later dissolve entirely).

> **Revision 2026-07-12 (§28):** Messaging routing did dissolve — the whole page (mirror
> channel, DM route, channel subscriptions, Unrouted with its ⚠ badge) moved to
> **Inbox ▸ Configure**; the Connectors sub-nav is now just Connectors · MCP servers.

## 27. Start screen: template tasks carry their own setup  *(Decided 2026-07-11 by owner; mock: `ocw-context/docs/ux-improvements/mocks/UX-007-start-tasks.html`; ledger UX-007)*

Extends §22's start-screen half and composes with §23. The fresh-Cowork empty state becomes
**exactly three concrete template tasks** — flat hairline rows (the de-boxed grammar), staggered
~300ms entrance — and nothing else between the greeting and the composer:

1. **Analyze the files in a directory** — action "Pick a folder →": shares a folder (inline
   add-folder form; straight to prefill when one is already shared), then prefills the composer.
2. **Create a report from my HubSpot leads** — gated on HubSpot.
3. **Automate a weekly GitHub progress report to Slack** — gated on GitHub + Slack; funnels
   into §24/§25's recipe + consent machinery.

- **No leading icon tiles** — the title is the row. Connector dots sit on the **sub-line**:
  brand color = connected and enabled for this session, grayscale = not (§23's vocabulary).
- **Sub-line copy is always the task's outcome** ("Sources, stages, and who needs follow-up"),
  never connection state — the dots and the trailing action carry that.
- **Row action contract:** sources ready → "Start →" revealed on hover, click prefills the
  composer with the template stem. Not ready → **"Configure ›" always visible** (for a gated
  row the setup action IS the row's meaning) and it opens the §23 Session settings drawer —
  the start screen adds **no second setup surface**.
- **"Set me up (optional)" is removed** — setup rides the task that needs it.
- Rejected on the way (competitor comparison): boxed category tiles expanding into template
  lists (reintroduces boxes + a navigation level); a specialist-coworker picker line on the
  composer edge and a tip/picker under the tasks (placement + "X is on this session" copy).
  The specialist entry point is **deliberately absent** — owner sketch to come.
- Same day: the ✳ greeting/boot/gate mark (read as a competitor's logo) → **✦** app-wide.

## 28. One page shell; Inbox absorbs Messaging routing  *(Decided 2026-07-12 by owner; mock: `ocw-context/docs/ux-improvements/mocks/UX-009-inbox-merge.html`; ledger UX-009)*

Owner walkthrough: Automations, Activity, and Inbox each had their own indentation and head
style; and "Messaging routing" hid inbox-delivery config under Connectors while the mirror
channel was editable in TWO places (that page's card + Inbox's inline configurator).

- **One page shell for every top-level page** — the Connectors/Activity pattern: full-bleed
  `main`, centered ≤4xl column, `PanelHead` (18px title, 12.5px muted subtitle BELOW it),
  card-based content. Page-level actions ("+ New automation") right-align with the head.
  Automations drops its icon-in-title and the boxed ⓘ banner (now a one-line muted note);
  Inbox drops its title+subtitle-on-one-line head. No page invents its own indentation again.
- **Inbox = two page-level tabs**, underline style — one visual level above the filter chips:
  - **Pending** (default): approvals/questions exactly as before (kind chips, persona chips,
    resolve-releases-agent). Badge = pending count. The routing status is a read-only line
    ("Also delivered to #ops-alerts — replies there resolve items here. Configure ›") whose
    link switches tabs; the inline editor is deleted — the mirror setting has ONE editor now.
  - **Configure**: the former Messaging-routing page moved whole — Unattended-approvals
    mirror + Direct-messages route (two-card row), Channel subscriptions, Unrouted. The ⚠
    unrouted count rides this tab. Rationale: Unrouted is "messages that never reached you" —
    the user asking "why didn't I get pinged?" looks in Inbox, not Connectors.
  - Tab name: owner offered "Destination"/"Configure" over the drafted "Delivery" —
    **Configure** won (honest umbrella for mixed settings; "Destination" names only the
    mirror card). Mirror target and routing line show the channel **name** when the recent
    list knows it (§24 revision 9's names-over-ids rule).
- **Connectors sub-nav shrinks to Connectors · MCP servers** (amends §26, which had already
  predicted the dissolution).

## Change log (requests, newest first)

- **2026-07-12 (10)** — Owner (walkthrough, screenshots of three pages): Automations /
  Activity / Inbox must share one look; Messaging routing belongs in Inbox → §28 (one page
  shell everywhere; Inbox tabs Pending / Configure; Connectors sub-nav = Connectors · MCP).
- **2026-07-11 (9)** — Owner (walkthrough, step 2 recipe): the channel box must show the
  channel NAME, not `slack:T…/C…` — ChannelPicker now separates display (#name at rest,
  raw address while editing + in the tooltip) from the stored target, on BOTH its surfaces
  (onboarding + session channel subscriptions); the consent line uses the name too. The
  fixed day+time cadence pairs → a day SelectMenu (Mon–Sun, Weekdays, Every day) × a free
  time field; digest instructions re-worded cadence-neutral ("since the last digest") →
  §24 revision.
- **2026-07-11 (8)** — Owner (Mac-app onboarding walkthrough): step 1 warms up ("Welcome to
  OpenCoworker" + connect-to-get-started sub-line); native selects → SelectMenu (the Settings
  Models control); default-model picker dropped (per-session anyway; never persisted);
  optional endpoints behind "Configure custom endpoint ›"; Test joins the Skip/Continue
  action row → §24 revision.
- **2026-07-11 (7)** — Owner (ledger UX-008, mock approved): the §23 session-settings row
  docks into the topbar's left region — one bar instead of two strips; contract untouched
  (rest = icon · hover/focus = glance · click = drawer). Expanded nav: icon first after the
  panel edge; collapsed: after the §22 cluster → §22/§23 amendments.
- **2026-07-11 (6)** — Owner (visual pass on the new shell): topbar ⋮ conversation menu
  removed (nav row's hover cluster covers it; title kept — sole identifier when the nav is
  collapsed); topbar goes edgeless glass (border dropped, paper-tinted blur); composer Mode
  trigger goes borderless and names the chosen mode → §22 amendments.
- **2026-07-11 (5)** — Owner (competitor new-session comparison; ledger UX-007, mock v3
  approved): start screen → three concrete template tasks that carry their own setup (no icon
  tiles; outcome-voiced sub-lines with connector dots; ready = hover "Start →" + prefill,
  gated = always-visible "Configure ›" → the §23 drawer); "Set me up (optional)" removed;
  specialist entry point deferred to an owner sketch → §27. Also: ✳ mark → ✦ app-wide.
- **2026-07-11 (4)** — Owner (Settings audit; ledger UX-006, mock v2 approved): sidebar bottom
  → one account row (name + cloud status dot + state-driven sticky-unlock inbox chip);
  "Settings & more" retired; Integrations renamed Connectors and lives in the account menu with
  Inbox; the two same-named "Activity" pages deduped (dead-letter → Messaging routing ▸
  Unrouted; audit log keeps the name); "Approvals" rename rejected → §26. Supersedes §12's
  bottom cluster.
- **2026-07-11 (3)** — Owner (UX-005 design discussion): per-automation standing scoped
  approvals — tool+target+task rules on the ScheduledTask record, minted only at the creation
  consent card (agent proposes via a `permissions` field) or a run card's "Allow every time";
  agent-written config.toml rejected → §25. Unblocks §24's build.
- **2026-07-11 (2)** — Owner (boss-flow study: new install → recurring GitHub→Slack digest;
  ledger UX-004, mock v2 approved): onboarding restructured to model → role-tabbed recipe →
  tips → §24. Recipe consent = standing scoped approval (ledger UX-005, design pending; §24
  build blocked on it). Gallery-seeding-by-tab parked until after UI cleanup.
- **2026-07-11** — Owner (hand-drawn sketch → UX ledger `ocw-context/docs/ux-improvements/`,
  entries UX-002/UX-003, mocks reviewed + approved): session-screen cleanup — contextual
  `[sidebar][+][search]` cluster (collapsed-sidebar only), facts subtitle replacing the locked
  pill + persona button, composer reduced to `[+][Mode ⌄][send]` with Unattended folded into the
  Mode menu, fresh-only model chip → §22. SourcesBar → session-settings row (hover glance, click
  manage; gray icon = the nudge; drawer renamed "Session settings" + working directories) → §23,
  superseding §3's bar. Naming (ledger UX-001, partial): in-app nav noun = **"Coworkers"**;
  **"Specialist"** reserved for marketing/gallery voice; internals keep `persona`.
- **2026-07-08** — Owner, reviewing the M3.5 Slack-workspaces tab: wanted connector detail as a
  subpage under Connectors (not a nav item), connected-first Connectors list, Apple-quiet styling,
  one add-connection entry point (header-button modal with One click | Manual pills), collapsed
  Tools everywhere, and enterprise privacy levers (Gmail sender/label exclusions — drop silently,
  no `<truncated>` tombstone; HubSpot hidden fields + read-only connect). Multi-account confirmed
  feasible for Gmail (accounts), HubSpot (portals), Teams (orgs). Teams consent corrected: self-
  serve via user install/team-owner RSC, admin only if org policy blocks apps → §21. Build order:
  Connectors list + subpage nav → Slack page → Gmail multi-account → HubSpot → Teams.
- **2026-07-05 (2)** — Owner aesthetic asks on the left nav: floating/collapsible on demand,
  a RECENT header with a group+filter control moved off the top bar, and
  auto-collapse when an artifact opens → §20. Chose hover-peek + pin, and auto-collapse with
  auto-restore.
- **2026-07-05** — Owner (Slack-on-PM setup): double-send first-contact flow called clumsy →
  §19 (park + one-step allow-and-deliver, connector card as config surface, gateway
  hot-reload). Root-caused "Slack keeps resetting": pre-Jul-3 pytest runs clobbered real
  tokens with a test stub (`xoxb-1`) — isolation fixture already fixed it; proved with
  hash-compare across the full suite. Sender names showing "unknown" = missing bot scopes
  (users:read, channels:read) — setup instructions updated.
- **2026-07-04 (2)** — Owner (testing pass, persona disable): disabled personas kept their
  sidebar sections because old sessions held them (never-orphan rule). Discussed hide vs grey vs
  time-based liveness; owner picked **archive-all-on-disable** (§18) with an inline confirm at
  the disable click.
- **2026-07-04** — Owner: model-key testing pass. First-class provider entries for
  OpenAI-compatible vendors + Together/Fireworks resellers with a curated model matrix (ids →
  labels → capabilities); custom provider picker (sections, key-set dot, last-used); "chose Opus,
  Kimi replied" → model rides every message and is **fixed per session** (§17).
- **2026-07-03 (2)** — Owner testing pass found 10 issues; while fixing the project-workspace
  cluster, owner challenged the workspace enum ("family:knowledge means scratch; family:code means
  explicit directory — why more?"). Decided §16: collapse workspace into family. Also: session
  delete → archive-first with soft confirm + scratch cleanup; sidebar caps sessions per persona
  (configurable); connect-in-context from the session drawer; Inbox filters + orphan pruning.
- **2026-07-03** — Owner: Gallery UX rethink (§15): delete personas; Gallery behind a link as a
  full-screen **modal** (owner preferred modal since installs finish in Personas); carousel + list;
  brand icons for connectors; default generated hero art. Iframe-HTML idea discussed and dropped
  (trust model). Team publish + persona updates designed, phased later.
- **2026-07-01 (2)** — Owner: "should the Slack-channel feature live in the right-side Sources panel?"
  Decided **yes** (§14): a **child panel with back** off the connected Slack/Telegram row manages the
  session's subscribed channels — not the composer `+` menu. Pure GUI, existing subscription APIs.
- **2026-07-01** — Owner: convert the Settings **modal → page** like Integrations, and re-shell
  **Activity** too. Decided **Option 2** (§13): Settings = Appearance/Files/Models/Personas;
  Integrations keeps Connectors/Messaging/MCP/Activity. Models + Personas re-skinned; modal + dead
  tabs removed; shared bodies → `ManageTabs.tsx`.
- **2026-06-30** — Owner: sidebar "very busy" vs other products. Decided: **bottom → Inbox + one ⚙
  "Settings & more" menu** (§12), folding Settings/Integrations/Automations/Activity into a
  click-to-open popup. 5 rows → 2.
- **2026-06-29 (3)** — Owner Q on the session top bar ("why the model name? what is 'Interactive'?").
  Decided: **remove both top-bar chips** (§11) — they duplicated the composer's model + "Ask for
  approval" controls, with the mode chip mislabeled vs. the composer. App + mock updated.
- **2026-06-29 (2)** — Confirmed: §4 hierarchy, §8 split button. New: grouped-nav needs clear
  **boundaries** + a per-persona **gear** (§7); **Persona detail page** (§9). Tabled: **persona
  enablement/onboarding** (§10) — keep it lightweight/progressive, not a gated wizard.
- **2026-06-29** — Req: `⚠ N` badge (not "N recommended"); per-session enable/disable toggle
  (hierarchy §4); richer manifests with `recommends` (§5, built); dual left-nav layouts (§7);
  New-session persona dropdown (§8). Owner endorsed the persona-connection drawer concept (§3).
- **2026-06-28/29** — Initial redesign asks: connector message card (§2a), Sources bar (§3),
  Integrations de-clutter (§6), connector-agnostic design (§1), collapsible steps (§2b).
