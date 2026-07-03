import { useEffect, useState } from "react";
import {
  cloudLogin,
  getCloudGallery,
  getCloudStatus,
  getPersonas,
  installPersona,
  type CloudStatus,
  type GalleryPersona,
} from "../api";

// Settings ▸ Gallery — the curated coworker catalog from OpenCoworker Cloud.
// Requires sign-in by design (per-user install tracking, tenant-scoped
// personas); local persona installs (Settings ▸ Personas: dir/Git) work
// without it. Installing here reuses the exact same manifest parser and
// consent flow as a local install: personas land disabled until approved
// under Settings ▸ Personas.

const CARD = "rounded-xl border border-line bg-panel/60";
const BTN_ACCENT =
  "text-[12.5px] px-3 py-2 rounded-lg bg-accent text-white shrink-0 disabled:opacity-40";

export function GalleryTab() {
  const [cloud, setCloud] = useState<CloudStatus | null>(null);
  const [cards, setCards] = useState<GalleryPersona[]>([]);
  const [installed, setInstalled] = useState<Set<string>>(new Set());
  const [unavailable, setUnavailable] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [signingIn, setSigningIn] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  const reload = async () => {
    getCloudStatus().then(setCloud).catch(() => setCloud(null));
    getPersonas()
      .then((ps) => setInstalled(new Set(ps.map((p) => p.id))))
      .catch(() => {});
    try {
      const g = await getCloudGallery();
      setCards(g.ok ? g.personas : []);
      setUnavailable(!g.ok);
    } catch {
      setCards([]);
      setUnavailable(true);
    }
  };
  useEffect(() => {
    reload();
  }, []);

  const signIn = async () => {
    setSigningIn(true);
    await cloudLogin(); // sidecar opens the browser; poll for completion
    setTimeout(() => {
      setSigningIn(false);
      reload();
    }, 3000);
  };

  const install = async (slug: string) => {
    setBusy(slug);
    setMsg(null);
    const r = await installPersona({ gallery_slug: slug });
    setBusy(null);
    if (!r.ok) {
      setMsg(r.error || "install failed");
      return;
    }
    setMsg(
      `Installed ${slug} — review and enable it under Settings ▸ Personas (it stays off until you approve it).`,
    );
    reload();
  };

  if (cloud && !cloud.signed_in) {
    return (
      <div className={CARD + " p-5 flex items-center gap-4"} data-testid="gallery-signin">
        <div className="min-w-0 flex-1">
          <div className="font-semibold text-[14px] mb-1">Sign in to browse the Gallery</div>
          <div className="text-[12.5px] text-muted leading-relaxed">
            The Gallery is a curated set of coworkers from OpenCoworker Cloud and needs a
            (free) cloud sign-in. Installing personas from a folder or Git URL — under
            Settings ▸ Personas — always works without an account.
          </div>
        </div>
        <button className={BTN_ACCENT} onClick={signIn} disabled={signingIn}>
          {signingIn ? "Check your browser…" : "Sign in"}
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-3" data-testid="gallery-cards">
      {msg && <div className="text-[12.5px] text-muted">{msg}</div>}
      {unavailable && cloud?.signed_in && (
        <div className="text-[12.5px] text-muted">The gallery is unreachable right now — try again in a moment.</div>
      )}
      {cards.map((p) => {
        const isInstalled = installed.has(p.slug);
        return (
          <div className={CARD + " p-4 flex gap-4"} key={p.slug} data-testid={`gallery-${p.slug}`}>
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2 mb-0.5">
                <span className="font-semibold text-[14px]">{p.name}</span>
                <span className="text-[10.5px] px-1.5 py-0.5 rounded border border-line text-muted">
                  {p.family}
                </span>
                <span className="text-[11px] text-faint">v{p.version} · {p.publisher}</span>
              </div>
              <div className="text-[12.5px] text-muted mb-1.5">{p.tagline}</div>
              <div className="text-[12px] text-faint leading-relaxed mb-2">{p.description}</div>
              {p.recommended_connectors.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {p.recommended_connectors.map((c) => (
                    <span
                      key={c}
                      className="text-[10.5px] px-1.5 py-0.5 rounded border border-line text-muted"
                    >
                      {c}
                    </span>
                  ))}
                </div>
              )}
            </div>
            <div className="shrink-0 self-center">
              {isInstalled ? (
                <span className="text-[12px] text-muted">Installed</span>
              ) : (
                <button
                  className={BTN_ACCENT}
                  onClick={() => install(p.slug)}
                  disabled={busy === p.slug}
                >
                  {busy === p.slug ? "Installing…" : "Install"}
                </button>
              )}
            </div>
          </div>
        );
      })}
      {cards.length === 0 && !unavailable && (
        <div className="text-[12.5px] text-muted">No personas published yet.</div>
      )}
    </div>
  );
}
