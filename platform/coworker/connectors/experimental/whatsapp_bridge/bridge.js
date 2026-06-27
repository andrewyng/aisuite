#!/usr/bin/env node
/**
 * OpenCoworker WhatsApp Personal bridge — EXPERIMENTAL, use at your own risk.
 *
 * Speaks the unofficial WhatsApp Web protocol via Baileys and talks to the Python
 * adapter over a stdio event stream: every inbound message, QR refresh, and state
 * change is one NDJSON line on stdout, so there is no inbound HTTP surface, no
 * polling, and no message queue to overflow. The only network listener is a
 * single localhost POST /send endpoint, kept so the stateless send_message tool
 * can deliver replies without holding a handle to this process.
 *
 * stdout events (one JSON object per line):
 *   {"event":"ready","port":N}             HTTP send endpoint is up
 *   {"event":"state","state":S,"me":J}     S: pairing|open|closed|reconnecting
 *   {"event":"qr","qr":"..."}              pairing QR (refreshes periodically)
 *   {"event":"message","id","chat","sender","name","group","text","ts"}
 *
 * Usage: node bridge.js --port 3941 --session <dir> --mode self|all
 */

import http from "node:http";
import { mkdirSync } from "node:fs";

import makeWASocket, {
  useMultiFileAuthState,
  fetchLatestBaileysVersion,
  DisconnectReason,
} from "@whiskeysockets/baileys";

const opt = (flag, fallback) => {
  const i = process.argv.indexOf(`--${flag}`);
  return i > 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
};

const PORT = Number(opt("port", "3941"));
const SESSION_DIR = opt("session", "./wa-session");
const MODE = opt("mode", "self") === "all" ? "all" : "self";
const TEXT_LIMIT = 4096;

const emit = (obj) => process.stdout.write(JSON.stringify(obj) + "\n");
const digits = (jid) => String(jid || "").replace(/[:@].*$/, "").split(":")[0];

mkdirSync(SESSION_DIR, { recursive: true });

class WhatsAppLink {
  constructor() {
    this.sock = null;
    this.me = null;
    this.ownSends = [];
  }

  rememberSend(id) {
    if (!id) return;
    this.ownSends.push(id);
    if (this.ownSends.length > 1000) this.ownSends.shift();
  }

  textOf(m) {
    const c = m.message || {};
    return (
      c.conversation ||
      c.extendedTextMessage?.text ||
      c.imageMessage?.caption ||
      c.videoMessage?.caption ||
      ""
    );
  }

  // mode "self": only the message-yourself thread; user's own messages are the input,
  // but anything this bridge itself sent must never loop back in.
  shouldForward(m) {
    const chat = m.key.remoteJid || "";
    if (!chat || chat === "status@broadcast") return false;
    const isSelfThread = digits(chat) === digits(this.me);
    if (MODE === "self" && !isSelfThread) return false;
    if (m.key.fromMe) {
      return isSelfThread && !this.ownSends.includes(m.key.id);
    }
    return true;
  }

  async open() {
    const { state, saveCreds } = await useMultiFileAuthState(SESSION_DIR);
    const { version } = await fetchLatestBaileysVersion().catch(() => ({}));
    this.sock = makeWASocket({ auth: state, version, printQRInTerminal: false });
    this.sock.ev.on("creds.update", saveCreds);

    this.sock.ev.on("connection.update", (u) => {
      if (u.qr) emit({ event: "qr", qr: u.qr }), emit({ event: "state", state: "pairing", me: null });
      if (u.connection === "open") {
        this.me = this.sock.user?.id || null;
        emit({ event: "state", state: "open", me: digits(this.me) });
      }
      if (u.connection === "close") {
        const code = u.lastDisconnect?.error?.output?.statusCode;
        if (code === DisconnectReason.loggedOut) {
          emit({ event: "state", state: "closed", me: null });
        } else {
          emit({ event: "state", state: "reconnecting", me: null });
          setTimeout(() => this.open().catch((e) => emit({ event: "state", state: "closed", error: String(e) })), 2500);
        }
      }
    });

    this.sock.ev.on("messages.upsert", ({ messages, type }) => {
      if (type !== "notify") return;
      for (const m of messages) {
        const text = this.textOf(m);
        if (!text || !this.shouldForward(m)) continue;
        emit({
          event: "message",
          id: m.key.id || "",
          chat: m.key.remoteJid || "",
          sender: digits(m.key.fromMe ? this.me : m.key.participant || m.key.remoteJid),
          name: m.pushName || "",
          group: String(m.key.remoteJid || "").endsWith("@g.us"),
          text,
          ts: Number(m.messageTimestamp) || 0,
        });
      }
    });
  }

  async deliver(to, body) {
    if (!this.sock || !this.me) throw new Error("not paired/connected yet");
    const sent = await this.sock.sendMessage(String(to), {
      text: String(body).slice(0, TEXT_LIMIT),
    });
    this.rememberSend(sent?.key?.id);
    return sent?.key?.id || "";
  }
}

const link = new WhatsAppLink();

const server = http.createServer((req, res) => {
  const reply = (code, body) => {
    res.writeHead(code, { "content-type": "application/json" });
    res.end(JSON.stringify(body));
  };
  if (req.method !== "POST" || req.url !== "/send") return reply(404, { sent: false, error: "unknown route" });
  let raw = "";
  req.on("data", (chunk) => (raw += chunk));
  req.on("end", async () => {
    try {
      const { to, body } = JSON.parse(raw || "{}");
      if (!to || !body) return reply(400, { sent: false, error: "to and body required" });
      reply(200, { sent: true, id: await link.deliver(to, body) });
    } catch (e) {
      reply(500, { sent: false, error: String(e?.message || e) });
    }
  });
});

// loopback only: the send endpoint must never be reachable off-machine
server.listen(PORT, "127.0.0.1", () => {
  emit({ event: "ready", port: PORT });
  link.open().catch((e) => {
    emit({ event: "state", state: "closed", error: String(e?.message || e) });
    process.exit(1);
  });
});

for (const sig of ["SIGINT", "SIGTERM"]) {
  process.on(sig, () => process.exit(0));
}
