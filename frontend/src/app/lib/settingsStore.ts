// User-facing preferences, persisted per-account in PostgreSQL. An in-memory
// cache keeps getSettings() synchronous for render; it is hydrated from the
// backend on login/startup and written through on save. Appearance is applied
// to CSS variables so the whole app reflects it live.

import { apiGetSettings, apiPutSettings } from "./dataApi";

export interface AppSettings {
  // Profile extras not held by the auth record.
  email: string;
  organisation: string;
  // Notification toggles.
  notifyInspections: boolean;
  notifyClaims: boolean;
  notifyRenewals: boolean;
  notifyInsurance: boolean;
  notifyAnnouncements: boolean;
  notifyDigest: boolean;
  // Security.
  twoFactorApp: boolean;
  twoFactorSms: boolean;
  // Regional.
  language: string;
  timezone: string;
  dateFormat: string;
  currency: string;
  // Appearance.
  theme: string;
  accent: string;
  sidebarCollapsed: boolean;
  compactMode: boolean;
}

export const DEFAULT_SETTINGS: AppSettings = {
  email: "",
  organisation: "",
  notifyInspections: true,
  notifyClaims: true,
  notifyRenewals: true,
  notifyInsurance: false,
  notifyAnnouncements: false,
  notifyDigest: true,
  twoFactorApp: false,
  twoFactorSms: false,
  language: "English (US)",
  timezone: "Asia/Kolkata (IST)",
  dateFormat: "DD/MM/YYYY",
  currency: "INR (₹)",
  theme: "Black & Yellow",
  accent: "#f5c518",
  sidebarCollapsed: false,
  compactMode: false,
};

// ── Live appearance application ───────────────────────────────────────────────
const EVENT = "vda-settings-changed";

// In-memory cache for the signed-in user's settings (defaults until hydrated).
let cache: AppSettings = { ...DEFAULT_SETTINGS };

/** Read the current settings synchronously (defaults merged over the cache). */
export function getSettings(_phone?: string): AppSettings {
  return { ...DEFAULT_SETTINGS, ...cache };
}

/** Fetch the signed-in user's settings from the backend, apply, and notify. */
export async function hydrateSettings(): Promise<void> {
  try {
    const stored = await apiGetSettings<AppSettings>();
    cache = { ...DEFAULT_SETTINGS, ...stored };
  } catch {
    cache = { ...DEFAULT_SETTINGS };
  }
  applyAppearance(cache);
  window.dispatchEvent(new CustomEvent(EVENT, { detail: cache }));
}

/** Reset to defaults (e.g. on logout) and re-apply the default appearance. */
export function resetSettings(): void {
  cache = { ...DEFAULT_SETTINGS };
  applyAppearance(cache);
  window.dispatchEvent(new CustomEvent(EVENT, { detail: cache }));
}

export function saveSettings(_phone: string, settings: AppSettings): void {
  cache = settings;
  applyAppearance(settings);
  window.dispatchEvent(new CustomEvent(EVENT, { detail: settings }));
  apiPutSettings(settings).catch(() => hydrateSettings());
}

// Pick readable text/icon colour for a given accent background.
function contrastOn(hex: string): string {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
  if (!m) return "#0a0a0a";
  const n = parseInt(m[1], 16);
  const r = (n >> 16) & 255, g = (n >> 8) & 255, b = n & 255;
  // Perceived luminance — light accents get dark text, dark accents get white.
  const lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  return lum > 0.6 ? "#0a0a0a" : "#ffffff";
}

// Push the saved appearance into CSS variables that the whole app reads.
export function applyAppearance(settings: AppSettings): void {
  const root = document.documentElement;
  root.style.setProperty("--accent", settings.accent);
  root.style.setProperty("--accent-foreground", contrastOn(settings.accent));
  root.style.setProperty("--ring", settings.accent);
  // UI density: compact shrinks the global scale, normal keeps the comfortable size.
  root.style.setProperty("--ui-scale", settings.compactMode ? "1" : "1.12");
}

// Subscribe to live settings changes (fired on save). Returns an unsubscribe fn.
export function subscribeSettings(cb: (s: AppSettings) => void): () => void {
  const handler = (e: Event) => cb((e as CustomEvent<AppSettings>).detail);
  window.addEventListener(EVENT, handler);
  return () => window.removeEventListener(EVENT, handler);
}
