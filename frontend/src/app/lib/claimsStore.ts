// Claims backed by the PostgreSQL API. An in-memory cache preserves the
// synchronous getter surface the UI relies on; it is hydrated from the backend
// on login/startup and kept in sync on every write. The notification "seen"
// marker stays in localStorage — it is device-local UI state.

import { apiListClaims, apiPutClaim, apiDeleteClaim } from "./dataApi";

export interface ClaimDetection {
  index: number;
  bbox: number[]; // [x1,y1,x2,y2] in ORIGINAL image pixels
  damage: string;
  part: string;
  severity: string;
  confidence: number; // 0–1
}

export interface ClaimFinding {
  area: string; // prettified part, e.g. "Front Bumper"
  damage: string; // e.g. "Severe dent"
  severity: string; // Minor | Moderate | Severe
  confidence: number; // 0–100
  costMin: number;
  costMax: number;
}

export type ClaimStatus = "auto-approved" | "pending-review" | "no-damage";

export interface Claim {
  id: string; // e.g. CLM-4821
  createdAt: string; // ISO timestamp
  vehicle: string; // "2022 Honda Accord"
  vehicleId?: string;
  thumbnail: string; // downscaled data URL of the uploaded image
  maskThumbnail?: string; // downscaled data URL of the SAM2 damage-mask overlay
  mergedThumbnail?: string; // downscaled data URL of the merged (VLM ∪ SAM2) overlay
  imgW: number; // original image width (for accurate box overlay)
  imgH: number; // original image height
  detections: ClaimDetection[];
  findings: ClaimFinding[];
  totalMin: number;
  totalMax: number;
  approval: string; // raw backend decision
  status: ClaimStatus;
  inferenceS?: number; // pipeline inference time in seconds
}

export type SeverityLevel = "Low" | "Med" | "High";

const SEV_RANK: Record<string, number> = { minor: 1, moderate: 2, severe: 3 };

/** Highest severity across a claim's findings, mapped to Low/Med/High. */
export function claimSeverity(c: Claim): SeverityLevel | null {
  if (!c.findings.length) return null;
  const max = Math.max(...c.findings.map((f) => SEV_RANK[f.severity.toLowerCase()] ?? 0));
  return max >= 3 ? "High" : max === 2 ? "Med" : "Low";
}

/** Average detection confidence (0–100) across a claim. */
export function claimConfidence(c: Claim): number {
  const scored = c.findings.filter((f) => f.confidence > 0);
  if (!scored.length) return 0;
  return Math.round(scored.reduce((s, f) => s + f.confidence, 0) / scored.length);
}

/** Compact relative time, e.g. "2 min ago", "3 hrs ago", "5 days ago". */
export function relativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const m = Math.floor(diff / 60000);
  if (m < 1) return "just now";
  if (m < 60) return `${m} min ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h} hr${h > 1 ? "s" : ""} ago`;
  const d = Math.floor(h / 24);
  return `${d} day${d > 1 ? "s" : ""} ago`;
}

const EVENT = "vda-claims-updated";
const SEEN_KEY = "vda_notifs_seen";

let cache: Claim[] = [];

function emit(): void {
  window.dispatchEvent(new Event(EVENT));
}

function sortClaims(list: Claim[]): Claim[] {
  return [...list].sort((a, b) => b.createdAt.localeCompare(a.createdAt));
}

/** Fetch the signed-in user's claims from the backend into the cache. */
export async function hydrateClaims(): Promise<void> {
  try {
    cache = sortClaims(await apiListClaims<Claim>());
  } catch {
    cache = [];
  }
  emit();
}

/** Clear the cache (e.g. on logout). */
export function resetClaims(): void {
  cache = [];
  emit();
}

export function getClaims(): Claim[] {
  return cache;
}

export function getClaim(id: string): Claim | undefined {
  return cache.find((c) => c.id === id);
}

export function saveClaim(claim: Claim): void {
  cache = sortClaims([claim, ...cache.filter((c) => c.id !== claim.id)]);
  emit();
  apiPutClaim(claim).catch(() => hydrateClaims());
}

export function deleteClaim(id: string): void {
  cache = cache.filter((c) => c.id !== id);
  emit();
  apiDeleteClaim(id).catch(() => hydrateClaims());
}

/** Subscribe to claim changes. Returns an unsubscribe fn. */
export function subscribeClaims(cb: () => void): () => void {
  const handler = () => cb();
  window.addEventListener(EVENT, handler);
  return () => window.removeEventListener(EVENT, handler);
}

export function statusFromApproval(approval: string, detectionCount: number): ClaimStatus {
  if (approval === "ESCALATE_TO_HUMAN") return "pending-review";
  if (detectionCount === 0) return "no-damage";
  return "auto-approved";
}

export function newClaimId(): string {
  return `CLM-${Math.floor(1000 + Math.random() * 9000)}`;
}

// ── Notifications (derived from claims) ───────────────────────────────────────

/** Number of claims created since the user last opened notifications. */
export function getUnseenCount(): number {
  const seen = localStorage.getItem(SEEN_KEY) || "";
  return cache.filter((c) => c.createdAt > seen).length;
}

/** Mark all current claims as seen (clears the badge). */
export function markNotificationsSeen(): void {
  localStorage.setItem(SEEN_KEY, new Date().toISOString());
  emit();
}

export function lastSeen(): string {
  return localStorage.getItem(SEEN_KEY) || "";
}

/**
 * Downscale an image File to a JPEG data URL (max edge ~720px) for storage,
 * and return the ORIGINAL natural dimensions (needed for accurate box overlay).
 */
export function makeThumbnail(
  file: File,
  maxEdge = 720,
): Promise<{ dataUrl: string; w: number; h: number }> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      const w = img.naturalWidth;
      const h = img.naturalHeight;
      const scale = Math.min(1, maxEdge / Math.max(w, h));
      const canvas = document.createElement("canvas");
      canvas.width = Math.round(w * scale);
      canvas.height = Math.round(h * scale);
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        URL.revokeObjectURL(url);
        reject(new Error("Canvas not supported"));
        return;
      }
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL("image/jpeg", 0.72);
      URL.revokeObjectURL(url);
      resolve({ dataUrl, w, h });
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Could not load image for thumbnail"));
    };
    img.src = url;
  });
}

/**
 * Fetch an image URL (e.g. the backend's SAM2 mask / merged-overlay endpoints)
 * and downscale it to a JPEG data URL for storage on the claim. Returns null if
 * the fetch fails or the image can't be produced (e.g. SAM2 weights unavailable).
 */
export async function makeRemoteThumbnail(url: string, maxEdge = 900): Promise<string | null> {
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    const blob = await res.blob();
    return await new Promise<string | null>((resolve) => {
      const objUrl = URL.createObjectURL(blob);
      const img = new Image();
      img.onload = () => {
        const scale = Math.min(1, maxEdge / Math.max(img.naturalWidth, img.naturalHeight));
        const canvas = document.createElement("canvas");
        canvas.width = Math.round(img.naturalWidth * scale);
        canvas.height = Math.round(img.naturalHeight * scale);
        const ctx = canvas.getContext("2d");
        if (!ctx) {
          URL.revokeObjectURL(objUrl);
          resolve(null);
          return;
        }
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        URL.revokeObjectURL(objUrl);
        resolve(canvas.toDataURL("image/jpeg", 0.72));
      };
      img.onerror = () => {
        URL.revokeObjectURL(objUrl);
        resolve(null);
      };
      img.src = objUrl;
    });
  } catch {
    return null;
  }
}
