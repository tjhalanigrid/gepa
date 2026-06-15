// API client for the vehicle damage assessment backend (backend/app.py).
// Base URL is configurable via VITE_API_URL; defaults to the local FastAPI port.
// The backend enables permissive CORS, so the browser can call it directly.

export const API_BASE_URL: string =
  (import.meta as any).env?.VITE_API_URL?.replace(/\/$/, "") || "http://localhost:8000";

// ── Response contracts — mirror pipeline/schema.py ────────────────────────────

export interface DamagePartEntry {
  damage: string; // dent | scratch | crack | glass_shatter | lamp_broken | tire_flat
  part: string; // front_bumper, hood, left_headlight, …
  severity: string; // minor | moderate | severe
  cost_min: number; // INR
  cost_max: number; // INR
}

export interface DetectionWithBBox {
  index: number;
  bbox: number[]; // [x1, y1, x2, y2] pixels
  damage: string;
  part: string;
  severity: string;
  confidence: number; // 0.0–1.0
}

export interface FinalDamageReport {
  image_path: string;
  damage_part_map: DamagePartEntry[];
  detections_with_bbox: DetectionWithBBox[];
  merged_detections: Record<string, unknown>[];
  total_min: number;
  total_max: number;
  currency: string;
  approval_decision: string; // AUTO_APPROVED | ESCALATE_TO_HUMAN | UNKNOWN
  tool_call_log: unknown[];
  iterations: Record<string, unknown>[];
  total_inference_s: number;
  warnings: string[];
  raw_vlm_response: string | null;
  annotated_image_path: string | null;
  merged_image_path: string | null;
  masked_image_path: string | null;
}

// A completed job's `result` is the report directly, OR wraps it when escalated.
export interface PendingReviewResult {
  session_id: string;
  report: FinalDamageReport;
  status: "pending_review";
}

export type AssessResult = FinalDamageReport | PendingReviewResult;

function isPendingReview(r: AssessResult): r is PendingReviewResult {
  return (r as PendingReviewResult).status === "pending_review";
}

/** Normalises either job-result shape into the underlying report. */
export function unwrapReport(r: AssessResult): FinalDamageReport {
  return isPendingReview(r) ? r.report : r;
}

// POST /assess is async: it returns a job id you poll via GET /job/{id}.
interface AssessJobHandle {
  job_id: string;
  status: string;
}

interface JobStatus {
  status: "queued" | "processing" | "complete" | "failed";
  elapsed_s?: number;
  result?: AssessResult;
  error?: string;
}

export interface HealthStatus {
  status: "ready" | "warming_up";
  vlm_loaded: boolean;
}

// ── Calls ─────────────────────────────────────────────────────────────────────

export async function checkHealth(): Promise<HealthStatus> {
  const res = await fetch(`${API_BASE_URL}/health`);
  if (!res.ok) throw new Error(`Health check failed (${res.status})`);
  return res.json();
}

async function parseError(res: Response, fallback: string): Promise<string> {
  let detail = `${fallback} (${res.status})`;
  try {
    const body = await res.json();
    if (body?.detail) detail = body.detail;
  } catch {
    /* non-JSON error body — keep the status message */
  }
  return detail;
}

/**
 * Submit a single vehicle image to the AI pipeline and wait for the result.
 *
 * The backend processes one image per call. `/assess` is asynchronous: it
 * returns a job id immediately and the pipeline runs in a threadpool, so this
 * helper polls `GET /job/{id}` until the job completes or fails.
 *
 * @param onProgress optional callback receiving elapsed seconds while polling.
 */
export async function assessDamage(
  image: File,
  opts: { claimId?: string; vehicleId?: string; onJobStart?: (jobId: string) => void } = {},
  onProgress?: (elapsedS: number) => void,
  signal?: AbortSignal,
): Promise<AssessResult> {
  const form = new FormData();
  form.append("image", image);
  if (opts.claimId) form.append("claim_id", opts.claimId);
  if (opts.vehicleId) form.append("vehicle_id", opts.vehicleId);

  // 1. Kick off the job.
  const start = await fetch(`${API_BASE_URL}/assess`, {
    method: "POST",
    body: form,
    signal,
  });
  if (!start.ok) throw new Error(await parseError(start, "Assessment failed"));
  const handle: AssessJobHandle = await start.json();
  // Expose the job id so callers can fetch job-scoped images (e.g. SAM2 masks).
  opts.onJobStart?.(handle.job_id);

  // 2. Poll until the job is done. VLM inference can take 30–90s.
  const POLL_MS = 2000;
  const MAX_WAIT_MS = 12 * 60 * 1000; // 12 min ceiling client-side
  const began = Date.now();

  while (Date.now() - began < MAX_WAIT_MS) {
    await new Promise((r) => setTimeout(r, POLL_MS));
    if (signal?.aborted) throw new Error("Assessment cancelled.");

    const poll = await fetch(`${API_BASE_URL}/job/${handle.job_id}`, { signal });
    if (!poll.ok) throw new Error(await parseError(poll, "Job lookup failed"));
    const job: JobStatus = await poll.json();

    if (job.status === "complete" && job.result) return job.result;
    if (job.status === "failed") throw new Error(job.error || "Assessment failed.");
    if (onProgress && typeof job.elapsed_s === "number") onProgress(job.elapsed_s);
  }

  throw new Error("Assessment timed out waiting for the AI pipeline.");
}

/** URL of the SAM2 damage-mask overlay for a completed job. */
export function jobMaskedImageUrl(jobId: string): string {
  return `${API_BASE_URL}/job/${encodeURIComponent(jobId)}/masked_image`;
}

/** URL of the merged-union (VLM ∪ SAM2) box overlay for a completed job. */
export function jobMergedImageUrl(jobId: string): string {
  return `${API_BASE_URL}/job/${encodeURIComponent(jobId)}/merged_image`;
}

// ── Display helpers ─────────────────────────────────────────────────────────

/** "front_bumper" → "Front Bumper". */
export function prettifyLabel(snake: string): string {
  return snake
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

/** "minor" → "Minor". */
export function capitalize(s: string): string {
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : s;
}

const inr = new Intl.NumberFormat("en-IN", {
  style: "currency",
  currency: "INR",
  maximumFractionDigits: 0,
});

/** Format an INR cost range, e.g. "₹12,000 – ₹18,000". */
export function formatCostRange(min: number, max: number): string {
  if (min === max) return inr.format(min);
  return `${inr.format(min)} – ${inr.format(max)}`;
}
