// Client for the PostgreSQL-backed persistence API (accounts, vehicles, claims,
// settings). All calls except signup/login require a bearer token, which is
// issued on signup/login and kept in localStorage for this device.

import { API_BASE_URL } from "./api";

const TOKEN_KEY = "vda_token";

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}
export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}
export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY);
}

function authHeaders(json = true): HeadersInit {
  const h: Record<string, string> = {};
  const t = getToken();
  if (t) h["Authorization"] = `Bearer ${t}`;
  if (json) h["Content-Type"] = "application/json";
  return h;
}

async function detail(res: Response, fallback: string): Promise<string> {
  try {
    const b = await res.json();
    if (b?.detail) return typeof b.detail === "string" ? b.detail : JSON.stringify(b.detail);
  } catch {
    /* keep fallback */
  }
  return `${fallback} (${res.status})`;
}

// ── Auth ──────────────────────────────────────────────────────────────────────

export interface PublicUser {
  name: string;
  phone: string;
}
export interface AuthResponse {
  token: string;
  user: PublicUser;
}

export async function apiSignup(name: string, phone: string, password: string): Promise<AuthResponse> {
  const res = await fetch(`${API_BASE_URL}/auth/signup`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, phone, password }),
  });
  if (!res.ok) throw new Error(await detail(res, "Sign up failed"));
  return res.json();
}

export async function apiLogin(phone: string, password: string): Promise<AuthResponse> {
  const res = await fetch(`${API_BASE_URL}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ phone, password }),
  });
  if (!res.ok) throw new Error(await detail(res, "Login failed"));
  return res.json();
}

export async function apiLogout(): Promise<void> {
  try {
    await fetch(`${API_BASE_URL}/auth/logout`, { method: "POST", headers: authHeaders() });
  } catch {
    /* best-effort; clearing the local token is what matters */
  }
}

export async function apiUpdateProfile(updates: Partial<PublicUser>): Promise<PublicUser> {
  const res = await fetch(`${API_BASE_URL}/auth/me`, {
    method: "PATCH",
    headers: authHeaders(),
    body: JSON.stringify(updates),
  });
  if (!res.ok) throw new Error(await detail(res, "Could not update profile"));
  return res.json();
}

export async function apiChangePassword(currentPassword: string, newPassword: string): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/auth/change-password`, {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify({ currentPassword, newPassword }),
  });
  if (!res.ok) throw new Error(await detail(res, "Could not change password"));
}

// ── Generic collection helpers (vehicles / claims share the same shape) ────────

async function listResource<T>(path: string): Promise<T[]> {
  const res = await fetch(`${API_BASE_URL}/${path}`, { headers: authHeaders(false) });
  if (!res.ok) throw new Error(await detail(res, `Could not load ${path}`));
  return res.json();
}

async function putResource<T extends { id: string }>(path: string, item: T): Promise<T> {
  const res = await fetch(`${API_BASE_URL}/${path}/${encodeURIComponent(item.id)}`, {
    method: "PUT",
    headers: authHeaders(),
    body: JSON.stringify(item),
  });
  if (!res.ok) throw new Error(await detail(res, `Could not save ${path}`));
  return res.json();
}

async function deleteResource(path: string, id: string): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/${path}/${encodeURIComponent(id)}`, {
    method: "DELETE",
    headers: authHeaders(false),
  });
  if (!res.ok) throw new Error(await detail(res, `Could not delete from ${path}`));
}

export const apiListVehicles = <T>() => listResource<T>("vehicles");
export const apiPutVehicle = <T extends { id: string }>(v: T) => putResource("vehicles", v);
export const apiDeleteVehicle = (id: string) => deleteResource("vehicles", id);

export const apiListClaims = <T>() => listResource<T>("claims");
export const apiPutClaim = <T extends { id: string }>(c: T) => putResource("claims", c);
export const apiDeleteClaim = (id: string) => deleteResource("claims", id);

export const apiListInsurance = <T>() => listResource<T>("insurance");
export const apiPutInsurance = <T extends { id: string }>(c: T) => putResource("insurance", c);
export const apiDeleteInsurance = (id: string) => deleteResource("insurance", id);

// ── Settings ──────────────────────────────────────────────────────────────────

export async function apiGetSettings<T>(): Promise<Partial<T>> {
  const res = await fetch(`${API_BASE_URL}/settings`, { headers: authHeaders(false) });
  if (!res.ok) throw new Error(await detail(res, "Could not load settings"));
  return res.json();
}

export async function apiPutSettings<T>(settings: T): Promise<T> {
  const res = await fetch(`${API_BASE_URL}/settings`, {
    method: "PUT",
    headers: authHeaders(),
    body: JSON.stringify(settings),
  });
  if (!res.ok) throw new Error(await detail(res, "Could not save settings"));
  return res.json();
}
