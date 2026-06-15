// Auth backed by the PostgreSQL API. Accounts and password hashes live on the
// server; this module keeps a bearer token + a cached copy of the public user
// (name/phone) in localStorage so `getCurrentUser()` can stay synchronous for
// initial render. Sign in/up and profile changes are async (they hit the API).

import {
  apiSignup,
  apiLogin,
  apiLogout,
  apiUpdateProfile,
  apiChangePassword,
  setToken,
  clearToken,
  getToken,
} from "./dataApi";

export interface AuthUser {
  name: string;
  phone: string;
}

const CURRENT_KEY = "vda_current_user";

export interface AuthResult {
  ok: boolean;
  error?: string;
  user?: AuthUser;
}

function cacheUser(user: AuthUser): void {
  localStorage.setItem(CURRENT_KEY, JSON.stringify(user));
}

export async function signup(name: string, phone: string, password: string): Promise<AuthResult> {
  try {
    const { token, user } = await apiSignup(name, phone, password);
    setToken(token);
    cacheUser(user);
    return { ok: true, user };
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : "Sign up failed." };
  }
}

export async function login(phone: string, password: string): Promise<AuthResult> {
  try {
    const { token, user } = await apiLogin(phone, password);
    setToken(token);
    cacheUser(user);
    return { ok: true, user };
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : "Login failed." };
  }
}

export function logout(): void {
  // Clear local state immediately; revoke the server session best-effort.
  void apiLogout();
  clearToken();
  localStorage.removeItem(CURRENT_KEY);
}

export function getCurrentUser(): AuthUser | null {
  // A session requires both a token and a cached user record.
  if (!getToken()) return null;
  try {
    const raw = localStorage.getItem(CURRENT_KEY);
    return raw ? (JSON.parse(raw) as AuthUser) : null;
  } catch {
    return null;
  }
}

export async function updateProfile(updates: Partial<AuthUser>): Promise<AuthResult> {
  try {
    const user = await apiUpdateProfile(updates);
    cacheUser(user);
    return { ok: true, user };
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : "Could not update profile." };
  }
}

export async function changePassword(currentPassword: string, newPassword: string): Promise<AuthResult> {
  try {
    await apiChangePassword(currentPassword, newPassword);
    return { ok: true };
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : "Could not change password." };
  }
}

export function initials(name: string): string {
  return (
    name
      .split(" ")
      .map((p) => p[0])
      .filter(Boolean)
      .slice(0, 2)
      .join("")
      .toUpperCase() || "U"
  );
}
