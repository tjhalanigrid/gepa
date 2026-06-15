// Vehicle registry backed by the PostgreSQL API. An in-memory cache keeps the
// synchronous getter surface (getVehicles) that the UI relies on; the cache is
// hydrated from the backend on login/startup and kept in sync on every write.

import type { VehicleRegistration } from "../types/vehicle";
import { apiListVehicles, apiPutVehicle, apiDeleteVehicle } from "./dataApi";

const EVENT = "vda-vehicles-updated";

let cache: VehicleRegistration[] = [];

function emit(): void {
  window.dispatchEvent(new Event(EVENT));
}

/** Fetch the signed-in user's vehicles from the backend into the cache. */
export async function hydrateVehicles(): Promise<void> {
  try {
    cache = await apiListVehicles<VehicleRegistration>();
  } catch {
    cache = [];
  }
  emit();
}

/** Clear the cache (e.g. on logout). */
export function resetVehicles(): void {
  cache = [];
  emit();
}

export function getVehicles(): VehicleRegistration[] {
  return cache;
}

export function getVehicle(id: string | undefined): VehicleRegistration | undefined {
  if (!id) return undefined;
  return cache.find((v) => v.id === id);
}

export function saveVehicle(v: VehicleRegistration): void {
  // Optimistic local update, then persist.
  cache = [v, ...cache.filter((x) => x.id !== v.id)];
  emit();
  apiPutVehicle(v).catch(() => hydrateVehicles());
}

export function deleteVehicle(id: string): void {
  cache = cache.filter((v) => v.id !== id);
  emit();
  apiDeleteVehicle(id).catch(() => hydrateVehicles());
}

export function subscribeVehicles(cb: () => void): () => void {
  const handler = () => cb();
  window.addEventListener(EVENT, handler);
  return () => window.removeEventListener(EVENT, handler);
}

export function newVehicleId(): string {
  return `V${Date.now().toString().slice(-6)}`;
}

export function newRegistrationNumber(): string {
  return `REG-${new Date().getFullYear()}-${Math.floor(100 + Math.random() * 900)}`;
}
