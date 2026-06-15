export type VehicleStatus = "active" | "inactive" | "maintenance" | "pending";

export type DamageSeverity = "none" | "minor" | "moderate" | "severe";

export interface DamageImage {
  side: "front" | "left" | "right" | "back";
  file: File | null;
  preview: string | null;
  notes: string;
}

export interface VehicleRegistration {
  id: string;
  registrationNumber: string;
  make: string;
  model: string;
  year: number;
  color: string;
  vin: string;
  ownerName: string;
  ownerPhone: string;
  ownerEmail: string;
  licenseplate: string;
  status: VehicleStatus;
  registrationDate: string;
  expiryDate: string;
  insuranceProvider: string;
  insurancePolicyNumber: string;
  mileage: number;
  damageImages: DamageImage[];
  notes: string;
}

export interface DashboardStats {
  totalVehicles: number;
  activeVehicles: number;
  pendingRegistrations: number;
  maintenanceDue: number;
  recentRegistrations: VehicleRegistration[];
}
