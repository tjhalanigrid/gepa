// Motor-insurance claim forms backed by the PostgreSQL API. Cache-backed like the
// other stores so getters stay synchronous; hydrated on login/startup.

import { apiListInsurance, apiPutInsurance, apiDeleteInsurance } from "./dataApi";

export type InsuranceStatus = "draft" | "submitted";

// Mirrors the motor-insurance claim form.
export interface InsuranceClaim {
  id: string; // e.g. INS-4821
  createdAt: string; // ISO
  vehicleId?: string;
  status: InsuranceStatus;

  // Policy / vehicle header
  policyNo: string;
  vehicleNo: string;
  engineNo: string;
  chassisNo: string;

  // 1) Insured details
  name: string;
  address: string;
  mobile: string;
  email: string;
  otherInsurance: string;

  // 2) Loss details
  accidentDate: string; // datetime-local string
  placeOfLoss: string;
  lossDamage: boolean;
  lossTheft: boolean;
  lossThirdParty: boolean;
  estimatedRepairCost: string;
  accidentDescription: string;

  // 3) Driver details
  driverName: string;
  driverAge: string;
  driverType: "" | "owner" | "paid" | "relative";
  licenseNo: string;
  licenseValidUpto: string;
  authorisedToDrive: string;
  issuingAuthority: string;

  // 4) Commercial vehicle details
  permitNo: string;
  permitValidUpto: string;
  permitIssuingAuthority: string;
  fitnessValidUpto: string;
  passengersCarried: string;
  goodsWeightNature: string;
  grLrNo: string;

  // 5) Injury / death & police report
  policeReportLodged: "" | "yes" | "no";
  firNo: string;
  policeStation: string;
  injuryOrDeath: "" | "yes" | "no";
  injuryDetails: string;

  // 6) Declaration
  declarationDate: string;
  declarationPlace: string;
  signature: string;
  agreed: boolean;
}

const EVENT = "vda-insurance-updated";
let cache: InsuranceClaim[] = [];

function emit(): void {
  window.dispatchEvent(new Event(EVENT));
}

function sortByDate(list: InsuranceClaim[]): InsuranceClaim[] {
  return [...list].sort((a, b) => b.createdAt.localeCompare(a.createdAt));
}

export async function hydrateInsurance(): Promise<void> {
  try {
    cache = sortByDate(await apiListInsurance<InsuranceClaim>());
  } catch {
    cache = [];
  }
  emit();
}

export function resetInsurance(): void {
  cache = [];
  emit();
}

export function getInsuranceClaims(): InsuranceClaim[] {
  return cache;
}

export function getInsuranceClaim(id: string | undefined): InsuranceClaim | undefined {
  if (!id) return undefined;
  return cache.find((c) => c.id === id);
}

export function saveInsuranceClaim(claim: InsuranceClaim): void {
  cache = sortByDate([claim, ...cache.filter((c) => c.id !== claim.id)]);
  emit();
  apiPutInsurance(claim).catch(() => hydrateInsurance());
}

export function deleteInsuranceClaim(id: string): void {
  cache = cache.filter((c) => c.id !== id);
  emit();
  apiDeleteInsurance(id).catch(() => hydrateInsurance());
}

export function subscribeInsurance(cb: () => void): () => void {
  const handler = () => cb();
  window.addEventListener(EVENT, handler);
  return () => window.removeEventListener(EVENT, handler);
}

export function newInsuranceId(): string {
  return `INS-${Math.floor(1000 + Math.random() * 9000)}`;
}

/** A blank form, optionally prefilled from a vehicle + user. */
export function emptyInsuranceClaim(prefill?: Partial<InsuranceClaim>): InsuranceClaim {
  const today = new Date().toISOString().slice(0, 10);
  return {
    id: newInsuranceId(),
    createdAt: new Date().toISOString(),
    status: "draft",
    policyNo: "",
    vehicleNo: "",
    engineNo: "",
    chassisNo: "",
    name: "",
    address: "",
    mobile: "",
    email: "",
    otherInsurance: "",
    accidentDate: "",
    placeOfLoss: "",
    lossDamage: false,
    lossTheft: false,
    lossThirdParty: false,
    estimatedRepairCost: "",
    accidentDescription: "",
    driverName: "",
    driverAge: "",
    driverType: "",
    licenseNo: "",
    licenseValidUpto: "",
    authorisedToDrive: "",
    issuingAuthority: "",
    permitNo: "",
    permitValidUpto: "",
    permitIssuingAuthority: "",
    fitnessValidUpto: "",
    passengersCarried: "",
    goodsWeightNature: "",
    grLrNo: "",
    policeReportLodged: "",
    firNo: "",
    policeStation: "",
    injuryOrDeath: "",
    injuryDetails: "",
    declarationDate: today,
    declarationPlace: "",
    signature: "",
    agreed: false,
    ...prefill,
  };
}
