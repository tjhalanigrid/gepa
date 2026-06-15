import { useEffect, useState } from "react";
import { LandingPage } from "./features/auth/LandingPage";
import { LoginPage } from "./features/auth/LoginPage";
import { SignupPage } from "./features/auth/SignupPage";
import { ForgotPasswordPage } from "./features/auth/ForgotPasswordPage";
import { Sidebar } from "./layouts/Sidebar";
import { TopBar } from "./layouts/TopBar";
import { DashboardPage } from "./features/dashboard/DashboardPage";
import { VehicleListPage } from "./features/vehicles/VehicleListPage";
import { RegisterVehiclePage } from "./features/vehicles/RegisterVehiclePage";
import { VehicleDetailPage } from "./features/vehicles/VehicleDetailPage";
import { InspectionPage } from "./features/inspections/InspectionPage";
import { ClaimsPage } from "./features/claims/ClaimsPage";
import { ReportsPage } from "./features/reports/ReportsPage";
import { InsurancePage } from "./features/insurance/InsurancePage";
import { NotificationsPage } from "./features/notifications/NotificationsPage";
import { SettingsPage } from "./features/settings/SettingsPage";
import { getVehicles, subscribeVehicles, hydrateVehicles, resetVehicles } from "./lib/vehiclesStore";
import { hydrateClaims, resetClaims } from "./lib/claimsStore";
import { hydrateInsurance, resetInsurance } from "./lib/insuranceStore";
import { getCurrentUser, logout, type AuthUser } from "./lib/authStore";
import { getSettings, subscribeSettings, hydrateSettings, resetSettings, type AppSettings } from "./lib/settingsStore";
import type { VehicleRegistration } from "./types/vehicle";

type Page = string;

const AUTH_PAGES = new Set(["landing", "login", "signup", "forgot-password"]);

const pageMeta: Record<string, { title: string; breadcrumb: string[] }> = {
  dashboard: { title: "Dashboard", breadcrumb: ["Home"] },
  vehicles: { title: "Vehicle Registry", breadcrumb: ["Home", "Vehicles"] },
  register: { title: "Register Vehicle", breadcrumb: ["Home", "Register"] },
  detail: { title: "Vehicle Detail", breadcrumb: ["Home", "Vehicles", "Detail"] },
  inspections: { title: "Inspections", breadcrumb: ["Home", "Inspections"] },
  claims: { title: "Claims", breadcrumb: ["Home", "Claims"] },
  reports: { title: "Reports", breadcrumb: ["Home", "Reports"] },
  insurance: { title: "Insurance", breadcrumb: ["Home", "Insurance"] },
  notifications: { title: "Notifications", breadcrumb: ["Home", "Notifications"] },
  settings: { title: "Settings", breadcrumb: ["Home", "Settings"] },
};

const MAIN_PAGES = ["dashboard", "vehicles", "register", "detail", "inspections", "claims", "reports", "insurance", "notifications", "settings"];

function PlaceholderPage({ title }: { title: string }) {
  return (
    <div style={{ padding: "24px", display: "flex", alignItems: "center", justifyContent: "center", minHeight: "60vh" }}>
      <div style={{ textAlign: "center" }}>
        <div style={{ width: "52px", height: "52px", background: "var(--accent)", borderRadius: "14px", display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 14px", fontSize: "20px" }}>📋</div>
        <h2 style={{ fontSize: "16px", fontWeight: 700, color: "#0a0a0a", marginBottom: "5px" }}>{title}</h2>
        <p style={{ fontSize: "12px", color: "#888882" }}>This section is coming soon.</p>
      </div>
    </div>
  );
}

export default function App() {
  const [user, setUser] = useState<AuthUser | null>(getCurrentUser());
  const [currentPage, setCurrentPage] = useState<Page>(getCurrentUser() ? "dashboard" : "landing");
  const [selectedVehicleId, setSelectedVehicleId] = useState<string | undefined>();
  const [vehicles, setVehicles] = useState<VehicleRegistration[]>(getVehicles());
  const [activeVehicle, setActiveVehicle] = useState<VehicleRegistration | null>(getVehicles()[0] ?? null);
  const [settings, setSettings] = useState<AppSettings>(() => getSettings(getCurrentUser()?.phone ?? ""));

  // Keep vehicles + active selection in sync with the store.
  useEffect(() => {
    return subscribeVehicles(() => {
      const v = getVehicles();
      setVehicles(v);
      setActiveVehicle((prev) => {
        const match = prev ? v.find((x) => x.id === prev.id) : undefined;
        return match ?? prev ?? v[0] ?? null;
      });
    });
  }, []);

  // Hydrate the signed-in user's data (vehicles, claims, settings) from the
  // backend on load and whenever the account changes; clear it on logout.
  // hydrateSettings applies the appearance and emits a settings-change event,
  // which the subscription below mirrors into local state.
  useEffect(() => {
    const unsub = subscribeSettings((next) => setSettings(next));
    if (user) {
      hydrateVehicles();
      hydrateClaims();
      hydrateInsurance();
      hydrateSettings();
    } else {
      resetVehicles();
      resetClaims();
      resetInsurance();
      resetSettings();
    }
    return unsub;
  }, [user?.phone]);

  function handleNavigate(page: string, vehicleId?: string) {
    setCurrentPage(page);
    if (vehicleId) setSelectedVehicleId(vehicleId);
  }

  function handleAuth(u: AuthUser) {
    setUser(u);
    setCurrentPage("dashboard");
  }

  function handleLogout() {
    logout();
    setUser(null);
    setCurrentPage("landing");
  }

  // ── Not authenticated → auth pages only ────────────────────────────────────
  if (!user) {
    if (currentPage === "login") return <LoginPage onNavigate={handleNavigate} onAuth={handleAuth} />;
    if (currentPage === "signup") return <SignupPage onNavigate={handleNavigate} onAuth={handleAuth} />;
    if (currentPage === "forgot-password") return <ForgotPasswordPage onNavigate={handleNavigate} />;
    return <LandingPage onNavigate={handleNavigate} />;
  }

  // Authenticated: never show an auth page.
  const page = AUTH_PAGES.has(currentPage) ? "dashboard" : currentPage;
  const meta = pageMeta[page] ?? { title: page, breadcrumb: ["Home"] };

  // Uniform UI scale — bumps every page's font size + spacing together.
  // Compact mode (from Appearance settings) drops it to a denser 1.0.
  const UI_SCALE = settings.compactMode ? 1 : 1.12;

  return (
    <div style={{ display: "flex", height: `calc(100vh / ${UI_SCALE})`, overflow: "hidden", background: "#f5f5f0", fontFamily: "'Inter', system-ui, -apple-system, sans-serif", zoom: UI_SCALE as unknown as number }}>
      <Sidebar
        currentPage={page}
        onNavigate={handleNavigate}
        userVehicles={vehicles}
        activeVehicle={activeVehicle}
        onSelectVehicle={(v) => setActiveVehicle(v)}
        user={user}
        onLogout={handleLogout}
        initialCollapsed={settings.sidebarCollapsed}
      />

      <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
        <TopBar title={meta.title} breadcrumb={meta.breadcrumb} onNavigate={handleNavigate} />

        <main style={{ flex: 1, overflowY: "auto", background: "#f5f5f0" }}>
          {page === "dashboard" && <DashboardPage onNavigate={handleNavigate} activeVehicle={activeVehicle} />}
          {page === "vehicles" && <VehicleListPage onNavigate={handleNavigate} />}
          {page === "register" && <RegisterVehiclePage onNavigate={handleNavigate} />}
          {page === "detail" && <VehicleDetailPage vehicleId={selectedVehicleId} onNavigate={handleNavigate} />}
          {page === "inspections" && <InspectionPage onNavigate={handleNavigate} activeVehicle={activeVehicle} />}
          {page === "claims" && <ClaimsPage onNavigate={handleNavigate} activeVehicle={activeVehicle} />}
          {page === "reports" && <ReportsPage activeVehicle={activeVehicle} onNavigate={handleNavigate} />}
          {page === "insurance" && <InsurancePage activeVehicle={activeVehicle} user={user} />}
          {page === "notifications" && <NotificationsPage onNavigate={handleNavigate} />}
          {page === "settings" && <SettingsPage />}
          {!MAIN_PAGES.includes(page) && <PlaceholderPage title={meta.title} />}
        </main>
      </div>
    </div>
  );
}
