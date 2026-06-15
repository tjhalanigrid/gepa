import { useEffect, useState } from "react";
import {
  LayoutDashboard,
  Car,
  PlusCircle,
  Settings,
  Bell,
  FileText,
  ChevronLeft,
  ChevronRight,
  Shield,
  LogOut,
  ClipboardList,
  ClipboardCheck,
  ChevronUp,
  Check,
} from "lucide-react";
import type { VehicleRegistration } from "../types/vehicle";
import { initials, type AuthUser } from "../lib/authStore";
import { getUnseenCount, subscribeClaims } from "../lib/claimsStore";

interface NavItem {
  icon: React.ReactNode;
  label: string;
  page: string;
  badge?: number;
}

interface SidebarProps {
  currentPage: string;
  onNavigate: (page: string) => void;
  userVehicles: VehicleRegistration[];
  activeVehicle: VehicleRegistration | null;
  onSelectVehicle: (v: VehicleRegistration) => void;
  user?: AuthUser | null;
  onLogout?: () => void;
  initialCollapsed?: boolean;
}

const navItems: NavItem[] = [
  { icon: <LayoutDashboard size={20} />, label: "Dashboard", page: "dashboard" },
  { icon: <PlusCircle size={20} />, label: "Register Vehicle", page: "register" },
  { icon: <ClipboardList size={20} />, label: "Inspections", page: "inspections" },
  { icon: <ClipboardCheck size={20} />, label: "Claims", page: "claims" },
  { icon: <FileText size={20} />, label: "Reports", page: "reports" },
  { icon: <Shield size={20} />, label: "Insurance", page: "insurance" },
];

const bottomItems: NavItem[] = [
  { icon: <Bell size={20} />, label: "Notifications", page: "notifications" },
  { icon: <Settings size={20} />, label: "Settings", page: "settings" },
];

const statusDot: Record<string, string> = {
  active: "#10b981",
  inactive: "#9ca3af",
  maintenance: "#f97316",
  pending: "var(--accent)",
};

export function Sidebar({ currentPage, onNavigate, userVehicles, activeVehicle, onSelectVehicle, user, onLogout, initialCollapsed = false }: SidebarProps) {
  const [collapsed, setCollapsed] = useState(initialCollapsed);
  const [switcherOpen, setSwitcherOpen] = useState(false);
  const [unseen, setUnseen] = useState(getUnseenCount());

  useEffect(() => {
    const refresh = () => setUnseen(getUnseenCount());
    refresh();
    return subscribeClaims(refresh);
  }, []);

  return (
    <aside
      style={{
        width: collapsed ? "68px" : "260px",
        minWidth: collapsed ? "68px" : "260px",
        background: "#0a0a0a",
        transition: "width 0.2s ease, min-width 0.2s ease",
        display: "flex",
        flexDirection: "column",
        height: "100%",
        position: "relative",
        borderRight: "1px solid #1e1e1e",
      }}
    >
      {/* Logo */}
      <div
        style={{
          padding: collapsed ? "18px 0" : "18px 20px",
          borderBottom: "1px solid #1e1e1e",
          display: "flex",
          alignItems: "center",
          gap: "12px",
          justifyContent: collapsed ? "center" : "flex-start",
          minHeight: "64px",
        }}
      >
        <div
          style={{
            width: "34px",
            height: "34px",
            background: "var(--accent)",
            borderRadius: "8px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            flexShrink: 0,
          }}
        >
          <Car size={20} color="#0a0a0a" />
        </div>
        {!collapsed && (
          <div>
            <div style={{ color: "#ffffff", fontSize: "16px", fontWeight: 700, lineHeight: 1.2 }}>VehicleDamageAI</div>
            <div style={{ color: "#444440", fontSize: "11px", letterSpacing: "0.05em" }}>VEHICLE REGISTRY</div>
          </div>
        )}
      </div>

      {/* Collapse toggle */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        style={{
          position: "absolute",
          right: "-12px",
          top: "76px",
          width: "24px",
          height: "24px",
          background: "var(--accent)",
          border: "none",
          borderRadius: "50%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          cursor: "pointer",
          zIndex: 10,
          color: "#0a0a0a",
        }}
      >
        {collapsed ? <ChevronRight size={12} /> : <ChevronLeft size={12} />}
      </button>

      {/* Section label */}
      {!collapsed && (
        <div style={{ padding: "18px 22px 10px", color: "#333330", fontSize: "11px", letterSpacing: "0.08em", fontWeight: 600 }}>
          MAIN MENU
        </div>
      )}

      {/* Nav */}
      <nav style={{ flex: 1, padding: collapsed ? "10px 0" : "0", overflowY: "auto" }}>
        {navItems.map((item) => {
          const active = currentPage === item.page;
          return (
            <NavButton key={item.page} item={item} active={active} collapsed={collapsed} onNavigate={onNavigate} />
          );
        })}

        {!collapsed && (
          <div style={{ padding: "18px 22px 10px", color: "#333330", fontSize: "11px", letterSpacing: "0.08em", fontWeight: 700, marginTop: "6px" }}>
            SYSTEM
          </div>
        )}

        {bottomItems.map((item) => {
          const active = currentPage === item.page;
          const withBadge = item.page === "notifications" && unseen > 0 ? { ...item, badge: unseen } : item;
          return (
            <NavButton key={item.page} item={withBadge} active={active} collapsed={collapsed} onNavigate={onNavigate} />
          );
        })}
      </nav>

      {/* Vehicle Switcher */}
      <div style={{ borderTop: "1px solid #1e1e1e", position: "relative" }}>
        {/* Switcher popup */}
        {switcherOpen && !collapsed && (
          <div
            style={{
              position: "absolute",
              bottom: "calc(100% + 8px)",
              left: "12px",
              right: "12px",
              background: "#1a1a1a",
              border: "1px solid #2a2a2a",
              borderRadius: "12px",
              overflow: "hidden",
              boxShadow: "0 -12px 40px rgba(0,0,0,0.7)",
              zIndex: 20,
            }}
          >
            <div style={{ padding: "12px 16px 8px", fontSize: "11px", fontWeight: 700, color: "#444440", letterSpacing: "0.08em" }}>
              YOUR VEHICLES
            </div>
            {userVehicles.map((v) => {
              const isActive = activeVehicle?.id === v.id;
              return (
                <button
                  key={v.id}
                  onClick={() => { onSelectVehicle(v); setSwitcherOpen(false); }}
                  style={{
                    width: "100%",
                    display: "flex",
                    alignItems: "center",
                    gap: "12px",
                    padding: "11px 16px",
                    background: isActive ? "rgba(245,197,24,0.08)" : "transparent",
                    border: "none",
                    cursor: "pointer",
                    textAlign: "left",
                  }}
                  onMouseEnter={(e) => !isActive && (e.currentTarget.style.background = "#222222")}
                  onMouseLeave={(e) => !isActive && (e.currentTarget.style.background = "transparent")}
                >
                  <div
                    style={{
                      width: "32px",
                      height: "32px",
                      background: isActive ? "var(--accent)" : "#2a2a2a",
                      borderRadius: "6px",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: "11px",
                      fontWeight: 800,
                      color: isActive ? "#0a0a0a" : "#888882",
                      flexShrink: 0,
                    }}
                  >
                    {v.make.slice(0, 2).toUpperCase()}
                  </div>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: "14px", fontWeight: 500, color: isActive ? "var(--accent)" : "#cccccc", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                      {v.year} {v.make} {v.model}
                    </div>
                    <div style={{ fontSize: "11px", color: "#444440", marginTop: "2px" }}>{v.licenseplate}</div>
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                    <span style={{ width: "7px", height: "7px", borderRadius: "50%", background: statusDot[v.status] ?? "#888882", flexShrink: 0 }} />
                    {isActive && <Check size={14} color="var(--accent)" />}
                  </div>
                </button>
              );
            })}
          </div>
        )}

        {/* Active vehicle button */}
        <button
          onClick={() => setSwitcherOpen(!switcherOpen)}
          title={collapsed ? activeVehicle ? `${activeVehicle.make} ${activeVehicle.model}` : "Select Vehicle" : undefined}
          style={{
            width: "100%",
            display: "flex",
            alignItems: "center",
            gap: "12px",
            padding: collapsed ? "16px 0" : "14px 18px",
            justifyContent: collapsed ? "center" : "flex-start",
            background: "transparent",
            border: "none",
            cursor: "pointer",
            borderBottom: "1px solid #1e1e1e",
          }}
          onMouseEnter={(e) => (e.currentTarget.style.background = "#111111")}
          onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
        >
          {activeVehicle ? (
            <>
              <div
                style={{
                  width: "34px",
                  height: "34px",
                  background: "var(--accent)",
                  borderRadius: "7px",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: "11px",
                  fontWeight: 800,
                  color: "#0a0a0a",
                  flexShrink: 0,
                }}
              >
                {activeVehicle.make.slice(0, 2).toUpperCase()}
              </div>
              {!collapsed && (
                <>
                  <div style={{ flex: 1, minWidth: 0, textAlign: "left" }}>
                    <div style={{ fontSize: "13px", color: "#ffffff", fontWeight: 600, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                      {activeVehicle.make} {activeVehicle.model}
                    </div>
                    <div style={{ fontSize: "11px", color: "#444440", marginTop: "1px" }}>{activeVehicle.licenseplate}</div>
                  </div>
                  <ChevronUp size={15} color={switcherOpen ? "var(--accent)" : "#444440"} style={{ transform: switcherOpen ? "none" : "rotate(180deg)", transition: "transform 0.2s" }} />
                </>
              )}
            </>
          ) : (
            <>
              <div
                style={{
                  width: "34px",
                  height: "34px",
                  background: "#1a1a1a",
                  borderRadius: "7px",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  flexShrink: 0,
                  border: "1px dashed #2a2a2a",
                }}
              >
                <Car size={15} color="#444440" />
              </div>
              {!collapsed && (
                <span style={{ fontSize: "13px", color: "#444440", fontWeight: 500 }}>Select vehicle…</span>
              )}
            </>
          )}
        </button>
      </div>

      {/* User profile */}
      <div
        style={{
          padding: collapsed ? "14px 0" : "14px 18px",
          display: "flex",
          alignItems: "center",
          gap: "12px",
          justifyContent: collapsed ? "center" : "flex-start",
        }}
      >
        <div
          style={{
            width: "34px",
            height: "34px",
            borderRadius: "50%",
            background: "var(--accent)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "#0a0a0a",
            fontSize: "12px",
            fontWeight: 800,
            flexShrink: 0,
          }}
        >
          {user ? initials(user.name) : "U"}
        </div>
        {!collapsed && (
          <>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ color: "#ffffff", fontSize: "13px", fontWeight: 600, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                {user?.name ?? "Guest"}
              </div>
              <div style={{ color: "#444440", fontSize: "11px", marginTop: "1px" }}>{user?.phone ?? ""}</div>
            </div>
            <button
              onClick={onLogout}
              style={{ background: "transparent", border: "none", cursor: "pointer", color: "#444440", padding: "4px", display: "flex", alignItems: "center" }}
              title="Logout"
              onMouseEnter={(e) => (e.currentTarget.style.color = "var(--accent)")}
              onMouseLeave={(e) => (e.currentTarget.style.color = "#444440")}
            >
              <LogOut size={15} />
            </button>
          </>
        )}
      </div>
    </aside>
  );
}

function NavButton({ item, active, collapsed, onNavigate }: { item: NavItem; active: boolean; collapsed: boolean; onNavigate: (p: string) => void }) {
  return (
    <button
      onClick={() => onNavigate(item.page)}
      title={collapsed ? item.label : undefined}
      style={{
        width: "100%",
        display: "flex",
        alignItems: "center",
        gap: "12px",
        padding: collapsed ? "12px 0" : "11px 20px",
        justifyContent: collapsed ? "center" : "flex-start",
        background: active ? "#1a1a1a" : "transparent",
        border: "none",
        cursor: "pointer",
        color: active ? "var(--accent)" : "#666660",
        borderLeft: active ? "2px solid var(--accent)" : "2px solid transparent",
        transition: "all 0.12s ease",
        fontSize: "14px",
        fontWeight: active ? 600 : 400,
        textAlign: "left",
      }}
      onMouseEnter={(e) => {
        if (!active) {
          (e.currentTarget as HTMLButtonElement).style.color = "#cccccc";
          (e.currentTarget as HTMLButtonElement).style.background = "#111111";
        }
      }}
      onMouseLeave={(e) => {
        if (!active) {
          (e.currentTarget as HTMLButtonElement).style.color = "#666660";
          (e.currentTarget as HTMLButtonElement).style.background = "transparent";
        }
      }}
    >
      <span style={{ flexShrink: 0 }}>{item.icon}</span>
      {!collapsed && <span>{item.label}</span>}
      {!collapsed && item.badge && (
        <span style={{ marginLeft: "auto", background: "var(--accent)", color: "#0a0a0a", fontSize: "10px", fontWeight: 700, borderRadius: "10px", padding: "2px 7px" }}>
          {item.badge}
        </span>
      )}
    </button>
  );
}