import { useEffect, useMemo, useRef, useState } from "react";
import { Bell, Search, ChevronRight, Car, ClipboardCheck } from "lucide-react";
import { getVehicles, subscribeVehicles } from "../lib/vehiclesStore";
import { getClaims, subscribeClaims, getUnseenCount } from "../lib/claimsStore";
import type { VehicleRegistration } from "../types/vehicle";
import type { Claim } from "../lib/claimsStore";

interface TopBarProps {
  title: string;
  breadcrumb?: string[];
  onNavigate: (page: string, vehicleId?: string) => void;
}

export function TopBar({ title, breadcrumb, onNavigate }: TopBarProps) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [vehicles, setVehicles] = useState<VehicleRegistration[]>(getVehicles());
  const [claims, setClaims] = useState<Claim[]>(getClaims());
  const [unseen, setUnseen] = useState(getUnseenCount());
  const wrapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const refresh = () => {
      setVehicles(getVehicles());
      setClaims(getClaims());
      setUnseen(getUnseenCount());
    };
    refresh();
    const unsubV = subscribeVehicles(refresh);
    const unsubC = subscribeClaims(refresh);
    return () => { unsubV(); unsubC(); };
  }, []);

  // Close the results dropdown on outside click.
  useEffect(() => {
    function onDoc(e: MouseEvent) {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, []);

  const results = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return { vehicles: [] as VehicleRegistration[], claims: [] as Claim[] };
    const v = vehicles
      .filter((x) =>
        `${x.year} ${x.make} ${x.model}`.toLowerCase().includes(q) ||
        x.licenseplate.toLowerCase().includes(q) ||
        x.ownerName.toLowerCase().includes(q))
      .slice(0, 4);
    const c = claims
      .filter((x) =>
        x.id.toLowerCase().includes(q) ||
        x.vehicle.toLowerCase().includes(q) ||
        x.findings.some((f) => f.damage.toLowerCase().includes(q)))
      .slice(0, 4);
    return { vehicles: v, claims: c };
  }, [query, vehicles, claims]);

  const hasResults = results.vehicles.length > 0 || results.claims.length > 0;

  function go(page: string, vehicleId?: string) {
    setOpen(false);
    setQuery("");
    onNavigate(page, vehicleId);
  }

  return (
    <header style={{ height: "64px", background: "#ffffff", borderBottom: "1px solid rgba(0,0,0,0.08)", display: "flex", alignItems: "center", padding: "0 24px", gap: "16px", position: "sticky", top: 0, zIndex: 5 }}>
      <div style={{ flex: 1 }}>
        {breadcrumb && breadcrumb.length > 0 && (
          <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "2px" }}>
            {breadcrumb.map((crumb, i) => (
              <span key={crumb} style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                <span style={{ color: "#888882", fontSize: "11px" }}>{crumb}</span>
                {i < breadcrumb.length - 1 && <ChevronRight size={10} color="#888882" />}
              </span>
            ))}
          </div>
        )}
        <h1 style={{ fontSize: "16px", fontWeight: 600, color: "#0a0a0a", lineHeight: 1 }}>{title}</h1>
      </div>

      {/* Global search */}
      <div ref={wrapRef} style={{ position: "relative", width: "260px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", background: "#f5f5f0", border: "1px solid rgba(0,0,0,0.08)", borderRadius: "8px", padding: "6px 12px" }}>
          <Search size={14} color="#888882" />
          <input
            type="text"
            placeholder="Search vehicles or claims…"
            value={query}
            onChange={(e) => { setQuery(e.target.value); setOpen(true); }}
            onFocus={() => setOpen(true)}
            style={{ background: "transparent", border: "none", outline: "none", fontSize: "13px", color: "#0a0a0a", width: "100%" }}
          />
        </div>

        {open && query.trim() && (
          <div style={{ position: "absolute", top: "calc(100% + 6px)", left: 0, right: 0, background: "#ffffff", border: "1px solid rgba(0,0,0,0.1)", borderRadius: "10px", boxShadow: "0 12px 32px rgba(0,0,0,0.14)", overflow: "hidden", zIndex: 20, maxHeight: "360px", overflowY: "auto" }}>
            {!hasResults && (
              <div style={{ padding: "16px", fontSize: "12px", color: "#888882", textAlign: "center" }}>No matches found.</div>
            )}
            {results.vehicles.length > 0 && (
              <div>
                <div style={{ padding: "8px 12px 4px", fontSize: "9px", fontWeight: 700, color: "#aaaaaa", letterSpacing: "0.08em" }}>VEHICLES</div>
                {results.vehicles.map((v) => (
                  <button key={v.id} onClick={() => go("detail", v.id)} style={resultBtn}>
                    <Car size={15} color="#888882" />
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontSize: "12px", fontWeight: 600, color: "#0a0a0a" }}>{v.year} {v.make} {v.model}</div>
                      <div style={{ fontSize: "10px", color: "#888882" }}>{v.licenseplate}</div>
                    </div>
                  </button>
                ))}
              </div>
            )}
            {results.claims.length > 0 && (
              <div>
                <div style={{ padding: "8px 12px 4px", fontSize: "9px", fontWeight: 700, color: "#aaaaaa", letterSpacing: "0.08em" }}>CLAIMS</div>
                {results.claims.map((c) => (
                  <button key={c.id} onClick={() => go("claims")} style={resultBtn}>
                    <ClipboardCheck size={15} color="#888882" />
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontSize: "12px", fontWeight: 600, color: "#0a0a0a" }}>{c.id}</div>
                      <div style={{ fontSize: "10px", color: "#888882", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{c.vehicle}</div>
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Notification bell */}
      <button
        onClick={() => onNavigate("notifications")}
        title="Notifications"
        style={{ position: "relative", background: "#f5f5f0", border: "1px solid rgba(0,0,0,0.08)", borderRadius: "8px", width: "36px", height: "36px", display: "flex", alignItems: "center", justifyContent: "center", cursor: "pointer", color: "#0a0a0a" }}
      >
        <Bell size={16} />
        {unseen > 0 && (
          <span style={{ position: "absolute", top: "-5px", right: "-5px", minWidth: "16px", height: "16px", padding: "0 4px", background: "#ef4444", color: "#ffffff", fontSize: "9px", fontWeight: 700, borderRadius: "8px", display: "flex", alignItems: "center", justifyContent: "center", border: "2px solid white" }}>
            {unseen}
          </span>
        )}
      </button>

      {/* Date */}
      <div style={{ padding: "6px 12px", background: "#0a0a0a", borderRadius: "8px", color: "var(--accent)", fontSize: "11px", fontWeight: 600, letterSpacing: "0.04em", whiteSpace: "nowrap" }}>
        {new Date().toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric", year: "numeric" })}
      </div>
    </header>
  );
}

const resultBtn: React.CSSProperties = {
  width: "100%", display: "flex", alignItems: "center", gap: "10px", padding: "9px 12px",
  background: "transparent", border: "none", cursor: "pointer", textAlign: "left",
};
