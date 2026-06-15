import { useEffect, useState } from "react";
import { Search, Filter, Plus, Eye, Edit2, Trash2, ChevronUp, ChevronDown } from "lucide-react";
import { getVehicles, subscribeVehicles } from "../../lib/vehiclesStore";
import type { VehicleRegistration, VehicleStatus } from "../../types/vehicle";

const statusColors: Record<VehicleStatus, { bg: string; text: string; dot: string }> = {
  active: { bg: "#d1fae5", text: "#065f46", dot: "#10b981" },
  inactive: { bg: "#f3f4f6", text: "#6b7280", dot: "#9ca3af" },
  maintenance: { bg: "#fff7ed", text: "#c2410c", dot: "#f97316" },
  pending: { bg: "#fef9c3", text: "#854d0e", dot: "var(--accent)" },
};

interface VehicleListPageProps {
  onNavigate: (page: string, vehicleId?: string) => void;
}

type SortKey = keyof Pick<VehicleRegistration, "make" | "year" | "ownerName" | "status" | "registrationDate">;

export function VehicleListPage({ onNavigate }: VehicleListPageProps) {
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState<VehicleStatus | "all">("all");
  const [sortKey, setSortKey] = useState<SortKey>("registrationDate");
  const [sortAsc, setSortAsc] = useState(false);
  const [vehicles, setVehicles] = useState<VehicleRegistration[]>(getVehicles());

  useEffect(() => {
    const load = () => setVehicles(getVehicles());
    load();
    return subscribeVehicles(load);
  }, []);

  const filtered = vehicles
    .filter((v) => {
      const q = search.toLowerCase();
      const matchesSearch =
        !search ||
        v.make.toLowerCase().includes(q) ||
        v.model.toLowerCase().includes(q) ||
        v.ownerName.toLowerCase().includes(q) ||
        v.licenseplate.toLowerCase().includes(q) ||
        v.registrationNumber.toLowerCase().includes(q);
      const matchesStatus = statusFilter === "all" || v.status === statusFilter;
      return matchesSearch && matchesStatus;
    })
    .sort((a, b) => {
      const av = String(a[sortKey]);
      const bv = String(b[sortKey]);
      return sortAsc ? av.localeCompare(bv) : bv.localeCompare(av);
    });

  function toggleSort(key: SortKey) {
    if (sortKey === key) setSortAsc(!sortAsc);
    else {
      setSortKey(key);
      setSortAsc(true);
    }
  }

  function SortIcon({ col }: { col: SortKey }) {
    if (sortKey !== col) return <ChevronUp size={12} color="#cccccc" />;
    return sortAsc ? <ChevronUp size={12} color="var(--accent)" /> : <ChevronDown size={12} color="var(--accent)" />;
  }

  return (
    <div style={{ padding: "24px", display: "flex", flexDirection: "column", gap: "20px" }}>
      {/* Toolbar */}
      <div style={{ display: "flex", alignItems: "center", gap: "12px", flexWrap: "wrap" }}>
        {/* Search */}
        <div
          style={{
            flex: 1,
            minWidth: "200px",
            display: "flex",
            alignItems: "center",
            gap: "8px",
            background: "#ffffff",
            border: "1px solid rgba(0,0,0,0.08)",
            borderRadius: "8px",
            padding: "8px 14px",
          }}
        >
          <Search size={14} color="#888882" />
          <input
            type="text"
            placeholder="Search by name, plate, make..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{
              background: "transparent",
              border: "none",
              outline: "none",
              fontSize: "13px",
              color: "#0a0a0a",
              width: "100%",
            }}
          />
        </div>

        {/* Status filter */}
        <div style={{ display: "flex", gap: "6px" }}>
          {(["all", "active", "pending", "maintenance", "inactive"] as const).map((s) => (
            <button
              key={s}
              onClick={() => setStatusFilter(s)}
              style={{
                padding: "7px 14px",
                borderRadius: "8px",
                border: "1px solid",
                borderColor: statusFilter === s ? "#0a0a0a" : "rgba(0,0,0,0.08)",
                background: statusFilter === s ? "#0a0a0a" : "#ffffff",
                color: statusFilter === s ? "var(--accent)" : "#888882",
                fontSize: "11px",
                fontWeight: 500,
                cursor: "pointer",
                textTransform: "capitalize",
              }}
            >
              {s === "all" ? "All Vehicles" : s}
            </button>
          ))}
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "6px", marginLeft: "auto" }}>
          <button
            style={{
              padding: "8px 12px",
              borderRadius: "8px",
              border: "1px solid rgba(0,0,0,0.08)",
              background: "#ffffff",
              color: "#444440",
              fontSize: "12px",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: "6px",
            }}
          >
            <Filter size={13} />
            Filters
          </button>
          <button
            onClick={() => onNavigate("register")}
            style={{
              padding: "8px 16px",
              borderRadius: "8px",
              border: "none",
              background: "var(--accent)",
              color: "#0a0a0a",
              fontSize: "12px",
              fontWeight: 600,
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: "6px",
            }}
          >
            <Plus size={14} />
            Add Vehicle
          </button>
        </div>
      </div>

      {/* Count */}
      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
        <span style={{ fontSize: "12px", color: "#888882" }}>
          Showing <strong style={{ color: "#0a0a0a" }}>{filtered.length}</strong> of {vehicles.length} vehicles
        </span>
      </div>

      {/* Table */}
      <div
        style={{
          background: "#ffffff",
          borderRadius: "12px",
          border: "1px solid rgba(0,0,0,0.06)",
          overflow: "hidden",
        }}
      >
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ background: "#fafafa", borderBottom: "1px solid rgba(0,0,0,0.06)" }}>
              {[
                { label: "Vehicle", key: "make" as SortKey },
                { label: "Owner", key: "ownerName" as SortKey },
                { label: "License Plate", key: null },
                { label: "Year", key: "year" as SortKey },
                { label: "Status", key: "status" as SortKey },
                { label: "Registered", key: "registrationDate" as SortKey },
                { label: "Mileage", key: null },
                { label: "Actions", key: null },
              ].map(({ label, key }) => (
                <th
                  key={label}
                  onClick={key ? () => toggleSort(key) : undefined}
                  style={{
                    padding: "10px 16px",
                    textAlign: "left",
                    fontSize: "10px",
                    fontWeight: 700,
                    color: "#888882",
                    letterSpacing: "0.06em",
                    cursor: key ? "pointer" : "default",
                    userSelect: "none",
                    whiteSpace: "nowrap",
                  }}
                >
                  <span style={{ display: "inline-flex", alignItems: "center", gap: "4px" }}>
                    {label}
                    {key && <SortIcon col={key} />}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 ? (
              <tr>
                <td colSpan={8} style={{ padding: "40px", textAlign: "center", color: "#888882", fontSize: "13px" }}>
                  No vehicles match your search.
                </td>
              </tr>
            ) : (
              filtered.map((v, i) => {
                const sc = statusColors[v.status];
                return (
                  <tr
                    key={v.id}
                    style={{
                      borderBottom: i < filtered.length - 1 ? "1px solid rgba(0,0,0,0.04)" : "none",
                      cursor: "pointer",
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.background = "#fafafa")}
                    onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                  >
                    <td style={{ padding: "14px 16px" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                        <div
                          style={{
                            width: "36px",
                            height: "36px",
                            background: "#f5f5f0",
                            borderRadius: "8px",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            fontSize: "10px",
                            fontWeight: 700,
                            color: "#888882",
                          }}
                        >
                          {v.make.slice(0, 2).toUpperCase()}
                        </div>
                        <div>
                          <div style={{ fontSize: "13px", fontWeight: 500, color: "#0a0a0a" }}>
                            {v.make} {v.model}
                          </div>
                          <div style={{ fontSize: "10px", color: "#888882" }}>{v.registrationNumber}</div>
                        </div>
                      </div>
                    </td>
                    <td style={{ padding: "14px 16px" }}>
                      <div style={{ fontSize: "12px", color: "#0a0a0a" }}>{v.ownerName}</div>
                      <div style={{ fontSize: "10px", color: "#888882" }}>{v.ownerPhone}</div>
                    </td>
                    <td style={{ padding: "14px 16px" }}>
                      <code style={{ fontSize: "12px", background: "#f5f5f0", padding: "3px 8px", borderRadius: "4px", color: "#0a0a0a", letterSpacing: "0.06em" }}>
                        {v.licenseplate}
                      </code>
                    </td>
                    <td style={{ padding: "14px 16px", fontSize: "12px", color: "#444440" }}>{v.year}</td>
                    <td style={{ padding: "14px 16px" }}>
                      <span
                        style={{
                          background: sc.bg,
                          color: sc.text,
                          fontSize: "10px",
                          fontWeight: 600,
                          padding: "3px 8px",
                          borderRadius: "10px",
                          display: "inline-flex",
                          alignItems: "center",
                          gap: "4px",
                          textTransform: "capitalize",
                        }}
                      >
                        <span style={{ width: "5px", height: "5px", borderRadius: "50%", background: sc.dot }} />
                        {v.status}
                      </span>
                    </td>
                    <td style={{ padding: "14px 16px", fontSize: "12px", color: "#888882" }}>
                      {new Date(v.registrationDate).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}
                    </td>
                    <td style={{ padding: "14px 16px", fontSize: "12px", color: "#444440" }}>
                      {v.mileage.toLocaleString()} mi
                    </td>
                    <td style={{ padding: "14px 16px" }}>
                      <div style={{ display: "flex", gap: "6px" }}>
                        <button
                          onClick={(e) => { e.stopPropagation(); onNavigate("detail", v.id); }}
                          style={{ padding: "5px", background: "#f5f5f0", border: "none", borderRadius: "6px", cursor: "pointer", color: "#444440" }}
                          title="View"
                        >
                          <Eye size={13} />
                        </button>
                        <button
                          style={{ padding: "5px", background: "#f5f5f0", border: "none", borderRadius: "6px", cursor: "pointer", color: "#444440" }}
                          title="Edit"
                        >
                          <Edit2 size={13} />
                        </button>
                        <button
                          style={{ padding: "5px", background: "#fff0f0", border: "none", borderRadius: "6px", cursor: "pointer", color: "#d4183d" }}
                          title="Delete"
                        >
                          <Trash2 size={13} />
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
