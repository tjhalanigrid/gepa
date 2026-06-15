import { useEffect, useState } from "react";
import { Bell, CheckCircle, Clock, AlertTriangle, ChevronRight } from "lucide-react";
import {
  getClaims,
  subscribeClaims,
  markNotificationsSeen,
  lastSeen,
  relativeTime,
  type Claim,
} from "../../lib/claimsStore";

interface NotificationsPageProps {
  onNavigate: (page: string) => void;
}

type Kind = "review" | "approved" | "info";

interface Notif {
  id: string;
  kind: Kind;
  title: string;
  desc: string;
  createdAt: string;
  unread: boolean;
}

const KIND_META: Record<Kind, { bg: string; fg: string; icon: React.ReactNode }> = {
  review: { bg: "#fef9c3", fg: "#854d0e", icon: <Clock size={16} /> },
  approved: { bg: "#d1fae5", fg: "#065f46", icon: <CheckCircle size={16} /> },
  info: { bg: "#f0f0eb", fg: "#666660", icon: <AlertTriangle size={16} /> },
};

function toNotif(c: Claim, seen: string): Notif {
  const unread = c.createdAt > seen;
  if (c.status === "pending-review") {
    return { id: c.id, kind: "review", title: `Claim ${c.id} needs review`, desc: `${c.vehicle} — flagged for human review`, createdAt: c.createdAt, unread };
  }
  if (c.status === "no-damage") {
    return { id: c.id, kind: "info", title: `Claim ${c.id}: no damage detected`, desc: c.vehicle, createdAt: c.createdAt, unread };
  }
  const count = c.findings.length;
  return { id: c.id, kind: "approved", title: `Claim ${c.id} auto-approved`, desc: `${c.vehicle} — ${count} damage ${count === 1 ? "region" : "regions"}`, createdAt: c.createdAt, unread };
}

export function NotificationsPage({ onNavigate }: NotificationsPageProps) {
  const [notifs, setNotifs] = useState<Notif[]>([]);

  useEffect(() => {
    // Snapshot the "last seen" BEFORE marking, so this view still highlights
    // what was unread when the user opened it.
    const seen = lastSeen();
    const build = () => setNotifs(getClaims().map((c) => toNotif(c, seen)));
    build();
    const unsub = subscribeClaims(build);
    markNotificationsSeen(); // clears the badge elsewhere
    return unsub;
  }, []);

  const unreadCount = notifs.filter((n) => n.unread).length;

  return (
    <div style={{ padding: "24px", display: "flex", flexDirection: "column", gap: "18px" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "12px" }}>
        <div>
          <h1 style={{ fontSize: "20px", fontWeight: 800, color: "#0a0a0a", display: "flex", alignItems: "center", gap: "8px" }}>
            <Bell size={20} color="var(--accent)" /> Notifications
          </h1>
          <p style={{ fontSize: "12px", color: "#888882", marginTop: "2px" }}>
            {notifs.length} total · {unreadCount} new
          </p>
        </div>
      </div>

      {notifs.length === 0 ? (
        <div style={{ background: "#ffffff", borderRadius: "12px", border: "1px dashed rgba(0,0,0,0.12)", padding: "56px 24px", textAlign: "center" }}>
          <div style={{ width: "56px", height: "56px", background: "#f5f5f0", borderRadius: "16px", display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 16px" }}>
            <Bell size={26} color="#888882" />
          </div>
          <h3 style={{ fontSize: "15px", fontWeight: 700, color: "#0a0a0a", marginBottom: "6px" }}>You're all caught up</h3>
          <p style={{ fontSize: "12px", color: "#888882" }}>Notifications appear here when you run inspections.</p>
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
          {notifs.map((n) => {
            const m = KIND_META[n.kind];
            return (
              <button
                key={n.id}
                onClick={() => onNavigate("claims")}
                style={{
                  display: "flex", alignItems: "center", gap: "14px", textAlign: "left", width: "100%",
                  background: n.unread ? "#fffdf5" : "#ffffff",
                  border: `1px solid ${n.unread ? "#f5e08a" : "rgba(0,0,0,0.06)"}`,
                  borderRadius: "12px", padding: "14px 16px", cursor: "pointer",
                }}
                onMouseEnter={(e) => (e.currentTarget.style.borderColor = "var(--accent)")}
                onMouseLeave={(e) => (e.currentTarget.style.borderColor = n.unread ? "#f5e08a" : "rgba(0,0,0,0.06)")}
              >
                <div style={{ width: "38px", height: "38px", borderRadius: "10px", background: m.bg, color: m.fg, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                  {m.icon}
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: "13px", fontWeight: 600, color: "#0a0a0a", display: "flex", alignItems: "center", gap: "8px" }}>
                    {n.title}
                    {n.unread && <span style={{ width: "7px", height: "7px", borderRadius: "50%", background: "var(--accent)" }} />}
                  </div>
                  <div style={{ fontSize: "11px", color: "#888882", marginTop: "2px" }}>{n.desc}</div>
                </div>
                <span style={{ fontSize: "11px", color: "#aaaaaa", whiteSpace: "nowrap" }}>{relativeTime(n.createdAt)}</span>
                <ChevronRight size={15} color="#cccccc" />
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
