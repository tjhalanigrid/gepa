import { Car } from "lucide-react";

interface AuthLayoutProps {
  children: React.ReactNode;
  onNavigate: (page: string) => void;
  navAction: { label: string; page: string; btnLabel: string };
}

export function AuthLayout({ children, onNavigate, navAction }: AuthLayoutProps) {
  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#0a0a0a",
        fontFamily: "'Inter', system-ui, -apple-system, sans-serif",
        display: "flex",
        flexDirection: "column",
      }}
    >
      {/* Nav */}
      <nav
        style={{
          background: "rgba(10,10,10,0.88)",
          backdropFilter: "blur(16px)",
          borderBottom: "1px solid #1e1e1e",
          padding: "0 48px",
          height: "64px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          position: "sticky",
          top: 0,
          zIndex: 50,
        }}
      >
        <button
          onClick={() => onNavigate("landing")}
          style={{
            background: "transparent",
            border: "none",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            gap: "10px",
          }}
        >
          <div
            style={{
              width: "36px",
              height: "36px",
              background: "var(--accent)",
              borderRadius: "9px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <Car size={18} color="#0a0a0a" />
          </div>
          <span style={{ fontSize: "15px", fontWeight: 700, color: "#ffffff" }}>
            AutoReg <span style={{ color: "var(--accent)" }}>AI</span>
          </span>
        </button>

        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <button
            onClick={() => onNavigate("landing")}
            style={{ background: "transparent", border: "none", fontSize: "13px", color: "#888882", cursor: "pointer", fontWeight: 500, padding: "8px 12px" }}
            onMouseEnter={(e) => (e.currentTarget.style.color = "#ffffff")}
            onMouseLeave={(e) => (e.currentTarget.style.color = "#888882")}
          >
            {navAction.label}
          </button>
          <button
            onClick={() => onNavigate(navAction.page)}
            style={{
              background: "var(--accent)",
              color: "#0a0a0a",
              border: "none",
              borderRadius: "8px",
              padding: "9px 20px",
              fontSize: "13px",
              fontWeight: 700,
              cursor: "pointer",
            }}
            onMouseEnter={(e) => (e.currentTarget.style.background = "#fde047")}
            onMouseLeave={(e) => (e.currentTarget.style.background = "var(--accent)")}
          >
            {navAction.btnLabel}
          </button>
        </div>
      </nav>

      {/* Decorative background grid */}
      <div
        style={{
          position: "fixed",
          inset: 0,
          backgroundImage: `
            linear-gradient(rgba(245,197,24,0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(245,197,24,0.03) 1px, transparent 1px)
          `,
          backgroundSize: "60px 60px",
          pointerEvents: "none",
          zIndex: 0,
        }}
      />
      {/* Radial glow */}
      <div
        style={{
          position: "fixed",
          top: "30%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          width: "600px",
          height: "600px",
          background: "radial-gradient(ellipse, rgba(245,197,24,0.06) 0%, transparent 70%)",
          pointerEvents: "none",
          zIndex: 0,
        }}
      />

      {/* Content */}
      <div
        style={{
          flex: 1,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          padding: "48px 24px",
          position: "relative",
          zIndex: 1,
        }}
      >
        {children}
      </div>
    </div>
  );
}
