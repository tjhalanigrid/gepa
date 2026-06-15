import { useState } from "react";
import { Eye, EyeOff, AlertCircle } from "lucide-react";
import { AuthLayout } from "./AuthLayout";
import { login, type AuthUser } from "../../lib/authStore";

interface LoginPageProps {
  onNavigate: (page: string) => void;
  onAuth: (user: AuthUser) => void;
}

export function LoginPage({ onNavigate, onAuth }: LoginPageProps) {
  const [phone, setPhone] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    const res = await login(phone, password);
    setBusy(false);
    if (res.ok && res.user) {
      onAuth(res.user);
    } else {
      setError(res.error ?? "Login failed.");
    }
  }

  const inputStyle: React.CSSProperties = {
    width: "100%",
    padding: "13px 16px",
    border: "1px solid #2a2a2a",
    borderRadius: "10px",
    fontSize: "14px",
    color: "#ffffff",
    background: "#111111",
    outline: "none",
    boxSizing: "border-box",
    transition: "border-color 0.15s",
  };

  const labelStyle: React.CSSProperties = {
    fontSize: "11px",
    fontWeight: 700,
    color: "#888882",
    letterSpacing: "0.07em",
    marginBottom: "8px",
    display: "block",
  };

  return (
    <AuthLayout
      onNavigate={onNavigate}
      navAction={{ label: "Home", page: "signup", btnLabel: "Sign Up" }}
    >
      <div
        style={{
          background: "#111111",
          borderRadius: "20px",
          padding: "44px 48px",
          width: "100%",
          maxWidth: "460px",
          border: "1px solid #1e1e1e",
          boxShadow: "0 24px 64px rgba(0,0,0,0.6)",
        }}
      >
        {/* Header */}
        <div style={{ marginBottom: "32px" }}>
          <h1 style={{ fontSize: "26px", fontWeight: 800, color: "#ffffff", marginBottom: "6px", letterSpacing: "-0.02em" }}>
            Welcome back!
          </h1>
          <p style={{ fontSize: "13px", color: "#666660" }}>
            Login to your account
          </p>
        </div>

        <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: "18px" }}>
          {error && (
            <div style={{ display: "flex", alignItems: "center", gap: "8px", padding: "11px 14px", background: "rgba(239,68,68,0.12)", border: "1px solid rgba(239,68,68,0.4)", borderRadius: "10px" }}>
              <AlertCircle size={15} color="#f87171" />
              <span style={{ fontSize: "12px", color: "#fca5a5" }}>{error}</span>
            </div>
          )}
          {/* Phone */}
          <div>
            <label style={labelStyle}>PHONE NUMBER</label>
            <div style={{ display: "flex", gap: "8px" }}>
              <div
                style={{
                  padding: "13px 16px",
                  border: "1px solid #2a2a2a",
                  borderRadius: "10px",
                  fontSize: "14px",
                  color: "var(--accent)",
                  background: "#1a1a1a",
                  fontWeight: 700,
                  whiteSpace: "nowrap",
                  flexShrink: 0,
                  letterSpacing: "0.02em",
                }}
              >
                +91
              </div>
              <input
                type="tel"
                placeholder="98765 43210"
                value={phone}
                onChange={(e) => setPhone(e.target.value)}
                style={{ ...inputStyle, flex: 1 }}
                onFocus={(e) => (e.currentTarget.style.borderColor = "var(--accent)")}
                onBlur={(e) => (e.currentTarget.style.borderColor = "#2a2a2a")}
              />
            </div>
          </div>

          {/* Password */}
          <div>
            <label style={labelStyle}>PASSWORD</label>
            <div style={{ position: "relative" }}>
              <input
                type={showPassword ? "text" : "password"}
                placeholder="Enter password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                style={{ ...inputStyle, paddingRight: "44px" }}
                onFocus={(e) => (e.currentTarget.style.borderColor = "var(--accent)")}
                onBlur={(e) => (e.currentTarget.style.borderColor = "#2a2a2a")}
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                style={{
                  position: "absolute",
                  right: "14px",
                  top: "50%",
                  transform: "translateY(-50%)",
                  background: "transparent",
                  border: "none",
                  cursor: "pointer",
                  color: "#666660",
                  display: "flex",
                  alignItems: "center",
                }}
              >
                {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </div>

          {/* Forgot */}
          <div style={{ textAlign: "right", marginTop: "-8px" }}>
            <button
              type="button"
              onClick={() => onNavigate("forgot-password")}
              style={{
                background: "transparent",
                border: "none",
                fontSize: "12px",
                color: "var(--accent)",
                cursor: "pointer",
                fontWeight: 600,
              }}
            >
              Forgot Password?
            </button>
          </div>

          {/* Submit */}
          <button
            type="submit"
            disabled={busy}
            style={{
              width: "100%",
              padding: "14px",
              background: "var(--accent)",
              color: "#0a0a0a",
              border: "none",
              borderRadius: "10px",
              fontSize: "15px",
              fontWeight: 800,
              cursor: busy ? "wait" : "pointer",
              letterSpacing: "0.01em",
              opacity: busy ? 0.7 : 1,
            }}
            onMouseEnter={(e) => (e.currentTarget.style.background = "#fde047")}
            onMouseLeave={(e) => (e.currentTarget.style.background = "var(--accent)")}
          >
            {busy ? "Signing in…" : "Login"}
          </button>

          {/* Divider */}
          <div style={{ display: "flex", alignItems: "center", gap: "12px", margin: "2px 0" }}>
            <div style={{ flex: 1, height: "1px", background: "#1e1e1e" }} />
            <span style={{ fontSize: "10px", color: "#444440", letterSpacing: "0.08em", fontWeight: 600 }}>
              OR CONTINUE WITH
            </span>
            <div style={{ flex: 1, height: "1px", background: "#1e1e1e" }} />
          </div>

          {/* Social */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px" }}>
            {[
              {
                label: "Google",
                icon: (
                  <svg width="16" height="16" viewBox="0 0 24 24">
                    <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                    <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                    <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                    <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                  </svg>
                ),
              },
              {
                label: "Apple",
                icon: (
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="#ffffff">
                    <path d="M18.71 19.5c-.83 1.24-1.71 2.45-3.05 2.47-1.34.03-1.77-.79-3.29-.79-1.53 0-2 .77-3.27.82-1.31.05-2.3-1.32-3.14-2.53C4.25 17 2.94 12.45 4.7 9.39c.87-1.52 2.43-2.48 4.12-2.51 1.28-.02 2.5.87 3.29.87.78 0 2.26-1.07 3.8-.91.65.03 2.47.26 3.64 1.98-.09.06-2.17 1.28-2.15 3.81.03 3.02 2.65 4.03 2.68 4.04-.03.07-.42 1.44-1.38 2.83M13 3.5c.73-.83 1.94-1.46 2.94-1.5.13 1.17-.34 2.35-1.04 3.19-.69.85-1.83 1.51-2.95 1.42-.15-1.15.41-2.35 1.05-3.11z"/>
                  </svg>
                ),
              },
            ].map((social) => (
              <button
                key={social.label}
                type="button"
                style={{
                  padding: "11px",
                  border: "1px solid #2a2a2a",
                  borderRadius: "10px",
                  background: "#1a1a1a",
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: "8px",
                  fontSize: "13px",
                  color: "#cccccc",
                  fontWeight: 500,
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = "var(--accent)";
                  e.currentTarget.style.color = "#ffffff";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = "#2a2a2a";
                  e.currentTarget.style.color = "#cccccc";
                }}
              >
                {social.icon}
                {social.label}
              </button>
            ))}
          </div>

          {/* Sign up link */}
          <p style={{ textAlign: "center", fontSize: "13px", color: "#666660", marginTop: "4px" }}>
            Don't have an account?{" "}
            <button
              type="button"
              onClick={() => onNavigate("signup")}
              style={{ background: "transparent", border: "none", color: "var(--accent)", fontWeight: 700, cursor: "pointer", fontSize: "13px" }}
            >
              Sign Up
            </button>
          </p>
        </form>
      </div>
    </AuthLayout>
  );
}
