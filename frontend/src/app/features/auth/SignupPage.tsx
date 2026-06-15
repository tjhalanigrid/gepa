import { useState } from "react";
import { Eye, EyeOff, AlertCircle } from "lucide-react";
import { AuthLayout } from "./AuthLayout";
import { signup, type AuthUser } from "../../lib/authStore";

interface SignupPageProps {
  onNavigate: (page: string) => void;
  onAuth: (user: AuthUser) => void;
}

export function SignupPage({ onNavigate, onAuth }: SignupPageProps) {
  const [form, setForm] = useState({ fullName: "", phone: "", password: "", confirmPassword: "" });
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const [agreed, setAgreed] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  function setField(key: keyof typeof form, value: string) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (form.password !== form.confirmPassword) {
      setError("Passwords do not match.");
      return;
    }
    if (!agreed) {
      setError("Please accept the terms to continue.");
      return;
    }
    setError(null);
    setBusy(true);
    const res = await signup(form.fullName, form.phone, form.password);
    setBusy(false);
    if (res.ok && res.user) {
      onAuth(res.user);
    } else {
      setError(res.error ?? "Sign up failed.");
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

  const socialButtons = [
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
  ];

  return (
    <AuthLayout
      onNavigate={onNavigate}
      navAction={{ label: "Home", page: "login", btnLabel: "Log In" }}
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
        <div style={{ marginBottom: "28px" }}>
          <h1 style={{ fontSize: "26px", fontWeight: 800, color: "#ffffff", marginBottom: "6px", letterSpacing: "-0.02em" }}>
            Create your account
          </h1>
          <p style={{ fontSize: "13px", color: "#666660" }}>
            Get started with automated damage intelligence
          </p>
        </div>

        <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          {error && (
            <div style={{ display: "flex", alignItems: "center", gap: "8px", padding: "11px 14px", background: "rgba(239,68,68,0.12)", border: "1px solid rgba(239,68,68,0.4)", borderRadius: "10px" }}>
              <AlertCircle size={15} color="#f87171" />
              <span style={{ fontSize: "12px", color: "#fca5a5" }}>{error}</span>
            </div>
          )}
          {/* Full name */}
          <div>
            <label style={labelStyle}>FULL NAME</label>
            <input
              type="text"
              placeholder="Enter your full name"
              value={form.fullName}
              onChange={(e) => setField("fullName", e.target.value)}
              style={inputStyle}
              onFocus={(e) => (e.currentTarget.style.borderColor = "var(--accent)")}
              onBlur={(e) => (e.currentTarget.style.borderColor = "#2a2a2a")}
            />
          </div>

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
                }}
              >
                +91
              </div>
              <input
                type="tel"
                placeholder="98765 43210"
                value={form.phone}
                onChange={(e) => setField("phone", e.target.value)}
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
                placeholder="Create a password"
                value={form.password}
                onChange={(e) => setField("password", e.target.value)}
                style={{ ...inputStyle, paddingRight: "44px" }}
                onFocus={(e) => (e.currentTarget.style.borderColor = "var(--accent)")}
                onBlur={(e) => (e.currentTarget.style.borderColor = "#2a2a2a")}
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                style={{
                  position: "absolute", right: "14px", top: "50%", transform: "translateY(-50%)",
                  background: "transparent", border: "none", cursor: "pointer", color: "#666660",
                  display: "flex", alignItems: "center",
                }}
              >
                {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </div>

          {/* Confirm password */}
          <div>
            <label style={labelStyle}>CONFIRM PASSWORD</label>
            <div style={{ position: "relative" }}>
              <input
                type={showConfirm ? "text" : "password"}
                placeholder="Confirm your password"
                value={form.confirmPassword}
                onChange={(e) => setField("confirmPassword", e.target.value)}
                style={{ ...inputStyle, paddingRight: "44px" }}
                onFocus={(e) => (e.currentTarget.style.borderColor = "var(--accent)")}
                onBlur={(e) => (e.currentTarget.style.borderColor = "#2a2a2a")}
              />
              <button
                type="button"
                onClick={() => setShowConfirm(!showConfirm)}
                style={{
                  position: "absolute", right: "14px", top: "50%", transform: "translateY(-50%)",
                  background: "transparent", border: "none", cursor: "pointer", color: "#666660",
                  display: "flex", alignItems: "center",
                }}
              >
                {showConfirm ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </div>

          {/* Terms */}
          <label style={{ display: "flex", alignItems: "center", gap: "10px", cursor: "pointer" }}>
            <input
              type="checkbox"
              checked={agreed}
              onChange={(e) => setAgreed(e.target.checked)}
              style={{ width: "16px", height: "16px", accentColor: "var(--accent)", cursor: "pointer", flexShrink: 0 }}
            />
            <span style={{ fontSize: "12px", color: "#888882" }}>
              I agree to the{" "}
              <button
                type="button"
                style={{ background: "transparent", border: "none", color: "var(--accent)", fontWeight: 600, cursor: "pointer", fontSize: "12px", padding: 0 }}
              >
                Terms & Conditions
              </button>
            </span>
          </label>

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
              marginTop: "2px",
              letterSpacing: "0.01em",
              opacity: busy ? 0.7 : 1,
            }}
            onMouseEnter={(e) => (e.currentTarget.style.background = "#fde047")}
            onMouseLeave={(e) => (e.currentTarget.style.background = "var(--accent)")}
          >
            {busy ? "Creating account…" : "Sign Up"}
          </button>

          {/* Divider */}
          <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
            <div style={{ flex: 1, height: "1px", background: "#1e1e1e" }} />
            <span style={{ fontSize: "10px", color: "#444440", letterSpacing: "0.08em", fontWeight: 600 }}>OR CONTINUE WITH</span>
            <div style={{ flex: 1, height: "1px", background: "#1e1e1e" }} />
          </div>

          {/* Social */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px" }}>
            {socialButtons.map((social) => (
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

          {/* Login link */}
          <p style={{ textAlign: "center", fontSize: "13px", color: "#666660" }}>
            Already have an account?{" "}
            <button
              type="button"
              onClick={() => onNavigate("login")}
              style={{ background: "transparent", border: "none", color: "var(--accent)", fontWeight: 700, cursor: "pointer", fontSize: "13px" }}
            >
              Log In
            </button>
          </p>
        </form>
      </div>
    </AuthLayout>
  );
}
