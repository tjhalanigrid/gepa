import { useState } from "react";
import { ArrowLeft, CheckCircle } from "lucide-react";
import { AuthLayout } from "./AuthLayout";

interface ForgotPasswordPageProps {
  onNavigate: (page: string) => void;
}

export function ForgotPasswordPage({ onNavigate }: ForgotPasswordPageProps) {
  const [phone, setPhone] = useState("");
  const [sent, setSent] = useState(false);

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSent(true);
  }

  return (
    <AuthLayout
      onNavigate={onNavigate}
      navAction={{ label: "Home", page: "signup", btnLabel: "Sign Up" }}
    >
      <div
        style={{
          background: "#111111",
          borderRadius: "20px",
          padding: "48px",
          width: "100%",
          maxWidth: "440px",
          border: "1px solid #1e1e1e",
          boxShadow: "0 24px 64px rgba(0,0,0,0.6)",
        }}
      >
        {sent ? (
          <div style={{ textAlign: "center" }}>
            <div
              style={{
                width: "64px",
                height: "64px",
                background: "var(--accent)",
                borderRadius: "50%",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                margin: "0 auto 24px",
              }}
            >
              <CheckCircle size={28} color="#0a0a0a" />
            </div>
            <h2 style={{ fontSize: "22px", fontWeight: 800, color: "#ffffff", marginBottom: "10px", letterSpacing: "-0.02em" }}>
              Reset link sent!
            </h2>
            <p style={{ fontSize: "13px", color: "#666660", marginBottom: "32px", lineHeight: 1.7 }}>
              We've sent a password reset link to the number ending in{" "}
              <strong style={{ color: "var(--accent)" }}>****{phone.slice(-4) || "0000"}</strong>.
              Check your messages and follow the instructions.
            </p>
            <button
              onClick={() => onNavigate("login")}
              style={{
                width: "100%",
                padding: "13px",
                background: "var(--accent)",
                color: "#0a0a0a",
                border: "none",
                borderRadius: "10px",
                fontSize: "14px",
                fontWeight: 800,
                cursor: "pointer",
              }}
              onMouseEnter={(e) => (e.currentTarget.style.background = "#fde047")}
              onMouseLeave={(e) => (e.currentTarget.style.background = "var(--accent)")}
            >
              Back to Login
            </button>
          </div>
        ) : (
          <>
            <h1 style={{ fontSize: "26px", fontWeight: 800, color: "#ffffff", marginBottom: "8px", letterSpacing: "-0.02em" }}>
              Reset your password
            </h1>
            <p style={{ fontSize: "13px", color: "#666660", marginBottom: "32px", lineHeight: 1.7 }}>
              Enter your phone number and we'll send you a reset link.
            </p>

            <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
              <div>
                <label
                  style={{
                    fontSize: "11px",
                    fontWeight: 700,
                    color: "#888882",
                    letterSpacing: "0.07em",
                    marginBottom: "8px",
                    display: "block",
                  }}
                >
                  PHONE NUMBER
                </label>
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
                    value={phone}
                    onChange={(e) => setPhone(e.target.value)}
                    required
                    style={{
                      flex: 1,
                      padding: "13px 16px",
                      border: "1px solid #2a2a2a",
                      borderRadius: "10px",
                      fontSize: "14px",
                      color: "#ffffff",
                      background: "#111111",
                      outline: "none",
                      boxSizing: "border-box" as const,
                      transition: "border-color 0.15s",
                    }}
                    onFocus={(e) => (e.currentTarget.style.borderColor = "var(--accent)")}
                    onBlur={(e) => (e.currentTarget.style.borderColor = "#2a2a2a")}
                  />
                </div>
              </div>

              <button
                type="submit"
                style={{
                  width: "100%",
                  padding: "14px",
                  background: "var(--accent)",
                  color: "#0a0a0a",
                  border: "none",
                  borderRadius: "10px",
                  fontSize: "15px",
                  fontWeight: 800,
                  cursor: "pointer",
                  letterSpacing: "0.01em",
                }}
                onMouseEnter={(e) => (e.currentTarget.style.background = "#fde047")}
                onMouseLeave={(e) => (e.currentTarget.style.background = "var(--accent)")}
              >
                Send Reset Link
              </button>

              <div style={{ textAlign: "center" }}>
                <button
                  type="button"
                  onClick={() => onNavigate("login")}
                  style={{
                    background: "transparent",
                    border: "none",
                    color: "var(--accent)",
                    fontWeight: 600,
                    cursor: "pointer",
                    fontSize: "13px",
                    display: "inline-flex",
                    alignItems: "center",
                    gap: "5px",
                  }}
                >
                  <ArrowLeft size={13} />
                  Back to login
                </button>
              </div>
            </form>
          </>
        )}
      </div>
    </AuthLayout>
  );
}
