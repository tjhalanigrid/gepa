import { useMemo, useState } from "react";
import { User, Bell, Shield, Monitor, Globe, Save, Check, AlertCircle } from "lucide-react";
import { getCurrentUser, initials, updateProfile, changePassword } from "../../lib/authStore";
import { getSettings, saveSettings, type AppSettings } from "../../lib/settingsStore";

type SettingTab = "profile" | "notifications" | "security" | "appearance" | "regional";

const tabs: { key: SettingTab; label: string; icon: React.ReactNode }[] = [
  { key: "profile", label: "Profile", icon: <User size={14} /> },
  { key: "notifications", label: "Notifications", icon: <Bell size={14} /> },
  { key: "security", label: "Security", icon: <Shield size={14} /> },
  { key: "appearance", label: "Appearance", icon: <Monitor size={14} /> },
  { key: "regional", label: "Regional", icon: <Globe size={14} /> },
];

function SectionHeader({ title, desc }: { title: string; desc: string }) {
  return (
    <div style={{ marginBottom: "20px" }}>
      <h3 style={{ fontSize: "14px", fontWeight: 600, color: "#0a0a0a", marginBottom: "4px" }}>{title}</h3>
      <p style={{ fontSize: "11px", color: "#888882" }}>{desc}</p>
    </div>
  );
}

function FieldGroup({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "200px 1fr", gap: "16px", alignItems: "start", paddingBottom: "16px", borderBottom: "1px solid rgba(0,0,0,0.05)", marginBottom: "16px" }}>
      <div>
        <div style={{ fontSize: "12px", fontWeight: 600, color: "#0a0a0a" }}>{label}</div>
        {hint && <div style={{ fontSize: "11px", color: "#888882", marginTop: "3px", lineHeight: 1.5 }}>{hint}</div>}
      </div>
      <div>{children}</div>
    </div>
  );
}

function TextInput({ placeholder, value, onChange, type = "text" }: { placeholder?: string; value: string; onChange: (v: string) => void; type?: string }) {
  return (
    <input
      type={type}
      placeholder={placeholder}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      style={{ width: "100%", padding: "9px 12px", border: "1px solid rgba(0,0,0,0.1)", borderRadius: "8px", fontSize: "13px", color: "#0a0a0a", background: "#ffffff", outline: "none", boxSizing: "border-box" }}
      onFocus={(e) => (e.currentTarget.style.borderColor = "var(--accent)")}
      onBlur={(e) => (e.currentTarget.style.borderColor = "rgba(0,0,0,0.1)")}
    />
  );
}

function Toggle({ checked, onChange }: { checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      onClick={() => onChange(!checked)}
      style={{ width: "40px", height: "22px", borderRadius: "11px", background: checked ? "var(--accent)" : "#d1d5db", border: "none", cursor: "pointer", position: "relative", transition: "background 0.2s" }}
    >
      <div style={{ width: "16px", height: "16px", borderRadius: "50%", background: "#ffffff", position: "absolute", top: "3px", left: checked ? "21px" : "3px", transition: "left 0.2s", boxShadow: "0 1px 3px rgba(0,0,0,0.2)" }} />
    </button>
  );
}

function Select({ value, options, onChange }: { value: string; options: string[]; onChange: (v: string) => void }) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      style={{ width: "100%", padding: "9px 12px", border: "1px solid rgba(0,0,0,0.1)", borderRadius: "8px", fontSize: "13px", color: "#0a0a0a", background: "#ffffff", outline: "none" }}
      onFocus={(e) => (e.currentTarget.style.borderColor = "var(--accent)")}
      onBlur={(e) => (e.currentTarget.style.borderColor = "rgba(0,0,0,0.1)")}
    >
      {options.map((o) => <option key={o}>{o}</option>)}
    </select>
  );
}

// A short, honest description of the session the user is on right now.
function currentSessionLabel(): string {
  const ua = navigator.userAgent;
  const browser = /Edg/.test(ua) ? "Edge" : /Chrome/.test(ua) ? "Chrome" : /Safari/.test(ua) ? "Safari" : /Firefox/.test(ua) ? "Firefox" : "Browser";
  const os = /Mac/.test(ua) ? "macOS" : /Win/.test(ua) ? "Windows" : /Android/.test(ua) ? "Android" : /iPhone|iPad/.test(ua) ? "iOS" : "Unknown OS";
  return `${browser} · ${os}`;
}

export function SettingsPage() {
  const [activeTab, setActiveTab] = useState<SettingTab>("profile");
  const user = getCurrentUser();
  const phone = user?.phone ?? "";

  // Editable profile (name/phone live in the auth record).
  const [name, setName] = useState(user?.name ?? "");
  const [phoneVal, setPhoneVal] = useState(user?.phone ?? "");

  // Preferences live in the settings store.
  const [settings, setSettings] = useState<AppSettings>(() => getSettings(phone));
  function patch<K extends keyof AppSettings>(key: K, val: AppSettings[K]) {
    setSettings((s) => ({ ...s, [key]: val }));
  }

  // Save feedback for the main preferences/profile save.
  const [savedMsg, setSavedMsg] = useState<{ ok: boolean; text: string } | null>(null);

  async function handleSave() {
    const res = await updateProfile({ name, phone: phoneVal });
    if (!res.ok) {
      setSavedMsg({ ok: false, text: res.error ?? "Could not save profile." });
      return;
    }
    saveSettings(res.user!.phone, settings);
    setSavedMsg({ ok: true, text: "Changes saved." });
    setTimeout(() => setSavedMsg(null), 2500);
  }

  // Password change has its own state + action.
  const [pwCurrent, setPwCurrent] = useState("");
  const [pwNew, setPwNew] = useState("");
  const [pwConfirm, setPwConfirm] = useState("");
  const [pwMsg, setPwMsg] = useState<{ ok: boolean; text: string } | null>(null);

  async function handleChangePassword() {
    if (pwNew !== pwConfirm) {
      setPwMsg({ ok: false, text: "New passwords do not match." });
      return;
    }
    const res = await changePassword(pwCurrent, pwNew);
    if (!res.ok) {
      setPwMsg({ ok: false, text: res.error ?? "Could not update password." });
      return;
    }
    setPwCurrent(""); setPwNew(""); setPwConfirm("");
    setPwMsg({ ok: true, text: "Password updated." });
    setTimeout(() => setPwMsg(null), 2500);
  }

  const avatarInitials = useMemo(() => initials(name || "User"), [name]);

  return (
    <div style={{ padding: "24px", display: "flex", flexDirection: "column", gap: "0" }}>
      <div style={{ display: "grid", gridTemplateColumns: "220px 1fr", gap: "20px", minHeight: "calc(100vh - 130px)" }}>
        {/* Tab sidebar */}
        <div style={{ background: "#ffffff", borderRadius: "12px", border: "1px solid rgba(0,0,0,0.06)", padding: "16px 0", height: "fit-content" }}>
          <div style={{ padding: "0 16px 10px", fontSize: "9px", fontWeight: 700, color: "#888882", letterSpacing: "0.08em" }}>SETTINGS</div>
          {tabs.map((tab) => {
            const active = activeTab === tab.key;
            return (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key)}
                style={{
                  width: "100%", display: "flex", alignItems: "center", gap: "10px", padding: "9px 16px",
                  background: active ? "#fffbeb" : "transparent", border: "none", cursor: "pointer",
                  fontSize: "12px", color: active ? "#0a0a0a" : "#888882", fontWeight: active ? 600 : 400,
                  borderLeft: `2px solid ${active ? "var(--accent)" : "transparent"}`, textAlign: "left",
                }}
                onMouseEnter={(e) => { if (!active) e.currentTarget.style.background = "#fafafa"; }}
                onMouseLeave={(e) => { if (!active) e.currentTarget.style.background = "transparent"; }}
              >
                <span style={{ color: active ? "var(--accent)" : "#888882" }}>{tab.icon}</span>
                {tab.label}
              </button>
            );
          })}
        </div>

        {/* Content */}
        <div style={{ background: "#ffffff", borderRadius: "12px", border: "1px solid rgba(0,0,0,0.06)", padding: "28px" }}>
          {activeTab === "profile" && (
            <div>
              <SectionHeader title="Profile Information" desc="Manage your personal details and account information." />
              {/* Avatar */}
              <div style={{ display: "flex", alignItems: "center", gap: "16px", marginBottom: "24px", padding: "16px", background: "#fafafa", borderRadius: "10px" }}>
                <div style={{ width: "56px", height: "56px", background: "var(--accent)", borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "18px", fontWeight: 800, color: "#0a0a0a", flexShrink: 0 }}>{avatarInitials}</div>
                <div>
                  <div style={{ fontSize: "13px", fontWeight: 600, color: "#0a0a0a" }}>{name || "Your name"}</div>
                  <div style={{ fontSize: "11px", color: "#888882" }}>{phoneVal || "—"}</div>
                </div>
              </div>
              <FieldGroup label="Full Name" hint="Your display name across the platform">
                <TextInput value={name} onChange={setName} placeholder="Your full name" />
              </FieldGroup>
              <FieldGroup label="Phone Number" hint="Used for login and alerts">
                <TextInput value={phoneVal} onChange={setPhoneVal} type="tel" placeholder="Phone number" />
              </FieldGroup>
              <FieldGroup label="Email Address" hint="Used for email notifications">
                <TextInput value={settings.email} onChange={(v) => patch("email", v)} type="email" placeholder="you@example.com" />
              </FieldGroup>
              <FieldGroup label="Organisation" hint="Your registered organisation">
                <TextInput value={settings.organisation} onChange={(v) => patch("organisation", v)} placeholder="Organisation name" />
              </FieldGroup>
            </div>
          )}

          {activeTab === "notifications" && (
            <div>
              <SectionHeader title="Notification Preferences" desc="Control which alerts and updates you receive." />
              {([
                { key: "notifyInspections", label: "Inspection Reports", hint: "Notify when an AI inspection report is generated" },
                { key: "notifyClaims", label: "Claim Status Updates", hint: "Get alerted when a claim changes status" },
                { key: "notifyRenewals", label: "Registration Renewals", hint: "Remind 30 days before a registration expires" },
                { key: "notifyInsurance", label: "Insurance Expiry", hint: "Alert when a policy is about to expire" },
                { key: "notifyAnnouncements", label: "System Announcements", hint: "Product updates and maintenance windows" },
                { key: "notifyDigest", label: "Weekly Summary Digest", hint: "Receive a weekly summary of activity" },
              ] as const).map((item) => (
                <FieldGroup key={item.key} label={item.label} hint={item.hint}>
                  <Toggle checked={settings[item.key]} onChange={(v) => patch(item.key, v)} />
                </FieldGroup>
              ))}
            </div>
          )}

          {activeTab === "security" && (
            <div>
              <SectionHeader title="Security Settings" desc="Manage your password and account security." />
              <FieldGroup label="Current Password" hint="Required to change password">
                <TextInput value={pwCurrent} onChange={setPwCurrent} placeholder="Enter current password" type="password" />
              </FieldGroup>
              <FieldGroup label="New Password" hint="Minimum 4 characters">
                <TextInput value={pwNew} onChange={setPwNew} placeholder="Create new password" type="password" />
              </FieldGroup>
              <FieldGroup label="Confirm Password">
                <TextInput value={pwConfirm} onChange={setPwConfirm} placeholder="Confirm new password" type="password" />
              </FieldGroup>
              <div style={{ display: "flex", alignItems: "center", gap: "12px", marginBottom: "8px" }}>
                <button
                  onClick={handleChangePassword}
                  style={{ padding: "9px 18px", background: "#0a0a0a", color: "var(--accent)", border: "none", borderRadius: "8px", fontSize: "12px", fontWeight: 700, cursor: "pointer" }}
                >
                  Update Password
                </button>
                {pwMsg && (
                  <span style={{ display: "flex", alignItems: "center", gap: "5px", fontSize: "12px", fontWeight: 600, color: pwMsg.ok ? "#10b981" : "#d4183d" }}>
                    {pwMsg.ok ? <Check size={14} /> : <AlertCircle size={14} />} {pwMsg.text}
                  </span>
                )}
              </div>

              <div style={{ marginTop: "16px", borderTop: "1px solid rgba(0,0,0,0.06)", paddingTop: "20px" }}>
                <SectionHeader title="Two-Factor Authentication" desc="Add an extra layer of security to your account." />
                <FieldGroup label="Authenticator App" hint="Use Google Authenticator or Authy">
                  <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                    <Toggle checked={settings.twoFactorApp} onChange={(v) => patch("twoFactorApp", v)} />
                    <span style={{ fontSize: "12px", color: "#888882" }}>{settings.twoFactorApp ? "Enabled" : "Not configured"}</span>
                  </div>
                </FieldGroup>
                <FieldGroup label="SMS Verification" hint="Receive codes via text message">
                  <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                    <Toggle checked={settings.twoFactorSms} onChange={(v) => patch("twoFactorSms", v)} />
                    <span style={{ fontSize: "12px", color: "#888882" }}>{settings.twoFactorSms ? `Active · ${phoneVal || "phone"}` : "Off"}</span>
                  </div>
                </FieldGroup>
              </div>

              <div style={{ marginTop: "16px", borderTop: "1px solid rgba(0,0,0,0.06)", paddingTop: "20px" }}>
                <SectionHeader title="Current Session" desc="The device you are signed in on right now." />
                <div style={{ display: "flex", alignItems: "center", gap: "12px", padding: "10px 0" }}>
                  <div style={{ width: "36px", height: "36px", background: "#fffbeb", borderRadius: "8px", display: "flex", alignItems: "center", justifyContent: "center" }}>
                    <Monitor size={16} color="var(--accent)" />
                  </div>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: "12px", fontWeight: 500, color: "#0a0a0a" }}>
                      {currentSessionLabel()} <span style={{ fontSize: "10px", color: "#10b981", fontWeight: 600 }}>· Current</span>
                    </div>
                    <div style={{ fontSize: "11px", color: "#888882" }}>Signed in as {name || "user"}</div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === "appearance" && (
            <div>
              <SectionHeader title="Appearance" desc="Customise how the application looks." />
              <FieldGroup label="Start Sidebar Collapsed" hint="Remember a collapsed sidebar on load">
                <Toggle checked={settings.sidebarCollapsed} onChange={(v) => patch("sidebarCollapsed", v)} />
              </FieldGroup>
              <FieldGroup label="Compact Mode" hint="Reduce padding for denser layouts">
                <Toggle checked={settings.compactMode} onChange={(v) => patch("compactMode", v)} />
              </FieldGroup>
              <FieldGroup label="Accent Colour" hint="Primary interactive colour">
                <div style={{ display: "flex", gap: "8px" }}>
                  {["#f5c518", "#3b82f6", "#10b981", "#8b5cf6", "#ef4444"].map((col) => (
                    <button
                      key={col}
                      onClick={() => patch("accent", col)}
                      style={{ width: "28px", height: "28px", borderRadius: "50%", background: col, cursor: "pointer", border: settings.accent === col ? "3px solid #0a0a0a" : "2px solid transparent", padding: 0 }}
                    />
                  ))}
                </div>
              </FieldGroup>
            </div>
          )}

          {activeTab === "regional" && (
            <div>
              <SectionHeader title="Regional Settings" desc="Configure language, timezone, and date formats." />
              <FieldGroup label="Language">
                <Select value={settings.language} onChange={(v) => patch("language", v)} options={["English (US)", "English (UK)", "Hindi", "French"]} />
              </FieldGroup>
              <FieldGroup label="Timezone">
                <Select value={settings.timezone} onChange={(v) => patch("timezone", v)} options={["Asia/Kolkata (IST)", "America/New_York (EST)", "Europe/London (GMT)"]} />
              </FieldGroup>
              <FieldGroup label="Date Format">
                <Select value={settings.dateFormat} onChange={(v) => patch("dateFormat", v)} options={["MM/DD/YYYY", "DD/MM/YYYY", "YYYY-MM-DD"]} />
              </FieldGroup>
              <FieldGroup label="Currency">
                <Select value={settings.currency} onChange={(v) => patch("currency", v)} options={["USD ($)", "INR (₹)", "EUR (€)", "GBP (£)"]} />
              </FieldGroup>
            </div>
          )}

          {/* Save button (profile + preferences). Password has its own action above. */}
          <div style={{ marginTop: "24px", paddingTop: "20px", borderTop: "1px solid rgba(0,0,0,0.06)", display: "flex", justifyContent: "flex-end", alignItems: "center", gap: "14px" }}>
            {savedMsg && (
              <span style={{ display: "flex", alignItems: "center", gap: "5px", fontSize: "12px", fontWeight: 600, color: savedMsg.ok ? "#10b981" : "#d4183d" }}>
                {savedMsg.ok ? <Check size={14} /> : <AlertCircle size={14} />} {savedMsg.text}
              </span>
            )}
            <button
              onClick={handleSave}
              style={{ padding: "10px 22px", background: "var(--accent)", color: "#0a0a0a", border: "none", borderRadius: "8px", fontSize: "13px", fontWeight: 700, cursor: "pointer", display: "flex", alignItems: "center", gap: "6px", transition: "background 0.2s" }}
            >
              <Save size={14} /> Save Changes
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
