import { useEffect, useRef, useState } from "react";
import { Car, ArrowRight, Shield, Zap, BarChart3, CheckCircle, Star } from "lucide-react";

interface LandingPageProps {
  onNavigate: (page: string) => void;
}

// Words the hero headline cycles through.
const ROTATING_WORDS = ["Vehicle Damage", "Every Dent", "Each Scratch", "Hidden Cracks", "Glass Shatter"];

const features = [
  {
    icon: <Zap size={20} />,
    title: "AI Damage Detection",
    desc: "Neural surface topography maps every dent, scratch, and structural deformation in under 30 seconds.",
  },
  {
    icon: <Shield size={20} />,
    title: "Fraud Prevention",
    desc: "Tamper-proof audit trail with cryptographic image fingerprinting for every claim filed.",
  },
  {
    icon: <BarChart3 size={20} />,
    title: "Claims Analytics",
    desc: "Real-time dashboards surface bottlenecks and cost drivers across your entire portfolio.",
  },
];

const stats = [
  { value: "80.0%", label: "Detection Accuracy" },
  { value: "< 6M", label: "Per Vehicle Assessment" },
  { value: "12×", label: "Faster Than Manual" },
  { value: "$2.1M", label: "Fraud Prevented / Year" },
];

const testimonials = [
  {
    name: "Priya Sharma",
    role: "Head of Claims, Axis Insurance",
    text: "DriveInspect AI cut our average claim cycle from 11 days to under 18 hours. The AI accuracy is genuinely remarkable.",
    stars: 5,
  },
  {
    name: "Marcus Chen",
    role: "Fleet Operations Director, LogiCorp",
    text: "We manage 4,200 vehicles. This platform gives us a real-time view of every vehicle's condition — nothing else comes close.",
    stars: 5,
  },
  {
    name: "Sandra Okafor",
    role: "Risk Manager, PrimeAuto",
    text: "The fraud detection alone paid for the platform in the first quarter. Highly recommended for any insurer.",
    stars: 5, // FIXED: Added missing star metrics
  },
];

export function LandingPage({ onNavigate }: LandingPageProps) {
  // Interactive 3D tilt for the hero visual (follows the cursor).
  const stageRef = useRef<HTMLDivElement>(null);
  const [tilt, setTilt] = useState({ rx: 0, ry: 0 });

  // Rotating headline word.
  const [wordIdx, setWordIdx] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setWordIdx((i) => (i + 1) % ROTATING_WORDS.length), 2400);
    return () => clearInterval(id);
  }, []);

  function handleTilt(e: React.MouseEvent) {
    const el = stageRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const px = (e.clientX - r.left) / r.width - 0.5; // -0.5 .. 0.5
    const py = (e.clientY - r.top) / r.height - 0.5;
    setTilt({ rx: -py * 12, ry: px * 16 });
  }
  function resetTilt() {
    setTilt({ rx: 0, ry: 0 });
  }

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#0a0a0a",
        fontFamily: "'Inter', system-ui, -apple-system, sans-serif",
        overflowX: "hidden",
        color: "#ffffff",
      }}
    >
      {/* Animations */}
      <style>{`
        @keyframes laserSweep {
          0% { top: 8%; opacity: 0.2; }
          15% { opacity: 0.9; }
          85% { opacity: 0.9; }
          100% { top: 92%; opacity: 0.2; }
        }
        @keyframes floatA { 0%,100% { transform: translateZ(60px) translateY(0); } 50% { transform: translateZ(60px) translateY(-16px); } }
        @keyframes floatB { 0%,100% { transform: translateZ(90px) translateY(0); } 50% { transform: translateZ(90px) translateY(12px); } }
        @keyframes orbDrift { 0%,100% { transform: translate(0,0); } 50% { transform: translate(40px,-30px); } }
        @keyframes orbDrift2 { 0%,100% { transform: translate(0,0); } 50% { transform: translate(-30px,25px); } }
        @keyframes wordSwap {
          0% { opacity: 0; transform: translateY(14px) rotateX(-40deg); filter: blur(4px); }
          100% { opacity: 1; transform: translateY(0) rotateX(0); filter: blur(0); }
        }
      `}</style>

      {/* Ambient 3D background — drifting gradient orbs */}
      <div style={{ position: "fixed", inset: 0, pointerEvents: "none", zIndex: -1, overflow: "hidden" }}>
        <div style={{ position: "absolute", top: "-160px", right: "-120px", width: "520px", height: "520px", background: "radial-gradient(circle, rgba(245,197,24,0.16) 0%, transparent 65%)", filter: "blur(20px)", animation: "orbDrift 14s ease-in-out infinite" }} />
        <div style={{ position: "absolute", bottom: "-200px", left: "-140px", width: "560px", height: "560px", background: "radial-gradient(circle, rgba(99,102,241,0.10) 0%, transparent 65%)", filter: "blur(20px)", animation: "orbDrift2 18s ease-in-out infinite" }} />
        {/* Perspective grid floor */}
        <div
          style={{
            position: "absolute",
            top: "260px",
            left: "-20%",
            right: "-20%",
            height: "520px",
            backgroundImage:
              "linear-gradient(rgba(245,197,24,0.10) 1px, transparent 1px), linear-gradient(90deg, rgba(245,197,24,0.10) 1px, transparent 1px)",
            backgroundSize: "44px 44px",
            transform: "perspective(420px) rotateX(62deg)",
            transformOrigin: "center top",
            maskImage: "radial-gradient(ellipse at center top, black 10%, transparent 70%)",
            WebkitMaskImage: "radial-gradient(ellipse at center top, black 10%, transparent 70%)",
          }}
        />
      </div>

      {/* ── Navbar ── */}
      <nav
        style={{
          position: "sticky",
          top: 0,
          zIndex: 50,
          background: "rgba(10,10,10,0.88)",
          backdropFilter: "blur(16px)",
          borderBottom: "1px solid #1e1e1e",
          padding: "0 48px",
          height: "64px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        {/* Logo */}
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
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
        </div>

        {/* Links */}
        <div style={{ display: "flex", alignItems: "center", gap: "36px" }}>
          {["Features", "Pricing", "About", "Contact"].map((l) => (
            <button
              key={l}
              style={{ background: "none", border: "none", color: "#888882", fontSize: "13px", cursor: "pointer", fontWeight: 500 }}
              onMouseEnter={(e) => (e.currentTarget.style.color = "#ffffff")}
              onMouseLeave={(e) => (e.currentTarget.style.color = "#888882")}
            >
              {l}
            </button>
          ))}
        </div>

        {/* Auth */}
        <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
          <button
            onClick={() => onNavigate("login")}
            style={{ background: "none", border: "none", color: "#cccccc", fontSize: "13px", cursor: "pointer", fontWeight: 500, padding: "8px 16px" }}
            onMouseEnter={(e) => (e.currentTarget.style.color = "#ffffff")}
            onMouseLeave={(e) => (e.currentTarget.style.color = "#cccccc")}
          >
            Log In
          </button>
          <button
            onClick={() => onNavigate("signup")}
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
            Sign Up Free
          </button>
        </div>
      </nav>

      {/* ── Hero Section ── */}
      <section style={{ maxWidth: "1400px", margin: "0 auto", padding: "70px 48px 0" }}>
        {/* Badge */}
        <div
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: "8px",
            background: "#1a1a1a",
            border: "1px solid #2a2a2a",
            borderRadius: "20px",
            padding: "6px 16px",
            marginBottom: "28px",
            fontSize: "11px",
            color: "var(--accent)",
            fontWeight: 600,
            letterSpacing: "0.06em",
          }}
        >
          <span style={{ width: "6px", height: "6px", background: "var(--accent)", borderRadius: "50%", display: "inline-block" }} />
          COMPUTER VISION TELEMETRY PLATFORM
        </div>

        {/* Flex Layout for Ideal Proportion Allocation */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "40px" }}>
          
          {/* Left Text Column */}
          <div style={{ flex: "0 0 450px", maxWidth: "450px" }}>
            <h1
              style={{
                fontSize: "clamp(38px, 4.2vw, 54px)",
                fontWeight: 800,
                color: "#ffffff",
                lineHeight: 1.1,
                marginBottom: "20px",
                letterSpacing: "-0.03em",
              }}
            >
              AI-Powered
              <br />
              <span
                style={{
                  display: "inline-block",
                  minHeight: "1.1em",
                  perspective: "600px",
                }}
              >
                <span
                  key={wordIdx}
                  style={{
                    display: "inline-block",
                    color: "var(--accent)",
                    transformOrigin: "center bottom",
                    animation: "wordSwap 0.5s cubic-bezier(0.22,1,0.36,1)",
                  }}
                >
                  {ROTATING_WORDS[wordIdx]}
                </span>
              </span>
              <br />
              Intelligence
            </h1>
            <p style={{ fontSize: "15px", color: "#888882", lineHeight: 1.8, marginBottom: "36px" }}>
              Accelerate commercial insurance claim pipelines with real-time neural surface topography evaluation — cutting review cycles from days to seconds.
            </p>

            <div style={{ display: "flex", gap: "12px", flexWrap: "wrap" }}>
              <button
                onClick={() => onNavigate("signup")}
                style={{
                  background: "var(--accent)",
                  color: "#0a0a0a",
                  border: "none",
                  borderRadius: "10px",
                  padding: "14px 28px",
                  fontSize: "14px",
                  fontWeight: 700,
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  gap: "8px",
                }}
                onMouseEnter={(e) => (e.currentTarget.style.background = "#fde047")}
                onMouseLeave={(e) => (e.currentTarget.style.background = "var(--accent)")}
              >
                Deploy Platform <ArrowRight size={15} />
              </button>
              <button
                onClick={() => onNavigate("login")}
                style={{
                  background: "transparent",
                  color: "#ffffff",
                  border: "1px solid #2a2a2a",
                  borderRadius: "10px",
                  padding: "14px 28px",
                  fontSize: "14px",
                  fontWeight: 500,
                  cursor: "pointer",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = "var(--accent)";
                  e.currentTarget.style.color = "var(--accent)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = "#2a2a2a";
                  e.currentTarget.style.color = "#ffffff";
                }}
              >
                Sign In →
              </button>
            </div>

            {/* Trust badges */}
            <div style={{ display: "flex", alignItems: "center", gap: "20px", marginTop: "36px" }}>
              {[
                { icon: <CheckCircle size={13} color="var(--accent)" />, label: "No credit card" },
                { icon: <CheckCircle size={13} color="var(--accent)" />, label: "14-day free trial" },
                { icon: <CheckCircle size={13} color="var(--accent)" />, label: "99.9% uptime SLA" },
              ].map((b) => (
                <div key={b.label} style={{ display: "flex", alignItems: "center", gap: "5px" }}>
                  {b.icon}
                  <span style={{ fontSize: "11px", color: "#666660" }}>{b.label}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Right Column: interactive 3D tilt stage — car blended into the background */}
          <div style={{ flex: 1, position: "relative", display: "flex", justifyContent: "center", perspective: "1200px" }}>
            <div
              ref={stageRef}
              onMouseMove={handleTilt}
              onMouseLeave={resetTilt}
              style={{
                position: "relative",
                width: "100%",
                maxWidth: "760px",
                transformStyle: "preserve-3d",
                transform: `rotateX(${tilt.rx}deg) rotateY(${tilt.ry}deg)`,
                transition: "transform 0.25s ease-out",
              }}
            >
              {/* Depth underglow — the car appears to emerge from the page */}
              <div
                style={{
                  position: "absolute",
                  inset: "-60px",
                  background: "radial-gradient(circle at center, rgba(245,197,24,0.16) 0%, transparent 66%)",
                  filter: "blur(30px)",
                  transform: "translateZ(-80px)",
                  pointerEvents: "none",
                }}
              />

              {/* Bare vehicle visual — no frame, edges fade into the dark background */}
              <div style={{ position: "relative", transform: "translateZ(40px)" }}>
                <img
                  src="images/damaged-car2.png"
                  alt="Vehicle damage telemetry system view"
                  style={{
                    width: "100%",
                    height: "auto",
                    objectFit: "contain",
                    display: "block",
                    mixBlendMode: "screen",
                    position: "relative",
                    zIndex: 2,
                    filter: "drop-shadow(0 40px 60px rgba(0,0,0,0.6))",
                    maskImage: "radial-gradient(ellipse 85% 85% at center, black 60%, transparent 100%)",
                    WebkitMaskImage: "radial-gradient(ellipse 85% 85% at center, black 60%, transparent 100%)",
                  }}
                />
                {/* Laser scan line */}
                <div
                  style={{
                    position: "absolute",
                    left: "10%",
                    right: "10%",
                    height: "2px",
                    background: "linear-gradient(90deg, transparent, rgba(245,197,24,0.7) 15%, rgba(245,197,24,0.7) 85%, transparent)",
                    boxShadow: "0 0 12px 3px rgba(245, 197, 24, 0.8)",
                    animation: "laserSweep 4.5s ease-in-out infinite alternate",
                    pointerEvents: "none",
                    zIndex: 3,
                  }}
                />
              </div>
            </div>
          </div>

        </div>
      </section>

      {/* ── Stats band ── */}
      <section
        style={{
          borderTop: "1px solid #1e1e1e",
          borderBottom: "1px solid #1e1e1e",
          margin: "96px 0 0",
          padding: "40px 48px",
          background: "#0d0d0d",
        }}
      >
        <div
          style={{
            maxWidth: "1100px",
            margin: "0 auto",
            display: "grid",
            gridTemplateColumns: "repeat(4, 1fr)",
            gap: "0",
          }}
        >
          {stats.map((s, i) => (
            <div
              key={s.label}
              style={{
                textAlign: "center",
                padding: "20px",
                borderRight: i < stats.length - 1 ? "1px solid #1e1e1e" : "none",
              }}
            >
              <div style={{ fontSize: "32px", fontWeight: 800, color: "var(--accent)", lineHeight: 1, letterSpacing: "-0.02em" }}>
                {s.value}
              </div>
              <div style={{ fontSize: "12px", color: "#666660", marginTop: "8px", fontWeight: 500 }}>
                {s.label}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Features ── */}
      <section style={{ maxWidth: "1100px", margin: "0 auto", padding: "96px 48px" }}>
        <div style={{ textAlign: "center", marginBottom: "56px" }}>
          <p style={{ fontSize: "11px", fontWeight: 700, letterSpacing: "0.1em", color: "var(--accent)", marginBottom: "12px" }}>
            PLATFORM CAPABILITIES
          </p>
          <h2
            style={{
              fontSize: "36px",
              fontWeight: 800,
              color: "#ffffff",
              marginBottom: "12px",
              letterSpacing: "-0.02em",
            }}
          >
            Everything to automate your claims
          </h2>
          <p style={{ fontSize: "14px", color: "#888882", maxWidth: "480px", margin: "0 auto", lineHeight: 1.7 }}>
            Purpose-built for commercial insurers, fleet operators, and collision repair networks worldwide.
          </p>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "20px" }}>
          {features.map((f, i) => (
            <div
              key={f.title}
              style={{
                background: "#111111",
                border: "1px solid #1e1e1e",
                borderRadius: "16px",
                padding: "32px 28px",
                transition: "transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease",
                transformStyle: "preserve-3d",
                cursor: "default",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = "var(--accent)";
                e.currentTarget.style.transform = "perspective(800px) translateY(-8px) rotateX(6deg)";
                e.currentTarget.style.boxShadow = "0 30px 60px -20px rgba(245,197,24,0.25), 0 18px 40px -18px rgba(0,0,0,0.7)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = "#1e1e1e";
                e.currentTarget.style.transform = "none";
                e.currentTarget.style.boxShadow = "none";
              }}
            >
              <div
                style={{
                  width: "44px",
                  height: "44px",
                  // FIXED: Changed backdrops to pure solid black (#000000)
                  background: "#000000", 
                  // FIXED: Active card gets golden border highlight, remaining get subtle dark borders
                  borderRadius: "12px",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  // FIXED: All internal elements use yellow icon fills now
                  color: "var(--accent)",
                  marginBottom: "18px",
                }}
              >
                {f.icon}
              </div>
              <h3 style={{ fontSize: "15px", fontWeight: 700, color: "#ffffff", marginBottom: "10px" }}>
                {f.title}
              </h3>
              <p style={{ fontSize: "13px", color: "#666660", lineHeight: 1.75 }}>{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── How it works ── */}
      <section style={{ background: "#0d0d0d", borderTop: "1px solid #1e1e1e", borderBottom: "1px solid #1e1e1e", padding: "80px 48px" }}>
        <div style={{ maxWidth: "1100px", margin: "0 auto" }}>
          <div style={{ textAlign: "center", marginBottom: "52px" }}>
            <p style={{ fontSize: "11px", fontWeight: 700, letterSpacing: "0.1em", color: "var(--accent)", marginBottom: "12px" }}>
              HOW IT WORKS
            </p>
            <h2 style={{ fontSize: "34px", fontWeight: 800, color: "#ffffff", letterSpacing: "-0.02em" }}>
              Up and running in minutes
            </h2>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "0", position: "relative" }}>
            {/* Connector line */}
            <div
              style={{
                position: "absolute",
                top: "28px",
                left: "12.5%",
                right: "12.5%",
                height: "1px",
                background: "linear-gradient(to right, var(--accent), var(--accent), var(--accent), var(--accent))",
                zIndex: 0,
              }}
            />
            {[
              { step: "01", title: "Register Vehicle", desc: "Enter vehicle details and capture all 4 sides via our guided upload flow." },
              { step: "02", title: "AI Analysis", desc: "Our model processes damage in real time, generating a topographic damage map." },
              { step: "03", title: "Report Generated", desc: "Receive a structured report with severity scores, repair estimates, and photos." },
              { step: "04", title: "Claim Filed", desc: "Push the verified report directly to your insurer's claims management system." },
            ].map((item, i) => (
              <div key={item.step} style={{ padding: "0 24px", textAlign: "center", position: "relative", zIndex: 1 }}>
                <div
                  style={{
                    width: "56px",
                    height: "56px",
                    // FIXED: All layout badges (including 03 & 04) now display uniformly in active yellow style
                    background: "var(--accent)",
                    border: "none",
                    borderRadius: "50%",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    margin: "0 auto 20px",
                    fontSize: "13px",
                    fontWeight: 800,
                    color: "#0a0a0a",
                    letterSpacing: "0.02em",
                    boxShadow: "0 14px 28px rgba(245,197,24,0.35), 0 0 0 6px rgba(245,197,24,0.08)",
                  }}
                >
                  {item.step}
                </div>
                <h4 style={{ fontSize: "14px", fontWeight: 700, color: "#ffffff", marginBottom: "8px" }}>
                  {item.title}
                </h4>
                <p style={{ fontSize: "12px", color: "#666660", lineHeight: 1.7 }}>{item.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Testimonials ── */}
      <section style={{ maxWidth: "1100px", margin: "0 auto", padding: "96px 48px" }}>
        <div style={{ textAlign: "center", marginBottom: "52px" }}>
          <p style={{ fontSize: "11px", fontWeight: 700, letterSpacing: "0.1em", color: "var(--accent)", marginBottom: "12px" }}>
            WHAT CUSTOMERS SAY
          </p>
          <h2 style={{ fontSize: "34px", fontWeight: 800, color: "#ffffff", letterSpacing: "-0.02em" }}>
            Trusted by industry leaders
          </h2> 
        </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "20px" }}>
          {testimonials.map((t) => (
            <div
              key={t.name}
              style={{
                background: "#111111",
                border: "1px solid #1e1e1e",
                borderRadius: "16px",
                padding: "28px",
              }}
            >
              <div style={{ display: "flex", gap: "2px", marginBottom: "16px" }}>
                {Array.from({ length: t.stars }).map((_, i) => (
                  <Star key={i} size={13} fill="var(--accent)" color="var(--accent)" />
                ))}
              </div>
              <p style={{ fontSize: "13px", color: "#cccccc", lineHeight: 1.75, marginBottom: "20px" }}>
                "{t.text}"
              </p>
              <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                <div
                  style={{
                    width: "36px",
                    height: "36px",
                    background: "var(--accent)",
                    borderRadius: "50%",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: "13px",
                    fontWeight: 700,
                    color: "#0a0a0a",
                    flexShrink: 0,
                  }}
                >
                  {t.name[0]}
                </div>
                <div>
                  <div style={{ fontSize: "13px", fontWeight: 600, color: "#ffffff" }}>{t.name}</div>
                  <div style={{ fontSize: "11px", color: "#666660" }}>{t.role}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ── CTA banner ── */}
      <section style={{ maxWidth: "1100px", margin: "0 auto 96px", padding: "0 48px" }}>
        <div
          style={{
            background: "var(--accent)",
            borderRadius: "24px",
            padding: "64px 56px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: "40px",
            position: "relative",
            overflow: "hidden",
          }}
        >
          {/* Decorative circles */}
          <div style={{ position: "absolute", right: "160px", top: "-40px", width: "180px", height: "180px", background: "rgba(0,0,0,0.06)", borderRadius: "50%" }} />
          <div style={{ position: "absolute", right: "80px", bottom: "-60px", width: "220px", height: "220px", background: "rgba(0,0,0,0.04)", borderRadius: "50%" }} />

          <div style={{ position: "relative" }}>
            <h2 style={{ fontSize: "30px", fontWeight: 800, color: "#0a0a0a", marginBottom: "10px", letterSpacing: "-0.02em" }}>
              Ready to transform your claims workflow?
            </h2>
            <div style={{ display: "flex", gap: "20px" }}>
              {["No credit card required", "14-day free trial", "Cancel anytime"].map((item) => (
                <div key={item} style={{ display: "flex", alignItems: "center", gap: "5px" }}>
                  <CheckCircle size={13} color="#0a0a0a" />
                  <span style={{ fontSize: "12px", color: "rgba(0,0,0,0.65)", fontWeight: 500 }}>{item}</span>
                </div>
              ))}
            </div>
          </div>
          <button
            onClick={() => onNavigate("signup")}
            style={{
              background: "#0a0a0a",
              color: "var(--accent)",
              border: "none",
              borderRadius: "12px",
              padding: "16px 32px",
              fontSize: "15px",
              fontWeight: 700,
              cursor: "pointer",
              whiteSpace: "nowrap",
              flexShrink: 0,
              position: "relative",
            }}
            onMouseEnter={(e) => (e.currentTarget.style.background = "#1a1a1a")}
            onMouseLeave={(e) => (e.currentTarget.style.background = "#0a0a0a")}
          >
            Get Started Free →
          </button>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer
        style={{
          borderTop: "1px solid #1e1e1e",
          padding: "32px 48px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          maxWidth: "1100px",
          margin: "0 auto",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <div style={{ width: "28px", height: "28px", background: "var(--accent)", borderRadius: "7px", display: "flex", alignItems: "center", justifyContent: "center" }}>
            <Car size={14} color="#0a0a0a" />
          </div>
          <span style={{ fontSize: "13px", color: "#444440", fontWeight: 500 }}>AutoReg AI © 2026. All rights reserved.</span>
        </div>
        <div style={{ display: "flex", gap: "28px" }}>
          {["Privacy", "Terms", "Security", "Support"].map((l) => (
            <button key={l} style={{ background: "none", border: "none", fontSize: "12px", color: "#444440", cursor: "pointer" }}
              onMouseEnter={(e) => (e.currentTarget.style.color = "var(--accent)")}
              onMouseLeave={(e) => (e.currentTarget.style.color = "#444440")}
            >
              {l}
            </button>
          ))}
        </div>
      </footer>
    </div>
  );
}