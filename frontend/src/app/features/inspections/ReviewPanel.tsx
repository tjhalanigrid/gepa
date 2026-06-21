import { useState } from "react";
import {
  DamagePartEntry,
  CorrectionAction,
  saveCorrection,
  VALID_DAMAGE_CLASSES,
  VALID_PARTS,
  VALID_SEVERITIES,
  prettifyLabel,
} from "../../lib/api";

interface ReviewPanelProps {
  sessionId: string;
  initialFindings: DamagePartEntry[];
  onSaved?: () => void;
}

interface Row {
  key: string;
  original: DamagePartEntry | null; // null → human-added finding
  damage: string;
  part: string;
  severity: string;
}

let _k = 0;
const nextKey = () => `r${_k++}`;

/**
 * TEMPORARY human-review panel for collecting GEPA ground truth.
 *
 * Shows the AI's findings as an editable list. A reviewer keeps / edits / removes
 * each one and can add damages the AI missed, then submits. This POSTs to
 * /session/{id}/save_correction, which writes a verified line to
 * corrections_log.jsonl — the ground-truth source GEPA optimizes against.
 *
 * Cost is NOT a human input — the backend recomputes it from COST_DB. The reviewer
 * only judges what they see: damage type, part, and severity.
 */
export function ReviewPanel({ sessionId, initialFindings, onSaved }: ReviewPanelProps) {
  const [rows, setRows] = useState<Row[]>(
    initialFindings.map((f) => ({
      key: nextKey(),
      original: f,
      damage: f.damage,
      part: f.part,
      severity: f.severity,
    })),
  );
  const [removed, setRemoved] = useState<DamagePartEntry[]>([]);
  const [annotatedBy, setAnnotatedBy] = useState("");
  const [notes, setNotes] = useState("");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function update(key: string, field: "damage" | "part" | "severity", value: string) {
    setRows((rs) => rs.map((r) => (r.key === key ? { ...r, [field]: value } : r)));
  }

  function removeRow(key: string) {
    setRows((rs) => {
      const row = rs.find((r) => r.key === key);
      if (row?.original) setRemoved((rm) => [...rm, row.original as DamagePartEntry]);
      return rs.filter((r) => r.key !== key);
    });
  }

  function addRow() {
    setRows((rs) => [
      ...rs,
      { key: nextKey(), original: null, damage: VALID_DAMAGE_CLASSES[0], part: VALID_PARTS[0], severity: "moderate" },
    ]);
  }

  function buildPayload() {
    const entry = (r: Row): DamagePartEntry => ({
      damage: r.damage,
      part: r.part,
      severity: r.severity,
      // Kept rows retain original cost; changed/added send 0 — backend recomputes totals.
      cost_min: r.original && unchanged(r) ? r.original.cost_min : 0,
      cost_max: r.original && unchanged(r) ? r.original.cost_max : 0,
    });
    const actions: CorrectionAction[] = [];
    for (const r of rows) {
      if (!r.original) {
        actions.push({ action: "add", original: null, corrected: entry(r) });
      } else if (unchanged(r)) {
        actions.push({ action: "keep", original: r.original, corrected: r.original });
      } else {
        actions.push({ action: "edit", original: r.original, corrected: entry(r) });
      }
    }
    for (const o of removed) {
      actions.push({ action: "remove", original: o, corrected: null });
    }
    return {
      correction_actions: actions,
      bbox_annotations: [],
      final_damage_map: rows.map(entry),
      annotated_by: annotatedBy || undefined,
      notes: notes || undefined,
    };
  }

  function unchanged(r: Row): boolean {
    return (
      !!r.original &&
      r.original.damage === r.damage &&
      r.original.part === r.part &&
      r.original.severity === r.severity
    );
  }

  async function submit() {
    if (!annotatedBy.trim()) {
      setError("Enter your name (reviewer) before submitting.");
      return;
    }
    setSaving(true);
    setError(null);
    try {
      await saveCorrection(sessionId, buildPayload());
      setSaved(true);
      onSaved?.();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not save correction.");
    } finally {
      setSaving(false);
    }
  }

  const sel: React.CSSProperties = {
    padding: "6px 8px", borderRadius: "8px", border: "1px solid rgba(0,0,0,0.15)",
    fontSize: "13px", background: "#fff",
  };

  if (saved) {
    return (
      <div style={{ padding: "16px", borderRadius: "12px", background: "#ecfdf5", border: "1px solid #6ee7b7", marginTop: "16px" }}>
        <strong style={{ color: "#047857" }}>✓ Review saved.</strong>{" "}
        <span style={{ color: "#065f46" }}>Ground truth recorded for GEPA. You can review the next image.</span>
      </div>
    );
  }

  return (
    <div style={{ padding: "16px", borderRadius: "12px", background: "#fffbeb", border: "1px solid #fcd34d", marginTop: "16px" }}>
      <div style={{ fontWeight: 700, marginBottom: "4px" }}>Human Review — verify the AI findings</div>
      <div style={{ fontSize: "13px", color: "#92400e", marginBottom: "12px" }}>
        Fix the damage type, part, and severity for each finding. Remove anything that isn't real
        damage, and add anything the AI missed. Cost is computed automatically.
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
        {rows.map((r, i) => (
          <div key={r.key} style={{ display: "flex", gap: "8px", alignItems: "center", flexWrap: "wrap" }}>
            <span style={{ width: "20px", color: "#92400e", fontWeight: 600 }}>{i + 1}.</span>
            <select style={sel} value={r.damage} onChange={(e) => update(r.key, "damage", e.target.value)}>
              {VALID_DAMAGE_CLASSES.map((d) => <option key={d} value={d}>{prettifyLabel(d)}</option>)}
            </select>
            <span style={{ color: "#92400e" }}>·</span>
            <select style={sel} value={r.part} onChange={(e) => update(r.key, "part", e.target.value)}>
              {VALID_PARTS.map((p) => <option key={p} value={p}>{prettifyLabel(p)}</option>)}
            </select>
            <select style={sel} value={r.severity} onChange={(e) => update(r.key, "severity", e.target.value)}>
              {VALID_SEVERITIES.map((s) => <option key={s} value={s}>{prettifyLabel(s)}</option>)}
            </select>
            <button
              onClick={() => removeRow(r.key)}
              style={{ marginLeft: "auto", padding: "4px 10px", borderRadius: "8px", border: "1px solid #fca5a5", background: "#fef2f2", color: "#b91c1c", cursor: "pointer", fontSize: "12px" }}
            >
              Remove
            </button>
          </div>
        ))}
        {rows.length === 0 && (
          <div style={{ fontSize: "13px", color: "#92400e", fontStyle: "italic" }}>
            No findings. Add any visible damage, or submit empty if the vehicle is undamaged.
          </div>
        )}
      </div>

      <button
        onClick={addRow}
        style={{ marginTop: "12px", padding: "6px 12px", borderRadius: "8px", border: "1px dashed #d97706", background: "transparent", color: "#b45309", cursor: "pointer", fontSize: "13px" }}
      >
        + Add missed damage
      </button>

      <div style={{ display: "flex", gap: "8px", marginTop: "16px", flexWrap: "wrap", alignItems: "center" }}>
        <input
          placeholder="Your name (reviewer)"
          value={annotatedBy}
          onChange={(e) => setAnnotatedBy(e.target.value)}
          style={{ ...sel, flex: "0 0 180px" }}
        />
        <input
          placeholder="Notes (optional)"
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          style={{ ...sel, flex: "1 1 220px" }}
        />
        <button
          onClick={submit}
          disabled={saving}
          style={{ padding: "8px 18px", borderRadius: "8px", border: "none", background: saving ? "#9ca3af" : "#047857", color: "#fff", cursor: saving ? "default" : "pointer", fontWeight: 600, fontSize: "14px" }}
        >
          {saving ? "Saving…" : "Submit review"}
        </button>
      </div>

      {error && <div style={{ marginTop: "10px", color: "#b91c1c", fontSize: "13px" }}>{error}</div>}
    </div>
  );
}
