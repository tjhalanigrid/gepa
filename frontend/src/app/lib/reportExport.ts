// Builds a self-contained, printable report document from a Claim and exports it
// as PDF (via the browser print pipeline) or as a Word .doc (HTML blob). The
// claim's photo is a data URL, so it embeds directly — no network needed.

import { formatCostRange } from "./api";
import { CLASS_COLORS } from "../features/inspections/AnnotatedImage";
import type { Claim } from "./claimsStore";

function esc(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function fmtDate(iso: string): string {
  return new Date(iso).toLocaleDateString("en-US", { month: "long", day: "numeric", year: "numeric" });
}

function avgConfidence(c: Claim): number {
  const scored = c.findings.filter((f) => f.confidence > 0);
  if (!scored.length) return 0;
  return Math.round(scored.reduce((s, f) => s + f.confidence, 0) / scored.length);
}

const STATUS_TEXT: Record<string, string> = {
  "auto-approved": "Auto-approved",
  "pending-review": "Pending human review",
  "no-damage": "No visible damage",
};

const SEV_COLOR: Record<string, string> = { Severe: "#b91c1c", Moderate: "#c2410c", Minor: "#854d0e" };

/** Annotated damage photo: the image with percent-positioned boxes over it. */
function photoBlock(c: Claim): string {
  if (!c.thumbnail) return "";
  const boxes = (c.detections ?? [])
    .map((d) => {
      if (!Array.isArray(d.bbox) || c.imgW <= 0 || c.imgH <= 0) return "";
      const [x1, y1, x2, y2] = d.bbox;
      const left = (x1 / c.imgW) * 100;
      const top = (y1 / c.imgH) * 100;
      const w = ((x2 - x1) / c.imgW) * 100;
      const h = ((y2 - y1) / c.imgH) * 100;
      const color = CLASS_COLORS[d.damage] ?? "#ef4444";
      return (
        `<div style="position:absolute;left:${left}%;top:${top}%;width:${w}%;height:${h}%;` +
        `border:2px solid ${color};box-sizing:border-box;">` +
        `<span style="position:absolute;top:-16px;left:-2px;background:${color};color:#fff;` +
        `font-size:10px;font-weight:700;padding:0 4px;border-radius:3px;">${d.index}</span></div>`
      );
    })
    .join("");
  return (
    `<div style="position:relative;display:inline-block;max-width:100%;">` +
    `<img src="${c.thumbnail}" style="max-width:100%;display:block;border-radius:8px;" />` +
    boxes +
    `</div>`
  );
}

function findingsRows(c: Claim): string {
  if (!c.findings.length) return `<tr><td colspan="4" style="padding:10px;color:#888;">No damage detected.</td></tr>`;
  const rows = c.findings
    .map(
      (r) =>
        `<tr>` +
        `<td style="padding:8px 10px;border-bottom:1px solid #eee;">${esc(r.area)}</td>` +
        `<td style="padding:8px 10px;border-bottom:1px solid #eee;">${esc(r.damage)}</td>` +
        `<td style="padding:8px 10px;border-bottom:1px solid #eee;color:${SEV_COLOR[r.severity] ?? "#444"};font-weight:700;">${esc(r.severity)}</td>` +
        `<td style="padding:8px 10px;border-bottom:1px solid #eee;font-weight:700;">${esc(formatCostRange(r.costMin, r.costMax))}</td>` +
        `</tr>`,
    )
    .join("");
  const total =
    `<tr style="background:#fffbeb;">` +
    `<td colspan="3" style="padding:10px;font-weight:700;border-top:2px solid #f5c518;">TOTAL ESTIMATE</td>` +
    `<td style="padding:10px;font-weight:800;border-top:2px solid #f5c518;">${esc(formatCostRange(c.totalMin, c.totalMax))}</td>` +
    `</tr>`;
  return rows + total;
}

/** Build the full report as a standalone HTML document. */
export function buildReportHtml(c: Claim, forPrint = false): string {
  const conf = avgConfidence(c);
  const statusText = STATUS_TEXT[c.status] ?? c.status;
  const summary =
    `This report presents an AI-assisted damage assessment for a ${esc(c.vehicle)}. ` +
    `${c.findings.length} damaged component${c.findings.length === 1 ? " was" : "s were"} identified, with an ` +
    `estimated total repair cost of <strong>${esc(formatCostRange(c.totalMin, c.totalMax))}</strong>. ` +
    `The claim was ${esc(statusText.toLowerCase())} by the assessment pipeline.`;

  const stat = (value: string, label: string) =>
    `<td style="text-align:center;padding:14px;background:#f5f5f0;border-radius:8px;">` +
    `<div style="font-size:16px;font-weight:800;color:#0a0a0a;">${esc(value)}</div>` +
    `<div style="font-size:10px;color:#888;margin-top:2px;">${esc(label)}</div></td>`;

  const printScript = forPrint
    ? `<script>window.onload=function(){setTimeout(function(){window.print();},250);};window.onafterprint=function(){window.close();};</script>`
    : "";

  return `<!DOCTYPE html>
<html xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:w="urn:schemas-microsoft-com:office:word" xmlns="http://www.w3.org/TR/REC-html40">
<head>
<meta charset="utf-8" />
<title>Report ${esc(c.id)}</title>
<style>
  @page { size: A4; margin: 18mm; }
  body { font-family: 'Calibri','Inter',Arial,sans-serif; color:#1a1a1a; font-size:13px; line-height:1.6; }
  h1 { font-size:20px; margin:0 0 4px; }
  h2 { font-size:15px; margin:24px 0 10px; border-bottom:1px solid #ddd; padding-bottom:4px; }
  table { width:100%; border-collapse:collapse; }
  .muted { color:#888; font-size:11px; }
</style>
${printScript}
</head>
<body>
  <div style="text-align:center;border-bottom:2px solid #f5c518;padding-bottom:14px;margin-bottom:18px;">
    <div style="font-size:10px;font-weight:700;letter-spacing:0.12em;color:#caa000;">VEHICLE DAMAGE ASSESSMENT REPORT</div>
    <h1>${esc(c.id)} &nbsp;|&nbsp; ${esc(c.vehicle)}</h1>
    <div class="muted">Generated ${esc(fmtDate(c.createdAt))}${conf > 0 ? ` &nbsp;|&nbsp; AI Confidence: ${conf}%` : ""}</div>
  </div>

  <h2>Executive Summary</h2>
  <p>${summary}</p>
  <table style="margin-top:12px;"><tr>
    ${stat(formatCostRange(c.totalMin, c.totalMax), "Total Repair Cost")}
    <td style="width:10px;"></td>
    ${stat(conf > 0 ? `${conf}%` : "—", "AI Confidence")}
    <td style="width:10px;"></td>
    ${stat(String(c.detections.length), "Damage Regions")}
  </tr></table>

  <h2>Damage Photo</h2>
  ${photoBlock(c) || '<p class="muted">No photo attached.</p>'}

  ${
    c.maskThumbnail
      ? `<h2>SAM2 Damage Mask</h2>
  <img src="${c.maskThumbnail}" style="max-width:100%;display:block;border-radius:8px;" />
  <p class="muted" style="margin-top:6px;">Pixel-level segmentation of each damaged region, produced by SAM2.</p>`
      : ""
  }

  ${
    c.mergedThumbnail
      ? `<h2>Merged Detection (VLM &cup; SAM2)</h2>
  <img src="${c.mergedThumbnail}" style="max-width:100%;display:block;border-radius:8px;" />
  <p class="muted" style="margin-top:6px;">Union of the VLM damage boxes and SAM2 regions, source-coloured.</p>`
      : ""
  }

  <h2>Cost Breakdown</h2>
  <table>
    <thead>
      <tr style="background:#f5f5f0;">
        <th style="text-align:left;padding:8px 10px;font-size:11px;color:#666;">Component</th>
        <th style="text-align:left;padding:8px 10px;font-size:11px;color:#666;">Damage</th>
        <th style="text-align:left;padding:8px 10px;font-size:11px;color:#666;">Severity</th>
        <th style="text-align:left;padding:8px 10px;font-size:11px;color:#666;">Estimated Cost</th>
      </tr>
    </thead>
    <tbody>${findingsRows(c)}</tbody>
  </table>

  <p class="muted" style="margin-top:28px;border-top:1px solid #eee;padding-top:10px;">
    Generated by AutoReg AI — AI-assisted assessment. Cost figures computed from the COST_DB reference table.
  </p>
</body>
</html>`;
}

/** Export the report as a PDF via the browser print dialog ("Save as PDF"). */
export function exportReportPdf(c: Claim): void {
  const html = buildReportHtml(c, true);
  const w = window.open("", "_blank", "width=900,height=1000");
  if (!w) {
    alert("Please allow pop-ups to export the PDF.");
    return;
  }
  w.document.open();
  w.document.write(html);
  w.document.close();
}

/** Export the report as a downloadable Word .doc document. */
export function exportReportWord(c: Claim): void {
  const html = buildReportHtml(c, false);
  // Leading BOM forces Word to read the file as UTF-8 (so ₹ renders correctly).
  const blob = new Blob(["﻿", html], { type: "application/msword" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${c.id}-damage-report.doc`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
