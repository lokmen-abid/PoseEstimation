"""
report_generator.py
────────────────────────────────────────────────────────────────
Génère le rapport PDF d'analyse biomécanique d'une session.

Utilise ReportLab Platypus (mise en page automatique avec sauts
de page) + Canvas pour l'en-tête et le pied de page.

Structure du PDF :
  1. En-tête  — logo IA/Serve, infos session
  2. Phases   — tableau des frames clés annotées
  3. Métriques articulaires — min / max / mean / std par joint
  4. Comparaison normative  — mesuré vs Gorce 2024, statut
  5. Alertes cliniques      — (section conditionnelle)
  6. Pied de page           — date de génération, numéro de page
"""

import io
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# ── Palette (calquée sur le design system IA/Serve) ────────────

C_BG_DARK   = colors.HexColor("#0A1628")   # fond nuit
C_BLUE      = colors.HexColor("#38BDF8")   # accent IA
C_GREEN     = colors.HexColor("#10F5A0")   # accent tennis
C_INDIGO    = colors.HexColor("#6366F1")   # admin / titres
C_SLATE     = colors.HexColor("#94A3B8")   # textes secondaires
C_WHITE     = colors.white
C_WARNING   = colors.HexColor("#EF9F27")   # warning
C_CRITICAL  = colors.HexColor("#EF4444")   # critical
C_OK        = colors.HexColor("#10F5A0")   # dans la norme
C_ROW_EVEN  = colors.HexColor("#F8FAFC")   # ligne paire (fond clair)
C_ROW_ODD   = colors.HexColor("#EEF2FF")   # ligne impaire légère
C_BORDER    = colors.HexColor("#CBD5E1")   # bordure tableaux

PAGE_W, PAGE_H = A4                         # 595.27 x 841.89 pts
MARGIN = 1.8 * cm

# ── Styles de texte ────────────────────────────────────────────

_base = getSampleStyleSheet()

S_TITLE = ParagraphStyle(
    "title",
    parent=_base["Normal"],
    fontSize=20,
    textColor=C_WHITE,
    fontName="Helvetica-Bold",
    spaceAfter=2,
)
S_SUBTITLE = ParagraphStyle(
    "subtitle",
    parent=_base["Normal"],
    fontSize=10,
    textColor=C_BLUE,
    fontName="Helvetica",
    spaceAfter=0,
)
S_SECTION = ParagraphStyle(
    "section",
    parent=_base["Normal"],
    fontSize=12,
    textColor=C_INDIGO,
    fontName="Helvetica-Bold",
    spaceBefore=14,
    spaceAfter=6,
)
S_BODY = ParagraphStyle(
    "body",
    parent=_base["Normal"],
    fontSize=9,
    textColor=colors.HexColor("#1E293B"),
    fontName="Helvetica",
    leading=13,
)
S_SMALL = ParagraphStyle(
    "small",
    parent=_base["Normal"],
    fontSize=8,
    textColor=C_SLATE,
    fontName="Helvetica",
    leading=11,
)
S_TABLE_HEADER = ParagraphStyle(
    "th",
    parent=_base["Normal"],
    fontSize=8,
    textColor=C_WHITE,
    fontName="Helvetica-Bold",
    alignment=TA_CENTER,
)
S_TABLE_CELL = ParagraphStyle(
    "td",
    parent=_base["Normal"],
    fontSize=8,
    textColor=colors.HexColor("#1E293B"),
    fontName="Helvetica",
    alignment=TA_CENTER,
)
S_TABLE_CELL_LEFT = ParagraphStyle(
    "td_left",
    parent=S_TABLE_CELL,
    alignment=TA_LEFT,
)

# ── Labels traduits ────────────────────────────────────────────

JOINT_LABELS: Dict[str, str] = {
    "knee_flexion_right":       "Flexion genou D",
    "knee_flexion_left":        "Flexion genou G",
    "trunk_inclination":        "Inclinaison tronc",
    "trunk_rotation":           "Rotation tronc",
    "shoulder_rotation_right":  "Rotation épaule D",
    "shoulder_elevation_right": "Élévation épaule D",
    "shoulder_elevation_left":  "Élévation épaule G",
    "elbow_right":              "Flexion coude D",
    "elbow_left":               "Flexion coude G",
    "hip_right":                "Hanche D",
    "hip_left":                 "Hanche G",
    "pelvis_rotation":          "Rotation bassin",
    "shoulder_separation":      "Séparation épaules",
    "wrist_extension_right":    "Extension poignet D",
    "wrist_extension_left":     "Extension poignet G",
}

GESTURE_LABELS: Dict[str, str] = {
    "service":    "Service",
    "coup_droit": "Coup droit",
    "revers":     "Revers",
}

PHASE_LABELS: Dict[str, str] = {
    "trophy_position":  "Trophy Position",
    "racket_low_point": "Racket Low Point",
    "ball_impact":      "Ball Impact",
    "preparation":      "Préparation",
    "acceleration":     "Accélération",
    "follow_through":   "Follow Through",
}


def _label_joint(key: str) -> str:
    return JOINT_LABELS.get(key, key.replace("_", " ").title())


def _label_phase(key: str) -> str:
    return PHASE_LABELS.get(key, key.replace("_", " ").title())


# ── En-tête et pied de page (Canvas) ──────────────────────────

class _ReportCanvas:
    """
    Mixin injecté dans BaseDocTemplate pour dessiner l'en-tête
    IA/Serve et le pied de page sur chaque page.
    """

    def __init__(self, athlete_name: str, session_date: str, gesture: str):
        self.athlete_name = athlete_name
        self.session_date = session_date
        self.gesture      = gesture

    def _draw_header(self, canvas, doc):
        """Bandeau sombre en haut avec logo + infos session."""
        canvas.saveState()

        # Fond du bandeau
        header_h = 2.2 * cm
        canvas.setFillColor(C_BG_DARK)
        canvas.rect(0, PAGE_H - header_h, PAGE_W, header_h, fill=1, stroke=0)

        # Logo "IA/Serve"
        canvas.setFont("Helvetica-Bold", 16)
        canvas.setFillColor(C_BLUE)
        canvas.drawString(MARGIN, PAGE_H - 1.4 * cm, "IA")
        canvas.setFillColor(C_GREEN)
        canvas.drawString(MARGIN + 22, PAGE_H - 1.4 * cm, "/")
        canvas.setFillColor(C_WHITE)
        canvas.drawString(MARGIN + 31, PAGE_H - 1.4 * cm, "Serve")

        # Sous-titre — Rapport d'analyse biomécanique
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(C_SLATE)
        canvas.drawString(MARGIN, PAGE_H - 1.85 * cm, "Rapport d'analyse biomécanique")

        # Infos à droite
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(C_WHITE)
        right_x = PAGE_W - MARGIN
        canvas.drawRightString(right_x, PAGE_H - 1.2 * cm, self.athlete_name)
        canvas.setFillColor(C_SLATE)
        canvas.drawRightString(right_x, PAGE_H - 1.65 * cm,
                               f"{self.gesture}  ·  {self.session_date}")

        # Ligne de séparation
        canvas.setStrokeColor(C_INDIGO)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, PAGE_H - header_h, PAGE_W - MARGIN, PAGE_H - header_h)

        canvas.restoreState()

    def _draw_footer(self, canvas, doc):
        """Pied de page : date de génération + numéro de page."""
        canvas.saveState()

        y = 0.85 * cm
        now = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")

        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(C_SLATE)
        canvas.drawString(MARGIN, y, f"Généré le {now}  ·  IA/Serve — Plateforme d'analyse biomécanique tennis")
        canvas.drawRightString(PAGE_W - MARGIN, y, f"Page {doc.page}")

        # Ligne
        canvas.setStrokeColor(C_BORDER)
        canvas.setLineWidth(0.3)
        canvas.line(MARGIN, y + 0.5 * cm, PAGE_W - MARGIN, y + 0.5 * cm)

        canvas.restoreState()

    def on_page(self, canvas, doc):
        self._draw_header(canvas, doc)
        self._draw_footer(canvas, doc)


# ── Constructeurs de tableaux ──────────────────────────────────

def _table_style_base() -> list:
    """Style de base commun à tous les tableaux."""
    return [
        ("BACKGROUND",  (0, 0), (-1, 0),  C_BG_DARK),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  C_WHITE),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0),  8),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_ROW_EVEN, C_ROW_ODD]),
        ("GRID",        (0, 0), (-1, -1), 0.3, C_BORDER),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]


def _build_phases_table(phase_annotations: Dict[str, int]) -> Table:
    """Tableau des phases détectées (phase → frame)."""
    headers = [
        Paragraph("Phase", S_TABLE_HEADER),
        Paragraph("Frame #", S_TABLE_HEADER),
    ]
    rows = [headers]
    for phase_key, frame_num in phase_annotations.items():
        rows.append([
            Paragraph(_label_phase(phase_key), S_TABLE_CELL_LEFT),
            Paragraph(str(frame_num), S_TABLE_CELL),
        ])

    col_widths = [9 * cm, 5 * cm]
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle(_table_style_base() + [
        ("ALIGN", (0, 1), (0, -1), "LEFT"),
    ]))
    return t


def _build_metrics_table(joint_metrics: Dict[str, Any]) -> Table:
    """Tableau métriques articulaires : min / max / mean / std."""
    headers = [
        Paragraph("Articulation",  S_TABLE_HEADER),
        Paragraph("Min (°)",       S_TABLE_HEADER),
        Paragraph("Max (°)",       S_TABLE_HEADER),
        Paragraph("Moyenne (°)",   S_TABLE_HEADER),
        Paragraph("Écart-type",    S_TABLE_HEADER),
    ]
    rows = [headers]
    for joint_key, m in joint_metrics.items():
        rows.append([
            Paragraph(_label_joint(joint_key), S_TABLE_CELL_LEFT),
            Paragraph(f"{m.get('min', 0):.1f}", S_TABLE_CELL),
            Paragraph(f"{m.get('max', 0):.1f}", S_TABLE_CELL),
            Paragraph(f"{m.get('mean', 0):.1f}", S_TABLE_CELL),
            Paragraph(f"{m.get('std', 0):.1f}",  S_TABLE_CELL),
        ])

    col_widths = [5.5 * cm, 2.5 * cm, 2.5 * cm, 2.8 * cm, 2.8 * cm]
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle(_table_style_base() + [
        ("ALIGN", (0, 1), (0, -1), "LEFT"),
    ]))
    return t


def _build_normative_table(normative_comparison: Dict[str, Any]) -> Table:
    """Tableau comparaison normative vs Gorce 2024."""
    headers = [
        Paragraph("Articulation",   S_TABLE_HEADER),
        Paragraph("Mesuré (°)",     S_TABLE_HEADER),
        Paragraph("Normative (°)",  S_TABLE_HEADER),
        Paragraph("Δ (°)",          S_TABLE_HEADER),
        Paragraph("Source",         S_TABLE_HEADER),
        Paragraph("Statut",         S_TABLE_HEADER),
    ]
    rows = [headers]
    style_cmds = list(_table_style_base())

    for row_idx, (joint_key, comp) in enumerate(normative_comparison.items(), start=1):
        # comp peut être un dict (format complet) ou un float (format legacy)
        if isinstance(comp, dict):
            measured   = comp.get("measured_mean",  0) or 0
            normative  = comp.get("normative_mean", 0) or 0
            delta      = comp.get("delta_degrees",  0) or 0
            within_1sd = comp.get("within_1std",    True)
            source     = comp.get("source",         "—")
        else:
            # Format legacy : valeur numérique directe = delta
            measured, normative, delta, within_1sd, source = 0, 0, float(comp), True, "—"

        statut_txt   = "Dans 1σ" if within_1sd else "Hors 1σ"
        statut_color = C_OK     if within_1sd else C_WARNING

        rows.append([
            Paragraph(_label_joint(joint_key), S_TABLE_CELL_LEFT),
            Paragraph(f"{measured:.1f}",  S_TABLE_CELL),
            Paragraph(f"{normative:.1f}", S_TABLE_CELL),
            Paragraph(f"{delta:+.1f}",   S_TABLE_CELL),
            Paragraph(source[:20],        S_TABLE_CELL),
            Paragraph(statut_txt,         S_TABLE_CELL),
        ])

        # Colorier la cellule Statut
        style_cmds.append(
            ("TEXTCOLOR", (5, row_idx), (5, row_idx), statut_color)
        )
        style_cmds.append(
            ("FONTNAME",  (5, row_idx), (5, row_idx), "Helvetica-Bold")
        )
        # Colorier le delta si hors norme
        if not within_1sd:
            style_cmds.append(
                ("TEXTCOLOR", (3, row_idx), (3, row_idx), C_WARNING)
            )

    col_widths = [4.5 * cm, 2.3 * cm, 2.6 * cm, 2.0 * cm, 2.5 * cm, 2.2 * cm]
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle(style_cmds + [
        ("ALIGN", (0, 1), (0, -1), "LEFT"),
    ]))
    return t


def _build_alerts_table(alerts: List[Dict]) -> Table:
    """Tableau alertes cliniques avec code couleur warning / critical."""
    headers = [
        Paragraph("Articulation", S_TABLE_HEADER),
        Paragraph("Phase",        S_TABLE_HEADER),
        Paragraph("Mesuré (°)",   S_TABLE_HEADER),
        Paragraph("Seuil (°)",    S_TABLE_HEADER),
        Paragraph("Sévérité",     S_TABLE_HEADER),
        Paragraph("Référence",    S_TABLE_HEADER),
    ]
    rows = [headers]
    style_cmds = list(_table_style_base())

    for row_idx, alert in enumerate(alerts, start=1):
        severity  = alert.get("severity", "warning")
        sev_color = C_CRITICAL if severity == "critical" else C_WARNING
        sev_label = "⚠ Critique" if severity == "critical" else "! Warning"

        rows.append([
            Paragraph(_label_joint(alert.get("joint", "—")), S_TABLE_CELL_LEFT),
            Paragraph(_label_phase(alert.get("phase", "—")), S_TABLE_CELL),
            Paragraph(f"{alert.get('value', 0):.1f}",     S_TABLE_CELL),
            Paragraph(f"{alert.get('threshold', 0):.1f}", S_TABLE_CELL),
            Paragraph(sev_label,                           S_TABLE_CELL),
            Paragraph(alert.get("reference", "—")[:20],   S_TABLE_CELL),
        ])

        # Colorier la colonne Sévérité
        style_cmds.append(("TEXTCOLOR",  (4, row_idx), (4, row_idx), sev_color))
        style_cmds.append(("FONTNAME",   (4, row_idx), (4, row_idx), "Helvetica-Bold"))
        style_cmds.append(("BACKGROUND", (0, row_idx), (-1, row_idx),
                           colors.HexColor("#FEF9F0") if severity == "warning"
                           else colors.HexColor("#FFF1F1")))

    col_widths = [4.0 * cm, 3.0 * cm, 2.3 * cm, 2.3 * cm, 2.5 * cm, 2.0 * cm]
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle(style_cmds + [
        ("ALIGN", (0, 1), (0, -1), "LEFT"),
        ("ALIGN", (1, 1), (1, -1), "LEFT"),
    ]))
    return t


# ── Fonction principale ────────────────────────────────────────

def generate_pdf(
    session_id:          str,
    gesture_type:        str,
    created_at:          Optional[str],
    pipeline_mode:       str,
    total_frames:        int,
    athlete_name:        str,
    athlete_age:         Optional[int],
    athlete_hand:        str,
    phase_annotations:   Optional[Dict[str, int]],
    joint_metrics:       Dict[str, Any],
    normative_comparison: Dict[str, Any],
    alerts:              List[Dict],
) -> bytes:
    """
    Génère le rapport PDF complet et retourne les bytes.

    Paramètres extraits de Session + Athlete + Metrics en MongoDB
    et passés directement par l'endpoint FastAPI.
    """

    buffer = io.BytesIO()

    # ── Mise en page ───────────────────────────────────────────
    # Zone de contenu : entre le bandeau header (2.2cm) et le footer (1.5cm)
    content_top    = 2.6 * cm
    content_bottom = 1.5 * cm

    canvas_helper = _ReportCanvas(
        athlete_name  = athlete_name,
        session_date  = _fmt_date(created_at),
        gesture       = GESTURE_LABELS.get(gesture_type, gesture_type),
    )

    frame = Frame(
        MARGIN, content_bottom,
        PAGE_W - 2 * MARGIN,
        PAGE_H - content_top - content_bottom,
        id="main",
    )
    template = PageTemplate(
        id="main",
        frames=[frame],
        onPage=canvas_helper.on_page,
    )
    doc = BaseDocTemplate(
        buffer,
        pagesize=A4,
        pageTemplates=[template],
        title=f"Rapport — {athlete_name} — {GESTURE_LABELS.get(gesture_type, gesture_type)}",
        author="IA/Serve — Plateforme d'analyse biomécanique",
    )

    # ── Contenu ────────────────────────────────────────────────
    story = []

    # 1. Titre principal
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(
        f"Rapport d'analyse — {GESTURE_LABELS.get(gesture_type, gesture_type)}",
        S_TITLE,
    ))
    story.append(Paragraph(
        f"Athlète : {athlete_name}"
        + (f"  ·  {athlete_age} ans" if athlete_age else "")
        + f"  ·  Main dominante : {athlete_hand}"
        + f"  ·  Pipeline : {pipeline_mode}"
        + f"  ·  {total_frames} frames",
        S_SUBTITLE,
    ))
    story.append(Spacer(1, 0.3 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_INDIGO))
    story.append(Spacer(1, 0.4 * cm))

    # 2. Phases détectées
    phases_src = phase_annotations or {}
    if phases_src:
        story.append(Paragraph("Phases détectées", S_SECTION))
        story.append(_build_phases_table(phases_src))
        story.append(Spacer(1, 0.5 * cm))

    # 3. Métriques articulaires
    if joint_metrics:
        story.append(Paragraph("Métriques articulaires (session complète)", S_SECTION))
        story.append(Paragraph(
            "Statistiques calculées sur l'ensemble des frames de la session.",
            S_SMALL,
        ))
        story.append(Spacer(1, 0.2 * cm))
        story.append(_build_metrics_table(joint_metrics))
        story.append(Spacer(1, 0.5 * cm))

    # 4. Comparaison normative
    if normative_comparison:
        story.append(Paragraph("Comparaison normative — Gorce 2024 / Elliott 2008", S_SECTION))
        story.append(Paragraph(
            "Δ = angle mesuré − valeur normative de référence. "
            "Statut « Dans 1σ » indique que l'écart reste dans un écart-type de la norme.",
            S_SMALL,
        ))
        story.append(Spacer(1, 0.2 * cm))
        story.append(_build_normative_table(normative_comparison))
        story.append(Spacer(1, 0.5 * cm))

    # 5. Alertes cliniques (section conditionnelle)
    if alerts:
        story.append(Paragraph(
            f"Alertes cliniques  ({len(alerts)} détectée{'s' if len(alerts) > 1 else ''})",
            S_SECTION,
        ))
        story.append(Paragraph(
            "Déviations significatives identifiées par rapport aux seuils biomécaniques. "
            "À interpréter en contexte clinique par un spécialiste qualifié.",
            S_SMALL,
        ))
        story.append(Spacer(1, 0.2 * cm))
        story.append(_build_alerts_table(alerts))
        story.append(Spacer(1, 0.5 * cm))
    else:
        story.append(Paragraph("Alertes cliniques", S_SECTION))
        story.append(Paragraph(
            "Aucune alerte clinique détectée pour cette session.",
            S_BODY,
        ))
        story.append(Spacer(1, 0.5 * cm))

    # 6. Note de bas de rapport
    story.append(HRFlowable(width="100%", thickness=0.3, color=C_BORDER))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        f"Session ID : {session_id}  ·  "
        "Ce rapport est généré automatiquement par la plateforme IA/Serve. "
        "Les valeurs normatives proviennent de : Gorce & Jacquier-Bret (2024), "
        "Frontiers in Sports and Active Living, PMC11260724.",
        S_SMALL,
    ))

    # ── Compilation ────────────────────────────────────────────
    doc.build(story)
    buffer.seek(0)
    return buffer.read()


# ── Utilitaire ─────────────────────────────────────────────────

def _fmt_date(iso_str: Optional[str]) -> str:
    """Formate une date ISO en dd/mm/yyyy."""
    if not iso_str:
        return "—"
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%d/%m/%Y")
    except Exception:
        return iso_str[:10]