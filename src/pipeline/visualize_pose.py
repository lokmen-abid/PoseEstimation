"""
=============================================================================
IA/Serve — visualize_pose.py
Visualisation OpenCV du squelette MediaPipe + angles articulaires
sur une vidéo analysée par le pipeline Ex3.

Affiche en temps réel :
  - Squelette 2D (17 keypoints + connexions) aligné sur le corps
  - Panneau angles — coin bas-gauche (ne couvre pas la joueuse)
  - Label phase — bandeau fin en haut
  - Légende — coin bas-droit compact
  - Barre de progression en bas

Contrôles :
  ESPACE  : pause / reprise
  →       : frame suivante (en pause)
  ←       : frame précédente (en pause)
  Q / ESC : quitter
  S       : screenshot

Usage :
  python src/pipeline/visualize_pose.py
      --video  output/tmp_video/back_side.mp4
      --frames output/results/frames_angles_back_side_sk3.json
      --phases "trophy_position=304,racket_low_point=224,ball_impact=169"
      --fps 10
=============================================================================
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
WINDOW_NAME  = "IA/Serve — Pose Estimation"
PLAYBACK_FPS = 15
POINT_RADIUS = 7
LINE_THICK   = 2
FONT         = cv2.FONT_HERSHEY_SIMPLEX
PHASE_WINDOW = 15

# Couleurs BGR
COL_POINT    = (0,   220, 160)   # teal — keypoint normal
COL_POINT_HL = (0,   200, 255)   # jaune — keypoint clé
COL_LINE     = (220, 220, 220)   # blanc cassé — connexion
COL_OK       = (60,  210,  60)   # vert — dans normative
COL_ERR      = (50,   50, 220)   # rouge — hors normative
COL_NA       = (130, 130, 130)   # gris — N/A
COL_PHASE    = (255, 200,   0)   # jaune vif — label phase
COL_WHITE    = (255, 255, 255)
COL_DARK     = (15,   15,  15)
COL_PANEL    = (20,   20,  20)   # fond panneau

# Connexions squelette (paires d'indices MediaPipe)
SKELETON_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),  # tronc
    (11, 13), (13, 15),                        # bras gauche
    (12, 14), (14, 16),                        # bras droit
    (23, 25), (25, 27), (27, 29), (27, 31),   # jambe gauche
    (24, 26), (26, 28), (28, 30), (28, 32),   # jambe droite
]

KEY_LANDMARKS = {11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28}

NORMATIVE = {
    "knee_flexion_right":       {"mean": 64.5,  "std": 9.7},
    "trunk_inclination":        {"mean": 25.0,  "std": 7.1},
    "shoulder_rotation_right":  {"mean": 130.1, "std": 26.5},
    "shoulder_elevation_right": {"mean": 110.7, "std": 16.9},
    "elbow_right":              {"mean": 30.1,  "std": 15.9},
}

KEYPOINT_NAME_TO_IDX = {
    "left_shoulder": 11,  "right_shoulder": 12,
    "left_elbow":    13,  "right_elbow":    14,
    "left_wrist":    15,  "right_wrist":    16,
    "left_hip":      23,  "right_hip":      24,
    "left_knee":     25,  "right_knee":     26,
    "left_ankle":    27,  "right_ankle":    28,
    "left_heel":     29,  "right_heel":     30,
    "left_foot_index": 31, "right_foot_index": 32,
}

# ─────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────

def load_frames(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_lookup(frames: List[Dict]) -> Dict[int, Dict]:
    return {f["frame_number"]: f for f in frames}

def parse_phases(phases_str: Optional[str]) -> Dict[str, int]:
    if not phases_str:
        return {}
    result = {}
    for item in phases_str.split(","):
        if "=" not in item:
            continue
        name, val = item.strip().split("=", 1)
        try:
            result[name.strip()] = int(val.strip())
        except ValueError:
            pass
    return result

def get_phase(frame_number: int, phases: Dict[str, int]) -> Optional[str]:
    for phase, target in phases.items():
        if abs(frame_number - target) <= PHASE_WINDOW:
            return phase
    return None

# ─────────────────────────────────────────────
# PROJECTION IMAGE
# ─────────────────────────────────────────────

def project_landmarks(image_kps: Dict, w: int, h: int) -> Dict[int, Tuple[int, int]]:
    """Convertit les coordonnées normalisées (0→1) en pixels réels."""
    positions = {}
    for name, idx in KEYPOINT_NAME_TO_IDX.items():
        kp = image_kps.get(name)
        if kp is not None:
            positions[idx] = (int(kp["x"] * w), int(kp["y"] * h))
    return positions

# ─────────────────────────────────────────────
# DESSIN SQUELETTE
# ─────────────────────────────────────────────

def draw_skeleton(frame: np.ndarray,
                  positions: Dict[int, Tuple[int, int]]) -> np.ndarray:
    # Connexions
    for (i, j) in SKELETON_CONNECTIONS:
        if i in positions and j in positions:
            cv2.line(frame, positions[i], positions[j],
                     COL_LINE, LINE_THICK, cv2.LINE_AA)
    # Points
    for idx, (px, py) in positions.items():
        color = COL_POINT_HL if idx in KEY_LANDMARKS else COL_POINT
        cv2.circle(frame, (px, py), POINT_RADIUS, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), POINT_RADIUS + 1, COL_DARK, 1, cv2.LINE_AA)
    return frame

# ─────────────────────────────────────────────
# PANNEAU ANGLES — BAS GAUCHE
# ─────────────────────────────────────────────

def draw_angles_panel(frame: np.ndarray, angles: Dict,
                      phase: Optional[str]) -> np.ndarray:
    """
    Panneau compact en bas-à-gauche.
    Ne couvre pas la zone centrale où se trouve le joueur.
    """
    h, w = frame.shape[:2]

    PANEL_W = 230
    PANEL_H = 175
    MARGIN  = 8
    px      = MARGIN
    py      = h - PANEL_H - 30   # au-dessus de la barre de progression

    # Fond semi-transparent
    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + PANEL_W, py + PANEL_H),
                  COL_PANEL, -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    # Bordure fine
    cv2.rectangle(frame, (px, py), (px + PANEL_W, py + PANEL_H),
                  (60, 60, 60), 1)

    # Titre
    title = phase.replace("_", " ").upper() if phase else "ANGLES"
    cv2.putText(frame, title, (px + 8, py + 18),
                FONT, 0.45, COL_PHASE, 1, cv2.LINE_AA)

    # Séparateur
    cv2.line(frame, (px + 4, py + 24), (px + PANEL_W - 4, py + 24),
             (60, 60, 60), 1)

    # Angles
    rows = [
        ("Genou D",    "knee_flexion_right",       "64.5±9.7"),
        ("Tronc incl", "trunk_inclination",         "25.0±7.1"),
        ("Ep.D rot",   "shoulder_rotation_right",   "130.1±26.5"),
        ("Ep.D elv",   "shoulder_elevation_right",  "110.7±16.9"),
        ("Coude D",    "elbow_right",               "30.1±15.9"),
    ]

    for i, (label, key, norm_str) in enumerate(rows):
        val  = angles.get(key)
        row_y = py + 42 + i * 27

        norm = NORMATIVE.get(key)
        if val is not None and norm:
            within  = abs(val - norm["mean"]) <= norm["std"]
            col_val = COL_OK if within else COL_ERR
            val_str = f"{val:6.1f}"
            status  = "OK" if within else "!!"
        elif val is not None:
            col_val = COL_WHITE
            val_str = f"{val:6.1f}"
            status  = ""
        else:
            col_val = COL_NA
            val_str = "  N/A"
            status  = ""

        # Label
        cv2.putText(frame, f"{label}:", (px + 8, row_y),
                    FONT, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
        # Valeur
        cv2.putText(frame, f"{val_str}", (px + 110, row_y),
                    FONT, 0.42, col_val, 1, cv2.LINE_AA)
        # Statut
        if status:
            cv2.putText(frame, status, (px + 170, row_y),
                        FONT, 0.38, col_val, 1, cv2.LINE_AA)
        # Normative en petit
        cv2.putText(frame, f"[{norm_str}]", (px + 8, row_y + 11),
                    FONT, 0.28, (80, 80, 80), 1, cv2.LINE_AA)

    return frame

# ─────────────────────────────────────────────
# HUD — BARRE EN HAUT + INFOS BAS
# ─────────────────────────────────────────────

def draw_hud(frame: np.ndarray, frame_number: int, total_frames: int,
             phase: Optional[str], paused: bool) -> np.ndarray:
    h, w = frame.shape[:2]

    # ── Bandeau titre fin en haut ──
    cv2.rectangle(frame, (0, 0), (w, 28), (10, 10, 10), -1)
    cv2.putText(frame, "IA/Serve  |  Pose Estimation  |  Ex3 SavGol",
                (8, 19), FONT, 0.5, COL_WHITE, 1, cv2.LINE_AA)
    if paused:
        cv2.putText(frame, "[ PAUSE ]", (w - 100, 19),
                    FONT, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

    # ── Label phase — bandeau fin sous le titre ──
    if phase:
        labels = {
            "trophy_position":  "TROPHY POSITION",
            "racket_low_point": "RACKET LOW POINT",
            "ball_impact":      "BALL IMPACT",
        }
        label = labels.get(phase, phase.upper())
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.65, 2)
        tx = (w - tw) // 2
        cv2.rectangle(frame, (tx - 10, 30), (tx + tw + 10, 58),
                      (15, 80, 40), -1)
        cv2.rectangle(frame, (tx - 10, 30), (tx + tw + 10, 58),
                      (30, 160, 80), 1)
        cv2.putText(frame, label, (tx, 52),
                    FONT, 0.65, COL_PHASE, 2, cv2.LINE_AA)

    # ── Frame counter — bas droit ──
    counter = f"Frame {frame_number} / {total_frames}"
    cv2.putText(frame, counter, (w - 200, h - 34),
                FONT, 0.42, (160, 160, 160), 1, cv2.LINE_AA)

    # ── Légende compacte — bas droit ──
    lx = w - 230
    ly = h - 28
    cv2.circle(frame, (lx, ly - 4), 5, COL_OK, -1)
    cv2.putText(frame, "Dans normative Gorce 2024",
                (lx + 10, ly), FONT, 0.35, COL_OK, 1, cv2.LINE_AA)
    cv2.circle(frame, (lx, ly + 12), 5, COL_ERR, -1)
    cv2.putText(frame, "Hors normative",
                (lx + 10, ly + 16), FONT, 0.35, COL_ERR, 1, cv2.LINE_AA)

    # ── Barre de progression ──
    bar_y = h - 8
    cv2.rectangle(frame, (0, bar_y - 5), (w, bar_y), (50, 50, 50), -1)
    if total_frames > 0:
        prog = int(w * frame_number / total_frames)
        cv2.rectangle(frame, (0, bar_y - 5), (prog, bar_y),
                      (0, 180, 100), -1)

    # ── Contrôles discrets ──
    cv2.putText(frame, "ESPACE:pause  S:screenshot  Q:quitter",
                (8, h - 12), FONT, 0.32, (90, 90, 90), 1, cv2.LINE_AA)

    return frame

# ─────────────────────────────────────────────
# PIPELINE DE VISUALISATION
# ─────────────────────────────────────────────

def run_visualization(video_path: str, frames_data: List[Dict],
                      phases: Dict[str, int], screenshot_dir: str):

    lookup       = build_lookup(frames_data)
    total_frames = 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Erreur] Impossible d'ouvrir : {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_native   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay_ms     = max(1, int(1000 / PLAYBACK_FPS))

    print(f"[Viz] {total_frames} frames @ {fps_native:.1f} FPS")
    print(f"[Viz] {len(frames_data)} frames analysées disponibles")
    print(f"[Viz] Phases : {phases}")
    print("[Viz] ESPACE=pause | →/← en pause | S=screenshot | Q=quitter")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 600)

    paused       = False
    frame_number = 0
    screenshots  = 0

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_number = 0
                continue
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                break

        # ── Données de la frame ──
        fdata     = lookup.get(frame_number, {})
        angles    = fdata.get("angles", {})
        image_kps = fdata.get("image_kps", {})
        phase     = get_phase(frame_number, phases)

        h, w = frame.shape[:2]

        # ── Squelette ──
        if image_kps:
            positions = project_landmarks(image_kps, w, h)
            if positions:
                frame = draw_skeleton(frame, positions)

        # ── HUD ──
        frame = draw_angles_panel(frame, angles, phase)
        frame = draw_hud(frame, frame_number, total_frames, phase, paused)

        cv2.imshow(WINDOW_NAME, frame)

        # ── Touches ──
        key = cv2.waitKey(delay_ms if not paused else 30) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('s'):
            os.makedirs(screenshot_dir, exist_ok=True)
            path = os.path.join(screenshot_dir, f"screenshot_{frame_number:05d}.png")
            cv2.imwrite(path, frame)
            screenshots += 1
            print(f"[Screenshot] {path}")
        elif key == 83 and paused:   # →
            frame_number = min(frame_number + 1, total_frames - 1)
        elif key == 81 and paused:   # ←
            frame_number = max(frame_number - 1, 0)

        if not paused:
            frame_number += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"[Viz] Terminé — {screenshots} screenshot(s)")

# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="IA/Serve — Visualisation OpenCV squelette + angles"
    )
    p.add_argument("--video",   required=True)
    p.add_argument("--frames",  required=True)
    p.add_argument("--phases",  default=None)
    p.add_argument("--fps",     type=int, default=15)
    p.add_argument("--screenshots", default="output/screenshots")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    PLAYBACK_FPS = args.fps

    if not os.path.exists(args.video):
        sys.exit(f"[Erreur] Vidéo introuvable : {args.video}")
    if not os.path.exists(args.frames):
        sys.exit(f"[Erreur] JSON introuvable : {args.frames}")

    print("[Viz] Chargement des frames JSON...")
    frames_data = load_frames(args.frames)
    print(f"[Viz] {len(frames_data)} frames chargées")

    phases = parse_phases(args.phases)
    run_visualization(args.video, frames_data, phases, args.screenshots)