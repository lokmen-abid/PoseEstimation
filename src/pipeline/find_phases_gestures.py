"""
=============================================================================
IA/Serve — find_phases_gestures.py
Détection des frames clés pour tous les gestes tennis
Version 3 — sans zones temporelles fixes

Principe fondamental :
  Les zones temporelles fixes (ex: RLP = 35–70%) échouent sur les vidéos
  dont le cadrage ne couvre pas tout le service depuis le début.
  Cette version détecte d'abord l'ancre biomécanique de chaque geste
  (le pic signal le plus fiable), puis dérive les autres phases
  par contrainte temporelle relative (avant/après l'ancre).

Ancres par geste :
  serve     : RLP = pic max de shoulder_rotation_right sur toute la vidéo
  forehand  : ball_impact = pic max de elbow_right (extension)
  backhand  : preparation = pic max de trunk_rotation (rotation arrière)

Puis contraintes :
  serve     : trophy_position AVANT le RLP (≥0.5s), ball_impact APRÈS (≥0.3s)
  forehand  : preparation AVANT ball_impact, acceleration ENTRE les deux
  backhand  : ball_impact APRÈS preparation, follow_through APRÈS ball_impact

Usage :
  python src/pipeline/find_phases_gestures.py \
    --frames output/results/frames_angles_{id}.json \
    --gesture serve|forehand|backhand \
    [--variant 1h|2h]  (backhand uniquement)
    [--detail]
=============================================================================
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from normatives import (
    NORMATIVE_STD_BY_JOINT,
    get_normatives,
)

# ─────────────────────────────────────────────
# SÉPARATIONS TEMPORELLES MINIMALES (secondes)
# ─────────────────────────────────────────────

MIN_SECONDS_BETWEEN: Dict[str, Dict[Tuple[str, str], float]] = {
    "serve": {
        ("trophy_position",  "racket_low_point"): 0.40,
        ("racket_low_point", "ball_impact"):       0.25,
    },
    "forehand": {
        ("preparation",  "acceleration"): 0.25,
        ("acceleration", "ball_impact"):  0.20,
    },
    "backhand": {
        ("preparation", "ball_impact"):    0.25,
        ("ball_impact", "follow_through"): 0.20,
    },
}

# ─────────────────────────────────────────────
# ANCRES BIOMÉCANIQUES PAR GESTE
#
# L'ancre est le signal le plus fiable et le plus caractéristique
# du geste, indépendamment du cadrage de la vidéo.
# ─────────────────────────────────────────────

ANCHOR_JOINT: Dict[str, str] = {
    # serve : rotation latérale épaule — maximale au RLP (Gorce 2024)
    "serve":    "shoulder_rotation_right",
    # forehand : extension du coude — maximale au ball_impact (Knudson 2001)
    "forehand": "elbow_right",
    # backhand : rotation du tronc — maximale en préparation (Elliott 2008)
    "backhand": "trunk_rotation",
}

ANCHOR_PHASE: Dict[str, str] = {
    "serve":    "racket_low_point",
    "forehand": "ball_impact",
    "backhand": "preparation",
}

ANCHOR_LOOK: Dict[str, str] = {
    "serve":    "max",   # shoulder_rotation_right est maximal au RLP
    "forehand": "max",   # elbow_right est maximal (bras le plus étendu) à l'impact
    "backhand": "max",   # trunk_rotation est maximal en préparation
}

# ─────────────────────────────────────────────
# HYPERPARAMÈTRES
# ─────────────────────────────────────────────

LOCAL_EXTREMUM_WINDOW_SECONDS = 0.40   # ±12 frames à 30fps
LOCAL_EXTREMUM_PENALTY        = 0.3
QUALITY_STD_THRESHOLD         = 2.0
FPS_REFERENCE                 = 30.0

# Camera guard : si le max d'un joint n'atteint pas ce seuil,
# l'axe de mesure est comprimé par l'angle de caméra.
CAMERA_GUARD_PENALTY = 0.1
CAMERA_GUARD_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "serve":    {"shoulder_rotation_right": 90.0},
    "forehand": {"trunk_rotation": 45.0},
    "backhand": {"trunk_rotation": 45.0},
}

# ─────────────────────────────────────────────
# AFFICHAGE
# ─────────────────────────────────────────────

CONFIDENCE_SYMBOL = {"HIGH": "★★★★", "MEDIUM": "★★★☆",
                     "LOW": "★★☆☆", "UNRELIABLE": "★☆☆☆"}
CONFIDENCE_COLOR  = {"HIGH": "✅", "MEDIUM": "🟡",
                     "LOW": "🟠", "UNRELIABLE": "❌"}

BACKHAND_2H_ELBOW_LEFT_RATIO     = 0.60
BACKHAND_2H_ELBOW_DIFF_THRESHOLD = 40.0


# ─────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────

def load_frames(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────
# FPS
# ─────────────────────────────────────────────

def detect_fps(frames: List[Dict]) -> float:
    timestamps = [f.get("timestamp_ms") for f in frames
                  if f.get("timestamp_ms") is not None]
    if len(timestamps) < 2:
        return FPS_REFERENCE
    deltas = [timestamps[i+1] - timestamps[i]
              for i in range(len(timestamps)-1)
              if timestamps[i+1] > timestamps[i]]
    if not deltas:
        return FPS_REFERENCE
    return round(1000.0 / float(np.median(deltas)), 1)


def seconds_to_frames(seconds: float, fps: float) -> int:
    return max(3, int(round(seconds * fps)))


# ─────────────────────────────────────────────
# VARIANTE BACKHAND
# ─────────────────────────────────────────────

def detect_backhand_variant(frames: List[Dict]) -> str:
    n     = len(frames)
    start = int(n * 0.40)
    end   = int(n * 0.75)
    zone  = frames[start:end]
    if not zone:
        return "1h"

    elbow_left_present = sum(
        1 for f in zone if f["angles"].get("elbow_left") is not None
    )
    presence_ratio = elbow_left_present / len(zone)

    diffs = []
    for f in zone:
        el = f["angles"].get("elbow_left")
        er = f["angles"].get("elbow_right")
        if el is not None and er is not None:
            diffs.append(abs(el - er))
    median_diff = float(np.median(diffs)) if diffs else 999.0

    is_2h_presence = presence_ratio >= BACKHAND_2H_ELBOW_LEFT_RATIO
    is_2h_symmetry = median_diff < BACKHAND_2H_ELBOW_DIFF_THRESHOLD

    print(f"[Variante] Présence elbow_left : {presence_ratio:.0%} → "
          f"{'2H' if is_2h_presence else '1H'}")
    print(f"[Variante] Différence médiane coudes : {median_diff:.1f}° → "
          f"{'2H' if is_2h_symmetry else '1H'}")

    if is_2h_presence and is_2h_symmetry:
        variant = "2h"
    elif is_2h_presence or is_2h_symmetry:
        print("[Variante] ⚠ Critères contradictoires — forcer avec --variant")
        variant = "2h" if is_2h_presence else "1h"
    else:
        variant = "1h"

    print(f"[Variante] Décision → Backhand {variant.upper()}")
    return variant


# ─────────────────────────────────────────────
# CAMERA GUARD
# ─────────────────────────────────────────────

def check_camera_angle_reliability(
    frames: List[Dict],
    gesture: str,
) -> Dict[str, float]:
    """
    Détecte les joints dont le signal est comprimé par l'angle de caméra.
    Retourne un multiplicateur par joint : 1.0 = fiable, 0.1 = comprimé.
    """
    reliability: Dict[str, float] = {}
    thresholds = CAMERA_GUARD_THRESHOLDS.get(gesture, {})

    for joint, threshold in thresholds.items():
        values = [f["angles"].get(joint) for f in frames
                  if f["angles"].get(joint) is not None]
        if not values:
            reliability[joint] = 1.0
            continue
        max_val = float(np.nanmax(values))
        if max_val < threshold:
            reliability[joint] = CAMERA_GUARD_PENALTY
            print(f"[CameraGuard] {joint} max={max_val:.1f}° < {threshold}° "
                  f"→ comprimé ({gesture}), poids ×{CAMERA_GUARD_PENALTY}")
        else:
            reliability[joint] = 1.0

    return reliability


# ─────────────────────────────────────────────
# UTILITAIRES SIGNAL
# ─────────────────────────────────────────────

def get_joint_series(frames: List[Dict], joint: str) -> np.ndarray:
    return np.array([
        f["angles"].get(joint) if f["angles"].get(joint) is not None else np.nan
        for f in frames
    ])


def is_local_extremum(values: np.ndarray, idx: int,
                      window: int = 12, kind: str = "max") -> bool:
    if np.isnan(values[idx]):
        return False
    start = max(0, idx - window)
    end   = min(len(values), idx + window + 1)
    valid = values[start:end][~np.isnan(values[start:end])]
    if len(valid) < 3:
        return True
    if kind == "max":
        return float(values[idx]) >= float(np.max(valid))
    return float(values[idx]) <= float(np.min(valid))


def compute_signal_quality(values: np.ndarray, target: float,
                           joint_name: str) -> float:
    std   = NORMATIVE_STD_BY_JOINT.get(joint_name, 20.0)
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return 0.0
    best = float(np.min(np.abs(valid - target)))
    return 0.2 if best > QUALITY_STD_THRESHOLD * std else 1.0


def compute_confidence(score: float, max_possible: float) -> str:
    if max_possible <= 0:
        return "UNRELIABLE"
    r = score / max_possible
    if r >= 0.75:   return "HIGH"
    elif r >= 0.50: return "MEDIUM"
    elif r >= 0.25: return "LOW"
    return "UNRELIABLE"


# ─────────────────────────────────────────────
# DÉTECTION DE L'ANCRE
#
# Trouve le top-5 candidats pour la phase ancre
# en cherchant le pic (max ou min) du joint ancre
# sur toute la vidéo, sans zone fixe.
# ─────────────────────────────────────────────

def find_anchor_candidates(
    frames: List[Dict],
    gesture: str,
    fps: float,
    camera_reliability: Dict[str, float],
) -> List[Tuple[int, float, str]]:
    """
    Trouve les top-5 candidats pour la phase ancre du geste.

    Stratégie :
      1. Extraire le signal du joint ancre sur toute la vidéo.
      2. Scorer chaque frame par proximité au pic global.
      3. Pénaliser les frames qui ne sont pas un extremum local.
      4. Retourner le top-5 trié par score décroissant.
    """
    joint   = ANCHOR_JOINT[gesture]
    look    = ANCHOR_LOOK[gesture]
    lw      = seconds_to_frames(LOCAL_EXTREMUM_WINDOW_SECONDS, fps)
    values  = get_joint_series(frames, joint)
    valid   = ~np.isnan(values)

    if not valid.any():
        return []

    # Score de proximité au pic global
    if look == "max":
        ref_val   = float(np.nanmax(values))
        proximity = 1.0 / (1.0 + np.abs(values - ref_val))
    else:
        ref_val   = float(np.nanmin(values))
        proximity = 1.0 / (1.0 + np.abs(values - ref_val))

    proximity[~valid] = 0.0

    # Pénalité extremum local
    for i in range(len(values)):
        if valid[i] and not is_local_extremum(values, i, window=lw, kind=look):
            proximity[i] *= LOCAL_EXTREMUM_PENALTY

    # Pondération par camera_reliability
    cam_mult = camera_reliability.get(joint, 1.0)
    scores   = proximity * cam_mult

    top = np.argsort(scores)[::-1][:5]
    max_score = float(scores[top[0]]) if len(top) > 0 else 0.0

    result = []
    for rank, i in enumerate(top):
        if scores[i] > 0:
            conf = compute_confidence(float(scores[i]), max_score)
            result.append((int(frames[i]["frame_number"]), float(scores[i]), conf))

    return result


# ─────────────────────────────────────────────
# DÉTECTION DES PHASES SECONDAIRES
#
# Cherche les phases non-ancres dans une fenêtre
# RELATIVE à l'ancre (avant ou après, en secondes).
# Pas de zone fixe en pourcentage de la vidéo.
# ─────────────────────────────────────────────

def find_secondary_candidates(
    frames: List[Dict],
    gesture: str,
    phase_name: str,
    normatives: Dict,
    anchor_frame: int,
    fps: float,
    camera_reliability: Dict[str, float],
    position: str,   # "before" | "after"
    min_gap_s: float,
    max_gap_s: float = 5.0,
) -> List[Tuple[int, float, str]]:
    """
    Trouve les top-5 candidats pour une phase secondaire.

    La fenêtre de recherche est définie par rapport à l'ancre :
      - position="before" : [anchor - max_gap_s, anchor - min_gap_s]
      - position="after"  : [anchor + min_gap_s, anchor + max_gap_s]

    max_gap_s = 5.0s par défaut : couvre les vidéos les plus longues.
    """
    n       = len(frames)
    min_gap = seconds_to_frames(min_gap_s, fps)
    max_gap = seconds_to_frames(max_gap_s, fps)

    # Trouver l'index de la frame ancre dans la liste
    anchor_idx = next(
        (i for i, f in enumerate(frames) if f["frame_number"] == anchor_frame),
        None
    )
    if anchor_idx is None:
        return []

    if position == "before":
        idx_start = max(0, anchor_idx - max_gap)
        idx_end   = max(0, anchor_idx - min_gap)
    else:  # after
        idx_start = min(n, anchor_idx + min_gap)
        idx_end   = min(n, anchor_idx + max_gap)

    if idx_end - idx_start < 3:
        # Fenêtre trop petite : élargir max_gap
        if position == "before":
            idx_start = max(0, anchor_idx - int(n * 0.6))
            idx_end   = max(0, anchor_idx - min_gap)
        else:
            idx_start = min(n, anchor_idx + min_gap)
            idx_end   = min(n, anchor_idx + int(n * 0.6))

    if idx_end <= idx_start:
        return []

    frames_zone  = frames[idx_start:idx_end]
    zone_size    = len(frames_zone)
    scores       = np.zeros(zone_size)
    max_possible = 0.0
    lw           = seconds_to_frames(LOCAL_EXTREMUM_WINDOW_SECONDS, fps)

    joints = normatives.get(phase_name, {})
    for joint_name, cfg in joints.items():
        target = cfg["target"]
        look   = cfg["look"]
        weight = cfg["weight"]

        values_zone = get_joint_series(frames_zone, joint_name)
        valid = ~np.isnan(values_zone)
        if not valid.any():
            continue

        values_full = get_joint_series(frames, joint_name)
        quality      = compute_signal_quality(values_full, target, joint_name)
        cam_mult     = camera_reliability.get(joint_name, 1.0)
        effective_w  = weight * quality * cam_mult
        max_possible += weight * quality

        if look == "min":
            ref = np.nanmin(values_zone)
            proximity = 1.0 / (1.0 + np.abs(values_zone - ref))
        elif look == "max":
            ref = np.nanmax(values_zone)
            proximity = 1.0 / (1.0 + np.abs(values_zone - ref))
        else:
            proximity = 1.0 / (1.0 + np.abs(values_zone - target))

        proximity[~valid] = 0.0

        if look in ("min", "max"):
            kind = "min" if look == "min" else "max"
            for i in range(zone_size):
                if not np.isnan(values_zone[i]):
                    if not is_local_extremum(values_zone, i, window=lw, kind=kind):
                        proximity[i] *= LOCAL_EXTREMUM_PENALTY

        scores += effective_w * proximity

    if max_possible <= 0:
        return []

    top = np.argsort(scores)[::-1][:5]
    best_score = float(scores[top[0]]) if len(top) > 0 else 0.0
    best_conf  = compute_confidence(best_score, max_possible)

    return [
        (
            int(frames_zone[i]["frame_number"]),
            float(scores[i]),
            best_conf if rank == 0
            else compute_confidence(float(scores[i]), max_possible),
        )
        for rank, i in enumerate(top)
        if scores[i] > 0
    ]


# ─────────────────────────────────────────────
# FILTRE POST-SCORING TROPHY
#
# Après le scoring par normatives, filtrer les candidats
# Trophy dont shoulder_elevation est hors de la plage
# attendue : bras en chemin vers le haut, pas encore au max.
# Gorce 2024 : Trophy → shoulder_elevation ≈ 80° (±30°)
# Plage retenue : 50°–140° (permissive pour couvrir les angles de caméra)
# En dessous de 50° : bras encore bas (début de service)
# Au dessus de 140° : déjà au niveau Ball Impact
# ─────────────────────────────────────────────

TROPHY_ELEV_MIN = 50.0   # degrés
TROPHY_ELEV_MAX = 140.0  # degrés


def filter_trophy_by_elevation(
    candidates: List[Tuple[int, float, str]],
    frames: List[Dict],
) -> List[Tuple[int, float, str]]:
    """
    Retire les candidats Trophy dont shoulder_elevation_right
    est hors plage [TROPHY_ELEV_MIN, TROPHY_ELEV_MAX].
    Si tous sont hors plage, retourne la liste originale
    (pas de candidates du tout serait pire).
    """
    frame_lookup = {f["frame_number"]: f for f in frames}
    filtered = []
    for fn, score, conf in candidates:
        fd = frame_lookup.get(fn)
        if fd is None:
            filtered.append((fn, score, conf))
            continue
        elev = fd["angles"].get("shoulder_elevation_right")
        if elev is None:
            # Pas de valeur → on garde (pas de raison de rejeter)
            filtered.append((fn, score, conf))
        elif TROPHY_ELEV_MIN <= elev <= TROPHY_ELEV_MAX:
            filtered.append((fn, score, conf))
        else:
            print(f"  [TrophyFilter] Frame {fn} rejetée : "
                  f"shoulder_elevation={elev:.1f}° hors [{TROPHY_ELEV_MIN},{TROPHY_ELEV_MAX}]°")

    if not filtered:
        print("  [TrophyFilter] Tous les candidats filtrés — retour liste originale")
        return candidates
    return filtered


# ─────────────────────────────────────────────
# ORCHESTRATION PRINCIPALE
# ─────────────────────────────────────────────

def find_all_candidates(
    frames: List[Dict],
    gesture: str,
    normatives: Dict,
    fps: float,
    camera_reliability: Dict[str, float],
) -> Dict[str, List[Tuple[int, float, str]]]:
    """
    Détecte les candidats pour toutes les phases du geste.

    Étape 1 : trouver l'ancre (phase biomécanique la plus fiable).
    Étape 2 : dériver les autres phases par contrainte temporelle
              relative à l'ancre (avant/après, en secondes).

    Aucune zone fixe en % de la vidéo n'est utilisée.
    """
    min_sep = MIN_SECONDS_BETWEEN[gesture]
    candidates: Dict[str, List] = {}

    # ── Étape 1 : ancre ──────────────────────────────────────
    anchor_phase    = ANCHOR_PHASE[gesture]
    anchor_cands    = find_anchor_candidates(frames, gesture, fps, camera_reliability)
    candidates[anchor_phase] = anchor_cands

    if not anchor_cands:
        print(f"[Candidats] Aucun candidat pour l'ancre {anchor_phase} — abandon")
        return candidates

    # Meilleur candidat ancre pour dériver les autres
    anchor_frame = anchor_cands[0][0]
    print(f"[Candidats] Ancre {anchor_phase} → frame {anchor_frame} "
          f"(score {anchor_cands[0][1]:.3f})")

    # ── Étape 2 : phases secondaires ─────────────────────────
    if gesture == "serve":
        # Trophy = AVANT le RLP
        # max_gap_s=2.5s : la Trophy précède le RLP d'environ 0.4–1.5s en pratique.
        # On borne à 2.5s pour exclure les frames du début de la vidéo
        # où le joueur est encore debout (genou tendu, épaule basse) et qui
        # scorent faux positif sur knee_flexion (look=min trouve le minimum
        # de la vidéo entière si la fenêtre est trop large).
        min_gap = min_sep.get(("trophy_position", "racket_low_point"), 0.40)
        trophy_raw = find_secondary_candidates(
            frames, gesture, "trophy_position", normatives,
            anchor_frame, fps, camera_reliability,
            position="before", min_gap_s=min_gap, max_gap_s=2.5,
        )
        candidates["trophy_position"] = filter_trophy_by_elevation(trophy_raw, frames)
        # Ball Impact = APRÈS le RLP
        # max_gap_s=1.5s : le BI suit le RLP d'environ 0.25–0.8s.
        min_gap = min_sep.get(("racket_low_point", "ball_impact"), 0.25)
        candidates["ball_impact"] = find_secondary_candidates(
            frames, gesture, "ball_impact", normatives,
            anchor_frame, fps, camera_reliability,
            position="after", min_gap_s=min_gap, max_gap_s=1.5,
        )

    elif gesture == "forehand":
        # ball_impact est l'ancre
        # preparation = AVANT ball_impact — max_gap_s=3.0s
        total_before = (min_sep.get(("preparation", "acceleration"), 0.25)
                        + min_sep.get(("acceleration", "ball_impact"), 0.20))
        candidates["preparation"] = find_secondary_candidates(
            frames, gesture, "preparation", normatives,
            anchor_frame, fps, camera_reliability,
            position="before", min_gap_s=total_before, max_gap_s=3.0,
        )
        # acceleration = AVANT ball_impact — max_gap_s=1.5s
        min_gap_accel = min_sep.get(("acceleration", "ball_impact"), 0.20)
        candidates["acceleration"] = find_secondary_candidates(
            frames, gesture, "acceleration", normatives,
            anchor_frame, fps, camera_reliability,
            position="before", min_gap_s=min_gap_accel, max_gap_s=1.5,
        )

    elif gesture == "backhand":
        # preparation est l'ancre (trunk_rotation max)
        # ball_impact = APRÈS preparation — max_gap_s=2.0s
        min_gap = min_sep.get(("preparation", "ball_impact"), 0.25)
        candidates["ball_impact"] = find_secondary_candidates(
            frames, gesture, "ball_impact", normatives,
            anchor_frame, fps, camera_reliability,
            position="after", min_gap_s=min_gap, max_gap_s=2.0,
        )
        # follow_through = APRÈS ball_impact — max_gap_s=2.0s
        bi_cands = candidates.get("ball_impact", [])
        ref_frame = bi_cands[0][0] if bi_cands else anchor_frame
        min_gap_ft = min_sep.get(("ball_impact", "follow_through"), 0.20)
        candidates["follow_through"] = find_secondary_candidates(
            frames, gesture, "follow_through", normatives,
            ref_frame, fps, camera_reliability,
            position="after", min_gap_s=min_gap_ft, max_gap_s=2.0,
        )

    return candidates


# ─────────────────────────────────────────────
# COHÉRENCE TEMPORELLE
# ─────────────────────────────────────────────

def validate_temporal_coherence(
    phase_frames: Dict[str, int],
    phase_order: List[str],
    min_between: Dict[Tuple[str, str], int],
) -> Tuple[bool, str]:
    for i in range(len(phase_order) - 1):
        a, b = phase_order[i], phase_order[i+1]
        fa, fb = phase_frames.get(a), phase_frames.get(b)
        if fa is None or fb is None:
            return False, f"Phase manquante : {a} ou {b}"
        if fa >= fb:
            return False, f"Ordre incorrect : {a}({fa}) >= {b}({fb})"
        min_sep = min_between.get((a, b), 0)
        if fb - fa < min_sep:
            return False, (f"{a}→{b} trop proches "
                           f"({fb-fa} frames, min {min_sep})")
    return True, "OK"


# ─────────────────────────────────────────────
# SÉLECTION OPTIMALE
# ─────────────────────────────────────────────

PHASE_ORDER: Dict[str, List[str]] = {
    "serve":    ["trophy_position", "racket_low_point", "ball_impact"],
    "forehand": ["preparation", "acceleration", "ball_impact"],
    "backhand": ["preparation", "ball_impact", "follow_through"],
}


def select_best_combination(
    candidates: Dict[str, List[Tuple[int, float, str]]],
    gesture: str,
    fps: float,
    frames: List[Dict] = None,
    camera_reliability: Dict[str, float] = None,
) -> Tuple[Optional[Dict[str, int]], str]:
    """
    Sélectionne la meilleure combinaison de frames (top-5³ = 125 max)
    respectant l'ordre temporel et les séparations minimales.
    """
    phase_order = PHASE_ORDER[gesture]
    if camera_reliability is None:
        camera_reliability = {}

    # Convertir les séparations en frames
    min_between: Dict[Tuple[str, str], int] = {}
    for pair, secs in MIN_SECONDS_BETWEEN[gesture].items():
        min_between[pair] = seconds_to_frames(secs, fps)

    for phase in phase_order:
        if not candidates.get(phase):
            return None, f"Candidats manquants pour : {phase}"

    best       = None
    best_score = -1.0
    best_msg   = ""
    lists      = [candidates[p] for p in phase_order]

    def recurse(idx: int, current: Dict, score: float, penalty: float) -> None:
        nonlocal best, best_score, best_msg
        if idx == len(phase_order):
            valid, _ = validate_temporal_coherence(current, phase_order, min_between)
            if not valid:
                return
            combined = score - penalty
            if combined > best_score:
                best_score = combined
                best       = dict(current)
                best_msg   = f"Score combiné : {combined:.3f}"
            return
        for rank, (fn, sc, _) in enumerate(lists[idx]):
            current[phase_order[idx]] = fn
            recurse(idx+1, current, score+sc, penalty + rank*0.1)
        del current[phase_order[idx]]

    recurse(0, {}, 0.0, 0.0)

    if best is None:
        return None, ("Aucune combinaison valide trouvée.\n"
                      "→ Utilise --detail pour inspecter les candidats manuellement.")
    return best, best_msg


# ─────────────────────────────────────────────
# AFFICHAGE
# ─────────────────────────────────────────────

def get_key_angles_str(frames: List[Dict], frame_number: int,
                       phase_name: str, normatives: Dict) -> str:
    for f in frames:
        if f["frame_number"] == frame_number:
            joints = list(normatives.get(phase_name, {}).keys())[:3]
            parts  = []
            for j in joints:
                val = f["angles"].get(j)
                if val is not None:
                    short = (j.replace("_right","D").replace("_left","G")
                              .replace("knee_flexion","genou")
                              .replace("shoulder_elevation","elev_ep")
                              .replace("shoulder_rotation","rot_ep")
                              .replace("trunk_rotation","rot_tronc")
                              .replace("trunk_inclination","incl_tronc")
                              .replace("elbow","coude").replace("hip","hanche"))
                    parts.append(f"{short}={val:.1f}°")
            return " | ".join(parts)
    return ""


def print_angles_at_frame(frames: List[Dict], frame_number: int) -> None:
    for f in frames:
        if f["frame_number"] == frame_number:
            print(f"\n  Angles à la frame {frame_number} :")
            for joint, val in f["angles"].items():
                if val is not None:
                    print(f"    {joint:35s} : {val:7.2f}°")
            return


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Détection des phases tennis — ancre biomécanique"
    )
    parser.add_argument("--frames",  required=True)
    parser.add_argument("--gesture", required=True,
                        choices=["serve", "forehand", "backhand"])
    parser.add_argument("--variant", default=None, choices=["1h", "2h"])
    parser.add_argument("--detail",  action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.frames):
        print(f"[Erreur] Fichier introuvable : {args.frames}")
        return

    frames  = load_frames(args.frames)
    n       = len(frames)
    gesture = args.gesture
    print(f"\n[find_phases_gestures] {n} frames | geste : {gesture.upper()}")

    fps = detect_fps(frames)
    print(f"[find_phases_gestures] FPS détecté : {fps}")

    variant = None
    if gesture == "backhand":
        variant = args.variant if args.variant else detect_backhand_variant(frames)
        print(f"[find_phases_gestures] Variante : {variant.upper()}")

    camera_reliability = check_camera_angle_reliability(frames, gesture)
    normatives         = get_normatives(gesture, variant)

    candidates = find_all_candidates(frames, gesture, normatives, fps,
                                     camera_reliability)

    print("\n" + "="*70)
    print(f"FRAMES CANDIDATES — {gesture.upper()}"
          + (f" ({variant.upper()})" if variant else ""))
    print("="*70)

    phase_order = PHASE_ORDER[gesture]
    for phase_name in phase_order:
        frame_list = candidates.get(phase_name, [])
        anchor_marker = " ← ANCRE" if phase_name == ANCHOR_PHASE[gesture] else ""
        print(f"\n  {phase_name.upper()}{anchor_marker}")
        if not frame_list:
            print("  Aucun candidat trouvé")
            continue
        print(f"  {'Frame':>8}  {'Score':>8}  {'Confiance':>12}  Angles clés")
        print(f"  {'-'*65}")
        for i, (fn, score, conf) in enumerate(frame_list):
            rank = ["★ Meilleur", "  2e choix", "  3e choix",
                    "  4e choix", "  5e choix"][i]
            conf_disp  = f"{CONFIDENCE_COLOR[conf]} {conf}"
            angles_str = get_key_angles_str(frames, fn, phase_name, normatives)
            print(f"  {fn:>8}  {score:>8.3f}  {conf_disp:>18}  "
                  f"{rank}  →  {angles_str}")

    print("\n" + "="*70)
    print("SÉLECTION OPTIMALE")
    print("="*70)

    best, msg = select_best_combination(
        candidates, gesture, fps, frames=frames,
        camera_reliability=camera_reliability,
    )

    if best is not None:
        for phase_name in phase_order:
            fn   = best[phase_name]
            conf = candidates[phase_name][0][2] if candidates.get(phase_name) else "N/A"
            print(f"  {phase_name:20s} → frame {fn:5d}  "
                  f"{CONFIDENCE_COLOR.get(conf,'?')} {conf}")
        print(f"\n  {msg}")

        if args.detail:
            for fn in best.values():
                print_angles_at_frame(frames, fn)

        basename   = os.path.basename(args.frames)
        session_id = basename.replace("frames_angles_","").replace(".json","")
        phases_str = ",".join(f"{p}={f}" for p, f in best.items())
        print(f"""
  python src/pipeline/pose_ex3_3d_savgol.py \\
    --video "output/tmp_video/input_video.mp4" \\
    --session_id "{session_id}_p2" \\
    --phases "{phases_str}"
""")
        if camera_reliability and any(v < 1.0 for v in camera_reliability.values()):
            degraded = [j for j, v in camera_reliability.items() if v < 1.0]
            print(f"  ⚠ Joints dégradés par angle caméra : {', '.join(degraded)}")
    else:
        print(f"\n  {msg}")

    print()


if __name__ == "__main__":
    main()