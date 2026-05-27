"""
=============================================================================
IA/Serve — find_phases_gestures.py
Détection des frames clés pour tous les gestes tennis
Version unifiée : serve | forehand | backhand (1 main / 2 mains auto)

Hérite de toutes les améliorations de find_phases_2.py :
  1. Zones temporelles contraintes par phase
  2. Détection d'extremum local réel (±10 frames)
  3. Pondération dynamique par qualité de signal
  4. Score de confiance (HIGH / MEDIUM / LOW / UNRELIABLE)
  5. Cohérence inter-phases — séparation temporelle minimale

Usage :
  # Service
  python src/pipeline/find_phases_gestures.py \
    --frames output/results/frames_angles_test02.json \
    --gesture serve

  # Coup droit
  python src/pipeline/find_phases_gestures.py \
    --frames output/results/frames_angles_forehand01.json \
    --gesture forehand

  # Revers (variante détectée automatiquement)
  python src/pipeline/find_phases_gestures.py \
    --frames output/results/frames_angles_backhand01.json \
    --gesture backhand

  Options supplémentaires :
    --detail   : affiche les angles détaillés aux frames sélectionnées
    --compare  : compare avec find_phases_2.py (serve uniquement)
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
# ZONES TEMPORELLES PAR GESTE
# Fractions relatives de la vidéo (0.0 → 1.0)
# ─────────────────────────────────────────────

PHASE_ZONES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "serve": {
        "trophy_position":  (0.10, 0.50),
        "racket_low_point": (0.35, 0.70),
        "ball_impact":      (0.55, 0.90),
    },
    "forehand": {
        "preparation":  (0.10, 0.45),
        "acceleration": (0.35, 0.70),
        "ball_impact":  (0.55, 0.90),
    },
    "backhand": {
        "preparation":    (0.10, 0.45),
        "ball_impact":    (0.40, 0.75),
        "follow_through": (0.60, 0.95),
    },
}

# ─────────────────────────────────────────────
# SÉPARATION MINIMALE ENTRE PHASES (secondes réelles)
# Converti en frames via adapt_to_fps() selon le FPS de la vidéo
# Valeurs basées sur la biomécanique du geste :
#   serve      : Trophy->RLP ~0.5s | RLP->Impact ~0.3s
#   forehand   : Prep->Accel ~0.3s | Accel->Impact ~0.25s
#   backhand   : Prep->Impact ~0.3s | Impact->Follow ~0.25s
# ─────────────────────────────────────────────

MIN_SECONDS_BETWEEN: Dict[str, Dict[Tuple[str, str], float]] = {
    "serve": {
        ("trophy_position",  "racket_low_point"): 0.50,
        ("racket_low_point", "ball_impact"):       0.30,
    },
    "forehand": {
        ("preparation",  "acceleration"): 0.30,
        ("acceleration", "ball_impact"):  0.25,
    },
    "backhand": {
        ("preparation", "ball_impact"):    0.30,
        ("ball_impact", "follow_through"): 0.25,
    },
}

# ─────────────────────────────────────────────
# PARAMÈTRES DE BASE (à 30 FPS — référence)
# ─────────────────────────────────────────────

LOCAL_EXTREMUM_WINDOW_SECONDS = 0.33   # +-10 frames a 30fps = +-0.33s
LOCAL_EXTREMUM_PENALTY        = 0.3
QUALITY_STD_THRESHOLD         = 2.0
FPS_REFERENCE                 = 30.0


def adapt_to_fps(fps: float) -> Tuple[int, Dict[str, Dict[Tuple[str, str], int]]]:
    """
    Adapte LOCAL_EXTREMUM_WINDOW et MIN_FRAMES_BETWEEN au FPS reel.

    Logique :
      - 0.5s reelle = 15 frames a 30fps, 30 frames a 60fps, 13 frames a 25fps
      - LOCAL_EXTREMUM_WINDOW suit la meme logique : +-0.33s reel
        = +-10 frames a 30fps, +-20 frames a 60fps

    Args:
        fps : FPS reel detecte depuis le JSON (timestamp_ms des frames)

    Returns:
        (local_window, min_frames_dict)
    """
    fps = max(fps, 1.0)
    local_window = max(3, int(round(LOCAL_EXTREMUM_WINDOW_SECONDS * fps)))
    min_frames: Dict[str, Dict[Tuple[str, str], int]] = {}
    for gesture, pairs in MIN_SECONDS_BETWEEN.items():
        min_frames[gesture] = {}
        for pair, seconds in pairs.items():
            min_frames[gesture][pair] = max(3, int(round(seconds * fps)))
    return local_window, min_frames


def detect_fps(frames: List[Dict]) -> float:
    """
    Estime le FPS reel depuis les timestamp_ms du JSON.
    Utilise la mediane des intervalles entre frames consecutives
    pour etre robuste aux frames manquantes.

    Returns:
        FPS estime (float), defaut 30.0 si non detectable
    """
    timestamps = [f.get("timestamp_ms") for f in frames if f.get("timestamp_ms") is not None]
    if len(timestamps) < 2:
        print("[FPS] Timestamps insuffisants — FPS par defaut : 30.0")
        return FPS_REFERENCE
    deltas = [timestamps[i+1] - timestamps[i]
              for i in range(len(timestamps) - 1)
              if timestamps[i+1] > timestamps[i]]
    if not deltas:
        print("[FPS] Deltas invalides — FPS par defaut : 30.0")
        return FPS_REFERENCE
    median_delta_ms = float(np.median(deltas))
    fps = round(1000.0 / median_delta_ms, 1)
    return fps

# Seuil de détection backhand 2 mains via les angles
# Si elbow_left est actif (non-None) sur plus de ce pourcentage de frames
# dans la zone ball_impact → 2 mains
BACKHAND_2H_ELBOW_LEFT_RATIO = 0.60

# Si la différence médiane entre elbow_left et elbow_right est inférieure
# à ce seuil en degrés → les deux coudes travaillent ensemble → 2 mains
BACKHAND_2H_ELBOW_DIFF_THRESHOLD = 40.0

CONFIDENCE_SYMBOL = {
    "HIGH":       "★★★★",
    "MEDIUM":     "★★★☆",
    "LOW":        "★★☆☆",
    "UNRELIABLE": "★☆☆☆",
}
CONFIDENCE_COLOR = {
    "HIGH":       "✅",
    "MEDIUM":     "🟡",
    "LOW":        "🟠",
    "UNRELIABLE": "❌",
}

# ─────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────

def load_frames(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ─────────────────────────────────────────────
# DÉTECTION VARIANTE BACKHAND (1M vs 2M)
# ─────────────────────────────────────────────

def detect_backhand_variant(frames: List[Dict]) -> str:
    """
    Détecte automatiquement si le backhand est à 1 main ou 2 mains.

    Logique basée sur les angles articulaires (indépendante de image_kps
    et de l'angle de caméra) :

    Critère 1 — Présence du coude gauche :
      Au backhand 2 mains, le bras non-dominant est actif tout au long
      du geste. Si elbow_left est détecté sur plus de 60% des frames
      dans la zone ball_impact → signal fort de 2 mains.

    Critère 2 — Symétrie des coudes :
      Au 2 mains, elbow_left et elbow_right ont des valeurs proches
      (les deux bras poussent ensemble). Si la différence médiane
      entre les deux coudes est < 40° → 2 mains.

    Les deux critères sont combinés pour robustesse.

    Returns:
        "1h" ou "2h"
    """
    n     = len(frames)
    start = int(n * 0.40)
    end   = int(n * 0.75)
    zone  = frames[start:end]

    if not zone:
        print("[Variante] Zone vide — hypothèse 1 main par défaut")
        return "1h"

    # ── Critère 1 : ratio de présence de elbow_left ──
    elbow_left_present = sum(
        1 for f in zone
        if f["angles"].get("elbow_left") is not None
    )
    presence_ratio = elbow_left_present / len(zone)

    # ── Critère 2 : différence médiane entre les deux coudes ──
    diffs = []
    for f in zone:
        el = f["angles"].get("elbow_left")
        er = f["angles"].get("elbow_right")
        if el is not None and er is not None:
            diffs.append(abs(el - er))

    median_diff = float(np.median(diffs)) if diffs else 999.0

    # ── Décision combinée ──
    is_2h_presence = presence_ratio >= BACKHAND_2H_ELBOW_LEFT_RATIO
    is_2h_symmetry = median_diff < BACKHAND_2H_ELBOW_DIFF_THRESHOLD

    print(f"[Variante] Présence elbow_left : {presence_ratio:.0%} "
          f"(seuil {BACKHAND_2H_ELBOW_LEFT_RATIO:.0%}) → "
          f"{'2H' if is_2h_presence else '1H'}")
    print(f"[Variante] Différence médiane coudes : {median_diff:.1f}° "
          f"(seuil {BACKHAND_2H_ELBOW_DIFF_THRESHOLD}°) → "
          f"{'2H' if is_2h_symmetry else '1H'}")

    # Les deux critères doivent être d'accord pour confirmer 2 mains
    # Un seul critère → 1 main par prudence
    if is_2h_presence and is_2h_symmetry:
        variant = "2h"
    elif is_2h_presence or is_2h_symmetry:
        # Un seul critère positif — on signale l'ambiguïté
        print("[Variante] ⚠ Critères contradictoires — "
              "utilise --variant 1h ou --variant 2h pour forcer")
        variant = "2h" if is_2h_presence else "1h"
    else:
        variant = "1h"

    print(f"[Variante] Décision finale → Backhand {variant.upper()}")
    return variant

# ─────────────────────────────────────────────
# EXTREMUM LOCAL
# ─────────────────────────────────────────────

def is_local_extremum(values: np.ndarray, idx: int,
                      window: int = 10,
                      kind: str = "min") -> bool:
    if np.isnan(values[idx]):
        return False
    start = max(0, idx - window)
    end   = min(len(values), idx + window + 1)
    neighborhood = values[start:end]
    valid = neighborhood[~np.isnan(neighborhood)]
    if len(valid) < 3:
        return True
    if kind == "min":
        return float(values[idx]) <= float(np.min(valid))
    return float(values[idx]) >= float(np.max(valid))

# ─────────────────────────────────────────────
# QUALITÉ DE SIGNAL
# ─────────────────────────────────────────────

def compute_signal_quality(values: np.ndarray, target: float,
                           joint_name: str) -> float:
    std   = NORMATIVE_STD_BY_JOINT.get(joint_name, 20.0)
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return 0.0
    best = float(np.min(np.abs(valid - target)))
    return 0.2 if best > QUALITY_STD_THRESHOLD * std else 1.0

# ─────────────────────────────────────────────
# SCORE DE CONFIANCE
# ─────────────────────────────────────────────

def compute_confidence(score: float, max_possible: float) -> str:
    if max_possible <= 0:
        return "UNRELIABLE"
    ratio = score / max_possible
    if ratio >= 0.75:   return "HIGH"
    elif ratio >= 0.50: return "MEDIUM"
    elif ratio >= 0.25: return "LOW"
    return "UNRELIABLE"

# ─────────────────────────────────────────────
# CANDIDATS PAR PHASE
# ─────────────────────────────────────────────

def find_candidate_frames(
    frames: List[Dict],
    gesture: str,
    normatives: Dict,
    zones: Dict[str, Tuple[float, float]],
    local_window: int = 10,
) -> Dict[str, List[Tuple[int, float, str]]]:
    """
    Pour chaque phase du geste, calcule un score de proximité
    aux valeurs normatives et retourne le top 5 candidats.

    Returns:
        { phase_name: [(frame_number, score, confidence), ...] }
    """
    n          = len(frames)
    candidates = {}

    for phase_name, joints in normatives.items():
        zone_start, zone_end = zones.get(phase_name, (0.0, 1.0))
        idx_start = int(n * zone_start)
        idx_end   = int(n * zone_end)
        zone_size = idx_end - idx_start

        if zone_size < 5:
            idx_start, idx_end = 0, n
            zone_size = n

        frames_zone  = frames[idx_start:idx_end]
        scores       = np.zeros(zone_size)
        max_possible = 0.0

        for joint_name, cfg in joints.items():
            target = cfg["target"]
            look   = cfg["look"]
            weight = cfg["weight"]

            values_zone = np.array([
                f["angles"].get(joint_name)
                if f["angles"].get(joint_name) is not None else np.nan
                for f in frames_zone
            ])
            valid = ~np.isnan(values_zone)
            if not valid.any():
                continue

            # Qualité de signal sur toute la vidéo
            values_full = np.array([
                f["angles"].get(joint_name)
                if f["angles"].get(joint_name) is not None else np.nan
                for f in frames
            ])
            quality          = compute_signal_quality(values_full, target, joint_name)
            effective_weight = weight * quality
            max_possible    += effective_weight

            # Proximité
            if look == "min":
                min_val   = np.nanmin(values_zone)
                proximity = 1.0 / (1.0 + np.abs(values_zone - min_val))
            elif look == "max":
                max_val   = np.nanmax(values_zone)
                proximity = 1.0 / (1.0 + np.abs(values_zone - max_val))
            else:  # "near"
                proximity = 1.0 / (1.0 + np.abs(values_zone - target))

            proximity[~valid] = 0.0

            # Pénalité si pas extremum local
            if look in ("min", "max"):
                kind = "min" if look == "min" else "max"
                for i in range(zone_size):
                    if not np.isnan(values_zone[i]):
                        if not is_local_extremum(values_zone, i, window=local_window, kind=kind):
                            proximity[i] *= LOCAL_EXTREMUM_PENALTY

            scores += effective_weight * proximity

        top = np.argsort(scores)[::-1][:5]
        best_score = float(scores[top[0]]) if len(top) > 0 else 0.0
        confidence = compute_confidence(best_score, max_possible)

        candidates[phase_name] = [
            (
                int(frames_zone[i]["frame_number"]),
                float(scores[i]),
                confidence if idx == 0
                else compute_confidence(float(scores[i]), max_possible)
            )
            for idx, i in enumerate(top)
            if scores[i] > 0
        ]

    return candidates

# ─────────────────────────────────────────────
# COHÉRENCE TEMPORELLE
# ─────────────────────────────────────────────

def validate_temporal_coherence(
    phase_frames: Dict[str, int],
    phase_order: List[str],
    min_between: Dict[Tuple[str, str], int],
) -> Tuple[bool, str]:
    """
    Vérifie l'ordre et les séparations minimales entre phases.
    """
    for i in range(len(phase_order) - 1):
        a, b = phase_order[i], phase_order[i + 1]
        fa, fb = phase_frames.get(a), phase_frames.get(b)
        if fa is None or fb is None:
            return False, f"Phase manquante : {a} ou {b}"
        if fa >= fb:
            return False, f"Ordre incorrect : {a}({fa}) >= {b}({fb})"
        min_sep = min_between.get((a, b), 0)
        if fb - fa < min_sep:
            return False, (
                f"{a}→{b} trop proches ({fb - fa} frames, min {min_sep})"
            )
    return True, "OK"


def select_best_combination(
    candidates: Dict[str, List[Tuple[int, float, str]]],
    gesture: str,
    min_frames: Dict[str, Dict[Tuple[str, str], int]] = None,
) -> Tuple[Optional[Dict[str, int]], str]:
    """
    Selectionne la meilleure combinaison de frames respectant
    l'ordre temporel et les separations minimales.

    Returns:
        (dict phase->frame, message)
        ou (None, message_erreur)
    """
    phase_order = list(PHASE_ZONES[gesture].keys())
    # Utilise min_frames adapte au FPS si fourni, sinon valeurs par defaut 30fps
    if min_frames is None:
        _, min_frames = adapt_to_fps(FPS_REFERENCE)
    min_between = min_frames[gesture]

    # Vérifie que tous les candidats existent
    for phase in phase_order:
        if not candidates.get(phase):
            return None, f"Candidats manquants pour la phase : {phase}"

    best       = None
    best_score = -1
    best_msg   = ""

    # Itère sur les combinaisons (top 5 × top 5 × top 5 = 125 max)
    lists = [candidates[p] for p in phase_order]

    def recurse(idx, current_frames, current_score, rank_penalty):
        nonlocal best, best_score, best_msg
        if idx == len(phase_order):
            valid, msg = validate_temporal_coherence(
                current_frames, phase_order, min_between
            )
            if valid:
                combined = current_score - rank_penalty
                if combined > best_score:
                    best_score = combined
                    best       = dict(current_frames)
                    best_msg   = f"Score combiné : {combined:.3f}"
            return
        for rank, (fn, score, _) in enumerate(lists[idx]):
            current_frames[phase_order[idx]] = fn
            recurse(idx + 1, current_frames, current_score + score,
                    rank_penalty + rank * 0.1)
        del current_frames[phase_order[idx]]

    recurse(0, {}, 0.0, 0.0)

    if best is None:
        return None, (
            "Aucune combinaison valide trouvée avec les contraintes temporelles.\n"
            "  → Utilise --detail pour inspecter les candidats manuellement."
        )
    return best, best_msg

# ─────────────────────────────────────────────
# AFFICHAGE
# ─────────────────────────────────────────────

def get_key_angles_str(frames: List[Dict], frame_number: int,
                       phase_name: str, normatives: Dict) -> str:
    for f in frames:
        if f["frame_number"] == frame_number:
            angles     = f["angles"]
            key_joints = list(normatives.get(phase_name, {}).keys())[:3]
            parts      = []
            for joint in key_joints:
                val = angles.get(joint)
                if val is not None:
                    short = joint.replace("_right", "D").replace("_left", "G") \
                                 .replace("knee_flexion", "genou") \
                                 .replace("shoulder_elevation", "elev_ep") \
                                 .replace("shoulder_rotation", "rot_ep") \
                                 .replace("trunk_rotation", "rot_tronc") \
                                 .replace("trunk_inclination", "incl_tronc") \
                                 .replace("elbow", "coude") \
                                 .replace("hip", "hanche")
                    parts.append(f"{short}={val:.1f}°")
            return " | ".join(parts)
    return ""


def print_angles_at_frame(frames: List[Dict], frame_number: int):
    for f in frames:
        if f["frame_number"] == frame_number:
            print(f"\n  Angles à la frame {frame_number} :")
            for joint, val in f["angles"].items():
                if val is not None:
                    print(f"    {joint:35s} : {val:7.2f}°")
            return
    print(f"  Frame {frame_number} introuvable")


def print_zone_info(n_frames: int, gesture: str):
    print(f"\n  Zones temporelles — {gesture.upper()} :")
    for phase, (zs, ze) in PHASE_ZONES[gesture].items():
        f_start = int(n_frames * zs)
        f_end   = int(n_frames * ze)
        print(f"    {phase:20s} → frames {f_start:4d} à {f_end:4d}"
              f"  ({zs*100:.0f}%–{ze*100:.0f}%)")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="IA/Serve — Détection des phases pour tous les gestes tennis"
    )
    parser.add_argument("--frames",  required=True,
                        help="Chemin vers frames_angles_{session_id}.json")
    parser.add_argument("--gesture", required=True,
                        choices=["serve", "forehand", "backhand"],
                        help="Geste à analyser")
    parser.add_argument("--variant", default=None,
                        choices=["1h", "2h"],
                        help="Backhand uniquement — force la variante (sinon auto-détectée)")
    parser.add_argument("--detail",  action="store_true",
                        help="Affiche les angles détaillés aux frames sélectionnées")
    args = parser.parse_args()

    if not os.path.exists(args.frames):
        print(f"[Erreur] Fichier introuvable : {args.frames}")
        return

    print(f"\n[find_phases_gestures] Lecture : {args.frames}")
    frames  = load_frames(args.frames)
    n       = len(frames)
    gesture = args.gesture
    print(f"[find_phases_gestures] {n} frames | geste : {gesture.upper()}")

    # ── Detection FPS + adaptation hyperparametres ──
    fps = detect_fps(frames)
    local_window, min_frames = adapt_to_fps(fps)
    print(f"[find_phases_gestures] FPS detecte : {fps} — "
          f"local_window={local_window} frames | "
          f"min_sep={list(min_frames[gesture].values())} frames")

    # ── Detection variante backhand ──
    variant = None
    if gesture == "backhand":
        variant = args.variant if args.variant else detect_backhand_variant(frames)
        print(f"[find_phases_gestures] Variante backhand : {variant.upper()}")

    # ── Chargement normatives ──
    normatives = get_normatives(gesture, variant)
    zones      = PHASE_ZONES[gesture]

    # ── Zones ──
    print_zone_info(n, gesture)

    # ── Candidats ──
    candidates = find_candidate_frames(frames, gesture, normatives, zones, local_window)

    print("\n" + "="*70)
    print(f"FRAMES CANDIDATES — {gesture.upper()}"
          + (f" ({variant.upper()})" if variant else ""))
    print("="*70)

    for phase_name, frame_list in candidates.items():
        zone_start, zone_end = zones[phase_name]
        f_start = int(n * zone_start)
        f_end   = int(n * zone_end)

        print(f"\n  {phase_name.upper()}")
        print(f"  Zone : frames {f_start}–{f_end}")
        print(f"  {'Frame':>8}  {'Score':>8}  {'Confiance':>12}  Angles clés")
        print(f"  {'-'*65}")

        for i, (fn, score, conf) in enumerate(frame_list):
            rank       = ["★ Meilleur", "  2e choix", "  3e choix",
                          "  4e choix", "  5e choix"][i]
            conf_disp  = f"{CONFIDENCE_COLOR[conf]} {conf}"
            angles_str = get_key_angles_str(frames, fn, phase_name, normatives)
            print(f"  {fn:>8}  {score:>8.3f}  {conf_disp:>18}  {rank}  →  {angles_str}")

    # ── Sélection optimale ──
    print("\n" + "="*70)
    print("SÉLECTION OPTIMALE")
    print("="*70)

    best, msg = select_best_combination(candidates, gesture, min_frames)

    if best is not None:
        phase_order = list(PHASE_ZONES[gesture].keys())
        for phase_name in phase_order:
            fn   = best[phase_name]
            conf = candidates[phase_name][0][2] if candidates.get(phase_name) else "N/A"
            conf_disp = f"{CONFIDENCE_COLOR.get(conf, '?')} {conf}"
            print(f"  {phase_name:20s} → frame {fn:5d}  {conf_disp}")

        print(f"\n  {msg}")

        if args.detail:
            for fn in best.values():
                print_angles_at_frame(frames, fn)

        # ── Commande Passe 2 ──
        print("\n" + "="*70)
        print("COMMANDE PASSE 2 — copie-colle directement :")
        print("="*70)

        basename   = os.path.basename(args.frames)
        session_id = basename.replace("frames_angles_", "").replace(".json", "")
        phases_str = ",".join(f"{p}={f}" for p, f in best.items())

        print(f"""
  python src/pipeline/pose_ex3_3d_savgol.py \\
    --video "output/tmp_video/input_video.mp4" \\
    --session_id "{session_id}_p2" \\
    --phases "{phases_str}"
""")

        # ── Recommandation UI ──
        all_confs = [candidates[p][0][2] for p in phase_order
                     if candidates.get(p)]
        print("\n" + "="*70)
        print("RECOMMANDATION FRONTEND")
        print("="*70)
        if all(c == "HIGH" for c in all_confs):
            print("\n  Toutes les phases HIGH → validation 1 clic suffisante")
        elif "UNRELIABLE" in all_confs:
            print("\n  Phase(s) UNRELIABLE → sélection manuelle forcée recommandée")
        else:
            print("\n  Confiance mixte → afficher slider d'ajustement au spécialiste")

    else:
        print(f"\n  {msg}")
        print("\n  → Lance avec --detail pour inspecter les candidats manuellement.")

    print()


if __name__ == "__main__":
    main()