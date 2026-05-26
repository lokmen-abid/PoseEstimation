"""
=============================================================================
IA/Serve — find_phases_2.py
Identification améliorée des frames clés du service tennis
Version 2 — 5 améliorations par rapport à find___phases.py

Améliorations vs find___phases.py :
  1. Zones temporelles contraintes par phase (élimine l'ordre inversé)
  2. Détection d'extremum local réel (±10 frames)
  3. Pondération dynamique par qualité de signal
  4. Score de confiance explicite (HIGH / MEDIUM / LOW / UNRELIABLE)
  5. Cohérence inter-phases — séparation temporelle minimale

Usage :
  python src/pipeline/find_phases_2.py --frames output/results/frames_angles_test02.json
  python src/pipeline/find_phases_2.py --frames output/results/frames_angles_test02.json --detail
  python src/pipeline/find_phases_2.py --frames output/results/frames_angles_test02.json --compare

  --compare : affiche côte à côte find___phases v1 vs v2 pour la même vidéo
=============================================================================
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────
# CONFIGURATION PHASES
# ─────────────────────────────────────────────

# Amélioration 1 — Zones temporelles relatives
# Basé sur la biomécanique du service (Gorce 2024)
# Les fractions correspondent à la position relative dans la vidéo
PHASE_ZONES = {
    "trophy_position":  (0.10, 0.50),  # première moitié de la vidéo
    "racket_low_point": (0.35, 0.70),  # zone intermédiaire — armement
    "ball_impact":      (0.55, 0.90),  # fin de vidéo — contact balle
}

# Séparation minimale entre phases (amélioration 5)
# Valeurs en frames — ajustées selon le FPS réel à l'usage
MIN_FRAMES_BETWEEN = {
    ("trophy_position",  "racket_low_point"): 15,  # ~0.5s à 30fps
    ("racket_low_point", "ball_impact"):      10,  # ~0.3s à 30fps
}

# Cibles normatives (Gorce 2024) — inchangées vs v1
PHASE_TARGETS = {
    "trophy_position": {
        "knee_flexion_right":       {"target": 64.5,  "look": "min",  "weight": 3},
        "trunk_inclination":        {"target": 25.0,  "look": "near", "weight": 2},
        "shoulder_elevation_right": {"target": 80.0,  "look": "near", "weight": 1},
    },
    "racket_low_point": {
        "shoulder_rotation_right":  {"target": 130.1, "look": "max",  "weight": 3},
        "elbow_right":              {"target": 90.0,  "look": "near", "weight": 2},
    },
    "ball_impact": {
        "shoulder_elevation_right": {"target": 110.7, "look": "near", "weight": 3},
        "elbow_right":              {"target": 30.1,  "look": "min",  "weight": 3},
        "knee_flexion_right":       {"target": 160.0, "look": "near", "weight": 1},
    },
}

# Écarts-types normatifs pour le calcul de qualité de signal
NORMATIVE_STD = {
    "knee_flexion_right":       9.7,
    "trunk_inclination":        7.1,
    "shoulder_rotation_right":  26.5,
    "shoulder_elevation_right": 16.9,
    "elbow_right":              15.9,
}

# Seuil de qualité : si la meilleure frame est à plus de N*SD de la normative
# → signal considéré comme biaisé (angle de caméra sous-optimal)
QUALITY_STD_THRESHOLD = 2.0

# Fenêtre pour la détection d'extremum local (amélioration 2)
LOCAL_EXTREMUM_WINDOW = 10

# Pénalité appliquée si la frame n'est pas un extremum local
LOCAL_EXTREMUM_PENALTY = 0.3

# ─────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────

def load_frames(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ─────────────────────────────────────────────
# AMÉLIORATION 2 — Détection d'extremum local
# ─────────────────────────────────────────────

def is_local_extremum(values: np.ndarray, idx: int,
                      window: int = LOCAL_EXTREMUM_WINDOW,
                      kind: str = "min") -> bool:
    """
    Vérifie que values[idx] est un vrai extremum local.
    kind = 'min' : valeur <= tous les voisins dans la fenêtre
    kind = 'max' : valeur >= tous les voisins dans la fenêtre
    Les NaN dans le voisinage sont ignorés.
    """
    if np.isnan(values[idx]):
        return False

    start = max(0, idx - window)
    end   = min(len(values), idx + window + 1)
    neighborhood = values[start:end]
    valid_neighbors = neighborhood[~np.isnan(neighborhood)]

    if len(valid_neighbors) < 3:
        # Pas assez de voisins pour juger — on accepte par défaut
        return True

    if kind == "min":
        return float(values[idx]) <= float(np.min(valid_neighbors))
    else:
        return float(values[idx]) >= float(np.max(valid_neighbors))

# ─────────────────────────────────────────────
# AMÉLIORATION 3 — Qualité de signal
# ─────────────────────────────────────────────

def compute_signal_quality(values: np.ndarray, target: float,
                           joint_name: str) -> float:
    """
    Retourne un facteur de qualité entre 0.2 et 1.0.

    Si aucune frame de la vidéo n'approche la normative pour ce joint
    (signal biaisé par l'angle de caméra), la contribution de ce joint
    est réduite automatiquement.

    Logique :
      - Calcule la proximité minimale à la cible sur toute la vidéo
      - Si la meilleure frame est à plus de QUALITY_STD_THRESHOLD * SD
        de la normative → signal faible → facteur 0.2
      - Sinon → facteur 1.0
    """
    std = NORMATIVE_STD.get(joint_name, 20.0)
    valid_values = values[~np.isnan(values)]

    if len(valid_values) == 0:
        return 0.0

    best_proximity = float(np.min(np.abs(valid_values - target)))

    if best_proximity > QUALITY_STD_THRESHOLD * std:
        return 0.2  # signal présent mais biaisé — contribution réduite
    return 1.0

# ─────────────────────────────────────────────
# AMÉLIORATION 4 — Score de confiance
# ─────────────────────────────────────────────

def compute_confidence(score: float, max_possible_score: float) -> str:
    """
    Retourne un niveau de confiance interprétable.

    HIGH        : ratio >= 75% — validation 1 clic recommandée
    MEDIUM      : ratio 50–74% — ajustement possible via slider
    LOW         : ratio 25–49% — vérification manuelle recommandée
    UNRELIABLE  : ratio < 25%  — sélection manuelle forcée
    """
    if max_possible_score <= 0:
        return "UNRELIABLE"
    ratio = score / max_possible_score
    if ratio >= 0.75:
        return "HIGH"
    elif ratio >= 0.50:
        return "MEDIUM"
    elif ratio >= 0.25:
        return "LOW"
    else:
        return "UNRELIABLE"

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
# AMÉLIORATION 1+2+3 — Candidats par phase
# ─────────────────────────────────────────────

def find_candidate_frames_v2(
    frames: List[Dict]
) -> Dict[str, List[Tuple[int, float, str]]]:
    """
    Pour chaque phase, calcule un score de proximité aux valeurs cibles
    en appliquant les 5 améliorations.

    Retourne un dict :
      { phase_name: [(frame_number, score, confidence), ...] }  — top 5
    """
    n = len(frames)
    candidates = {}

    for phase_name, joints in PHASE_TARGETS.items():

        # ── Amélioration 1 : zone temporelle ──
        zone_start, zone_end = PHASE_ZONES[phase_name]
        idx_start = int(n * zone_start)
        idx_end   = int(n * zone_end)
        zone_size = idx_end - idx_start

        if zone_size < 5:
            # Vidéo trop courte pour cette zone — fallback sur toute la vidéo
            idx_start, idx_end = 0, n
            zone_size = n

        frames_in_zone = frames[idx_start:idx_end]
        scores = np.zeros(zone_size)

        # Calcul du score maximum théorique (pour la confiance)
        max_possible = 0.0

        for joint_name, cfg in joints.items():
            target = cfg["target"]
            look   = cfg["look"]
            weight = cfg["weight"]

            # Extrait les valeurs sur la zone
            values_zone = np.array([
                f["angles"].get(joint_name)
                if f["angles"].get(joint_name) is not None
                else np.nan
                for f in frames_in_zone
            ])

            valid = ~np.isnan(values_zone)
            if not valid.any():
                continue

            # ── Amélioration 3 : qualité de signal ──
            # Calcul sur TOUTE la vidéo (pas seulement la zone)
            values_full = np.array([
                f["angles"].get(joint_name)
                if f["angles"].get(joint_name) is not None
                else np.nan
                for f in frames
            ])
            quality = compute_signal_quality(values_full, target, joint_name)
            effective_weight = weight * quality

            max_possible += effective_weight  # contribution max = 1.0 par frame parfaite

            # ── Calcul proximité ──
            if look == "min":
                # Amélioration vs v1 : minimum LOCAL dans la zone, pas global vidéo
                min_val_zone = np.nanmin(values_zone)
                proximity = 1.0 / (1.0 + np.abs(values_zone - min_val_zone))
            elif look == "max":
                max_val_zone = np.nanmax(values_zone)
                proximity = 1.0 / (1.0 + np.abs(values_zone - max_val_zone))
            else:  # "near"
                proximity = 1.0 / (1.0 + np.abs(values_zone - target))

            proximity[~valid] = 0.0

            # ── Amélioration 2 : pénalité si pas extremum local ──
            if look in ("min", "max"):
                kind = "min" if look == "min" else "max"
                for i in range(zone_size):
                    if not np.isnan(values_zone[i]):
                        if not is_local_extremum(values_zone, i, kind=kind):
                            proximity[i] *= LOCAL_EXTREMUM_PENALTY

            scores += effective_weight * proximity

        # ── Top 5 dans la zone ──
        top_indices = np.argsort(scores)[::-1][:5]

        # Calcul confiance pour le meilleur candidat
        best_score = float(scores[top_indices[0]]) if len(top_indices) > 0 else 0.0
        confidence = compute_confidence(best_score, max_possible)

        candidates[phase_name] = [
            (
                int(frames_in_zone[i]["frame_number"]),
                float(scores[i]),
                confidence if idx == 0 else compute_confidence(float(scores[i]), max_possible)
            )
            for idx, i in enumerate(top_indices)
            if scores[i] > 0
        ]

    return candidates

# ─────────────────────────────────────────────
# AMÉLIORATION 5 — Cohérence temporelle
# ─────────────────────────────────────────────

def validate_temporal_coherence(trophy_f: int, rlp_f: int,
                                 impact_f: int) -> Tuple[bool, str]:
    """
    Vérifie que les trois frames respectent :
    1. L'ordre Trophy < RLP < Ball Impact
    2. Les séparations minimales entre phases
    """
    if not (trophy_f < rlp_f < impact_f):
        return False, f"Ordre incorrect : {trophy_f} → {rlp_f} → {impact_f}"

    min_tr = MIN_FRAMES_BETWEEN[("trophy_position", "racket_low_point")]
    min_ri = MIN_FRAMES_BETWEEN[("racket_low_point", "ball_impact")]

    if rlp_f - trophy_f < min_tr:
        return False, f"Trophy→RLP trop proches ({rlp_f - trophy_f} frames, min {min_tr})"
    if impact_f - rlp_f < min_ri:
        return False, f"RLP→Impact trop proches ({impact_f - rlp_f} frames, min {min_ri})"

    return True, "OK"


def select_best_combination(
    candidates: Dict[str, List[Tuple[int, float, str]]]
) -> Tuple[Optional[int], Optional[int], Optional[int], str]:
    """
    Sélectionne la meilleure combinaison Trophy/RLP/Impact en respectant :
    - L'ordre temporel
    - Les séparations minimales (amélioration 5)
    - Le score combiné maximal

    Retourne (trophy_f, rlp_f, impact_f, message).
    Plus de fallback sans contrainte — si aucune combinaison valide,
    retourne None avec un message explicatif.
    """
    best        = None
    best_score  = -1
    best_msg    = ""

    trophy_list = candidates.get("trophy_position",  [])
    rlp_list    = candidates.get("racket_low_point", [])
    impact_list = candidates.get("ball_impact",      [])

    if not trophy_list or not rlp_list or not impact_list:
        return None, None, None, "Candidats manquants pour une ou plusieurs phases"

    for ti, (tf, ts, _) in enumerate(trophy_list):
        for ri, (rf, rs, _) in enumerate(rlp_list):
            for ii, (iff, is_, _) in enumerate(impact_list):
                valid, msg = validate_temporal_coherence(tf, rf, iff)
                if not valid:
                    continue
                # Score combiné — pénalise les rangs inférieurs
                combined = ts + rs + is_ - (ti + ri + ii) * 0.1
                if combined > best_score:
                    best_score = combined
                    best       = (tf, rf, iff)
                    best_msg   = f"Score combiné : {combined:.3f}"

    if best is None:
        return None, None, None, (
            "Aucune combinaison valide trouvée avec les contraintes temporelles.\n"
            "  → Les zones temporelles sont peut-être inadaptées à cette vidéo.\n"
            "  → Utilise --detail pour inspecter les candidats et annoter manuellement."
        )

    return best[0], best[1], best[2], best_msg

# ─────────────────────────────────────────────
# AFFICHAGE
# ─────────────────────────────────────────────

def get_key_angles_str(frames: List[Dict], frame_number: int,
                       phase_name: str) -> str:
    """Retourne une string des angles clés pour une frame et une phase."""
    for f in frames:
        if f["frame_number"] == frame_number:
            angles = f["angles"]
            key_angles = []
            if phase_name == "trophy_position":
                kf = angles.get("knee_flexion_right")
                ti = angles.get("trunk_inclination")
                if kf is not None: key_angles.append(f"genou={kf:.1f}°")
                if ti is not None: key_angles.append(f"tronc={ti:.1f}°")
            elif phase_name == "racket_low_point":
                sr = angles.get("shoulder_rotation_right")
                er = angles.get("elbow_right")
                if sr is not None: key_angles.append(f"rot_épaule={sr:.1f}°")
                if er is not None: key_angles.append(f"coude={er:.1f}°")
            elif phase_name == "ball_impact":
                se = angles.get("shoulder_elevation_right")
                er = angles.get("elbow_right")
                if se is not None: key_angles.append(f"elev_épaule={se:.1f}°")
                if er is not None: key_angles.append(f"coude={er:.1f}°")
            return " | ".join(key_angles)
    return ""


def print_angles_at_frame(frames: List[Dict], frame_number: int):
    """Affiche tous les angles à une frame précise."""
    for f in frames:
        if f["frame_number"] == frame_number:
            print(f"\n  Angles à la frame {frame_number} :")
            for joint, val in f["angles"].items():
                if val is not None:
                    print(f"    {joint:35s} : {val:7.2f}°")
            return
    print(f"  Frame {frame_number} introuvable")


def print_signal_quality_report(frames: List[Dict]):
    """Affiche le rapport de qualité du signal pour tous les joints."""
    print("\n" + "="*70)
    print("RAPPORT QUALITÉ DU SIGNAL (Amélioration 3)")
    print("="*70)
    print(f"\n  {'Joint':35s}  {'Min vidéo':>10}  {'Cible':>8}  {'Δ min':>8}  {'Qualité'}")
    print(f"  {'-'*70}")

    joint_targets = {
        "knee_flexion_right":       64.5,
        "trunk_inclination":        25.0,
        "shoulder_rotation_right":  130.1,
        "shoulder_elevation_right": 110.7,
        "elbow_right":              30.1,
    }

    for joint_name, target in joint_targets.items():
        values = np.array([
            f["angles"].get(joint_name)
            if f["angles"].get(joint_name) is not None else np.nan
            for f in frames
        ])
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            print(f"  {joint_name:35s}  {'N/A':>10}  {target:>8.1f}  {'N/A':>8}  ❌ Absent")
            continue

        best_val      = float(np.min(np.abs(valid - target)))
        quality       = compute_signal_quality(values, target, joint_name)
        std           = NORMATIVE_STD.get(joint_name, 20.0)
        quality_label = "Bon" if quality == 1.0 else "⚠ Biaisé (caméra?)"

        closest_val = valid[np.argmin(np.abs(valid - target))]
        print(f"  {joint_name:35s}  {closest_val:>10.1f}°  {target:>8.1f}°  {best_val:>7.1f}°  {quality_label}")


def print_zone_info(n_frames: int):
    """Affiche les zones temporelles appliquées."""
    print("\n  Zones temporelles appliquées (Amélioration 1) :")
    for phase, (zs, ze) in PHASE_ZONES.items():
        f_start = int(n_frames * zs)
        f_end   = int(n_frames * ze)
        print(f"    {phase:20s} → frames {f_start:4d} à {f_end:4d}  ({zs*100:.0f}%–{ze*100:.0f}%)")

# ─────────────────────────────────────────────
# COMPARAISON V1 vs V2
# ─────────────────────────────────────────────

def run_v1_candidates(frames: List[Dict]) -> Dict[str, List[Tuple[int, float]]]:
    """
    Reproduit exactement la logique de find___phases.py v1
    pour permettre la comparaison côte à côte.
    """
    n = len(frames)
    candidates = {}

    for phase_name, joints in PHASE_TARGETS.items():
        scores = np.zeros(n)

        for joint_name, cfg in joints.items():
            target = cfg["target"]
            look   = cfg["look"]
            weight = cfg["weight"]

            values = np.array([
                f["angles"].get(joint_name)
                if f["angles"].get(joint_name) is not None else np.nan
                for f in frames
            ])
            valid = ~np.isnan(values)
            if not valid.any():
                continue

            if look == "min":
                min_val = np.nanmin(values)
                proximity = 1.0 / (1.0 + np.abs(values - min_val))
            elif look == "max":
                max_val = np.nanmax(values)
                proximity = 1.0 / (1.0 + np.abs(values - max_val))
            else:
                proximity = 1.0 / (1.0 + np.abs(values - target))

            proximity[~valid] = 0.0
            scores += weight * proximity

        top_indices = np.argsort(scores)[::-1][:5]
        candidates[phase_name] = [
            (int(frames[i]["frame_number"]), float(scores[i]))
            for i in top_indices if scores[i] > 0
        ]

    return candidates


def analyze_temporal_order_v1(candidates):
    """Reproduit analyze_temporal_order de v1."""
    trophy_frames = [f for f, _ in candidates.get("trophy_position", [])]
    rlp_frames    = [f for f, _ in candidates.get("racket_low_point", [])]
    impact_frames = [f for f, _ in candidates.get("ball_impact", [])]

    if not trophy_frames or not rlp_frames or not impact_frames:
        return None, None, None

    best = None
    best_score = -1
    for ti, (tf, ts) in enumerate(candidates.get("trophy_position", [])):
        for ri, (rf, rs) in enumerate(candidates.get("racket_low_point", [])):
            for ii, (iff, is_) in enumerate(candidates.get("ball_impact", [])):
                if tf < rf < iff:
                    combined = ts + rs + is_ - (ti + ri + ii) * 0.1
                    if combined > best_score:
                        best_score = combined
                        best = (tf, rf, iff)

    if best is None:
        best = (trophy_frames[0], rlp_frames[0], impact_frames[0])  # fallback dangereux
    return best


def print_comparison(frames: List[Dict]):
    """Affiche un tableau comparatif v1 vs v2."""
    print("\n" + "="*70)
    print("COMPARAISON find_phases v1 vs find_phases_2 v2")
    print("="*70)

    # V1
    cands_v1 = run_v1_candidates(frames)
    result_v1 = analyze_temporal_order_v1(cands_v1)
    trophy_v1, rlp_v1, impact_v1 = result_v1

    # V2
    cands_v2 = find_candidate_frames_v2(frames)
    trophy_v2, rlp_v2, impact_v2, msg_v2 = select_best_combination(cands_v2)

    print(f"\n  {'Phase':25s}  {'V1 (find_phases)':>18}  {'V2 (find_phases_2)':>20}")
    print(f"  {'-'*65}")

    phases = [
        ("trophy_position",  trophy_v1,  trophy_v2),
        ("racket_low_point", rlp_v1,     rlp_v2),
        ("ball_impact",      impact_v1,  impact_v2),
    ]

    for phase_name, fv1, fv2 in phases:
        v1_str = f"frame {fv1}" if fv1 is not None else "N/A"
        v2_str = f"frame {fv2}" if fv2 is not None else "N/A"

        # Confiance v2
        if fv2 is not None:
            conf = cands_v2.get(phase_name, [(None, 0, "UNRELIABLE")])[0][2]
            conf_sym = CONFIDENCE_COLOR.get(conf, "?")
            v2_str += f"  {conf_sym} {conf}"

        changed = " ← DIFFÉRENT" if fv1 != fv2 else ""
        print(f"  {phase_name:25s}  {v1_str:>18}  {v2_str:<30}{changed}")

    # Ordre temporel V1
    if trophy_v1 and rlp_v1 and impact_v1:
        order_v1_ok = trophy_v1 < rlp_v1 < impact_v1
        order_v1_str = "Correct" if order_v1_ok else "❌ INVERSÉ"
    else:
        order_v1_str = "Incomplet"

    # Ordre temporel V2
    if trophy_v2 and rlp_v2 and impact_v2:
        order_v2_ok = trophy_v2 < rlp_v2 < impact_v2
        order_v2_str = "Correct (garanti)"
    else:
        order_v2_str = "combinaison valide"

    print(f"\n  {'Ordre temporel':25s}  {order_v1_str:>18}  {order_v2_str}")
    print(f"\n  V2 — {msg_v2}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="IA/Serve v2 — Identification améliorée des phases du service"
    )
    parser.add_argument("--frames", type=str, required=True,
                        help="Chemin vers frames_angles_{session_id}.json")
    parser.add_argument("--detail", action="store_true",
                        help="Affiche les angles détaillés aux frames sélectionnées")
    parser.add_argument("--compare", action="store_true",
                        help="Affiche la comparaison v1 vs v2 côte à côte")
    args = parser.parse_args()

    if not os.path.exists(args.frames):
        print(f"[Erreur] Fichier introuvable : {args.frames}")
        return

    print(f"\n[find_phases_2] Lecture : {args.frames}")
    frames = load_frames(args.frames)
    n      = len(frames)
    print(f"[find_phases_2] {n} frames chargées")

    # ── Zones temporelles appliquées ──
    print_zone_info(n)

    # ── Rapport qualité signal ──
    print_signal_quality_report(frames)

    # ── Candidats v2 ──
    candidates = find_candidate_frames_v2(frames)

    print("\n" + "="*70)
    print("FRAMES CANDIDATES PAR PHASE (zones contraintes)")
    print("="*70)

    norm_info = {
        "trophy_position":  "Genou ~64.5° | Tronc ~25°  (Gorce 2024)",
        "racket_low_point": "Rotation épaule ~130.1°     (Gorce 2024)",
        "ball_impact":      "Élévation épaule ~110.7° | Coude ~30.1° (Gorce 2024)",
    }

    for phase_name, frame_list in candidates.items():
        zone_start, zone_end = PHASE_ZONES[phase_name]
        f_start = int(n * zone_start)
        f_end   = int(n * zone_end)

        print(f"\n  {phase_name.upper()}")
        print(f"  Référence : {norm_info.get(phase_name, '')}")
        print(f"  Zone recherche : frames {f_start}–{f_end}")
        print(f"  {'Frame':>8}  {'Score':>8}  {'Confiance':>12}  Angles clés")
        print(f"  {'-'*65}")

        for i, (fn, score, conf) in enumerate(frame_list):
            rank      = ["★ Meilleur", "  2e choix", "  3e choix", "  4e choix", "  5e choix"][i]
            conf_disp = f"{CONFIDENCE_COLOR[conf]} {conf}"
            angles_str = get_key_angles_str(frames, fn, phase_name)
            print(f"  {fn:>8}  {score:>8.3f}  {conf_disp:>18}  {rank}  →  {angles_str}")

    # ── Sélection optimale ──
    print("\n" + "="*70)
    print("SÉLECTION OPTIMALE (ordre garanti + séparation minimale)")
    print("="*70)

    trophy_f, rlp_f, impact_f, msg = select_best_combination(candidates)

    if trophy_f is not None:
        conf_trophy = candidates["trophy_position"][0][2]   if candidates.get("trophy_position")  else "N/A"
        conf_rlp    = candidates["racket_low_point"][0][2]  if candidates.get("racket_low_point") else "N/A"
        conf_impact = candidates["ball_impact"][0][2]       if candidates.get("ball_impact")      else "N/A"

        print(f"\n  trophy_position    → frame {trophy_f:5d}  {CONFIDENCE_COLOR[conf_trophy]} {conf_trophy}")
        print(f"  racket_low_point   → frame {rlp_f:5d}  {CONFIDENCE_COLOR[conf_rlp]} {conf_rlp}")
        print(f"  ball_impact        → frame {impact_f:5d}  {CONFIDENCE_COLOR[conf_impact]} {conf_impact}")
        print(f"\n  {msg}")

        if args.detail:
            print_angles_at_frame(frames, trophy_f)
            print_angles_at_frame(frames, rlp_f)
            print_angles_at_frame(frames, impact_f)

        # ── Commande Passe 2 ──
        print("\n" + "="*70)
        print("COMMANDE PASSE 2 — copie-colle directement :")
        print("="*70)

        basename   = os.path.basename(args.frames)
        session_id = basename.replace("frames_angles_", "").replace(".json", "")
        video_hint = "output\\tmp_video\\input_video_ex3.mp4"

        print(f"""
  python src/pipeline/pose_ex3_3d_savgol.py \\
    --video "{video_hint}" \\
    --session_id "{session_id}_p2" \\
    --phases "trophy_position={trophy_f},racket_low_point={rlp_f},ball_impact={impact_f}"
""")
        print("  Note : retire --no_mongo si tu veux sauvegarder dans MongoDB")

        # ── Recommandation UI ──
        print("\n" + "="*70)
        print("RECOMMANDATION FRONTEND (Phase 4)")
        print("="*70)
        all_confs = [conf_trophy, conf_rlp, conf_impact]
        if all(c == "HIGH" for c in all_confs):
            print("\n  Toutes les phases HIGH → validation 1 clic suffisante")
        elif "UNRELIABLE" in all_confs:
            print("\n  Phase(s) UNRELIABLE → sélection manuelle forcée recommandée")
        else:
            print("\n  Confiance mixte → afficher slider d'ajustement au spécialiste")

    else:
        print(f"\n  {msg}")
        print("\n  → Lance avec --detail pour inspecter les candidats manuellement.")

    # ── Comparaison v1 vs v2 ──
    if args.compare:
        print_comparison(frames)

    print()


if __name__ == "__main__":
    main()