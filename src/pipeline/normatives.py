"""
=============================================================================
IA/Serve — normatives.py
Valeurs normatives biomécaniques centralisées — tous gestes confondus

Sources :
  - Gorce & Jacquier-Bret (2024) Frontiers in Sports         → Service
  - Gorce & Jacquier-Bret (2024) Bioengineering 11(10), 974  → Service MSD
  - Knudson & Bahamonde (2001) Journal of Sports Sciences     → Forehand
  - Landlinger et al. JSSM 9(4), 643                         → Forehand
  - Creveaux et al. (2009) PMC7734511                        → Forehand hanche
  - Blackwell & Cole (1994) Journal of Biomechanics          → Backhand
  - Elliott (2008) BJSM PMC2577481                           → Backhand / Service
=============================================================================
"""

from typing import Dict

# ─────────────────────────────────────────────
# FORMAT COMMUN PAR ENTRÉE
# ─────────────────────────────────────────────
# {
#   "target"    : float  — valeur cible (mean normative)
#   "std"       : float  — écart-type normatif
#   "look"      : str    — "min" | "max" | "near"
#   "weight"    : int    — importance relative dans le score (1–3)
#   "alert_min" : float  — seuil alerte bas  (optionnel)
#   "alert_max" : float  — seuil alerte haut (optionnel)
#   "ref"       : str    — référence bibliographique
#   "note"      : str    — interprétation clinique (optionnel)
# }

# ─────────────────────────────────────────────
# SERVICE (SERVE)
# ─────────────────────────────────────────────

SERVE_NORMATIVES: Dict[str, Dict] = {

    # ── Trophy Position ──────────────────────
    "trophy_position": {
        "knee_flexion_right": {
            "target": 64.5, "std": 9.7,
            "look": "min", "weight": 3,
            "alert_min": 45.0, "alert_max": 84.2,
            "ref": "Gorce2024",
            "note": "Flexion genou avant — poussée des jambes"
        },
        "trunk_inclination": {
            "target": 25.0, "std": 7.1,
            "look": "near", "weight": 2,
            "alert_min": 10.0, "alert_max": 40.0,
            "ref": "Gorce2024",
            "note": "Inclinaison tronc vers l'avant"
        },
        "shoulder_elevation_right": {
            "target": 80.0, "std": 15.0,
            "look": "near", "weight": 1,
            "ref": "Gorce2024",
            "note": "Élévation épaule droite en Trophy"
        },
    },

    # ── Racket Low Point ─────────────────────
    "racket_low_point": {
        "shoulder_rotation_right": {
            "target": 130.1, "std": 26.5,
            "look": "max", "weight": 3,
            "alert_min": 77.1, "alert_max": 183.1,
            "ref": "Gorce2024",
            "note": "Rotation latérale épaule — armement maximal"
        },
        "elbow_right": {
            "target": 90.0, "std": 20.0,
            "look": "near", "weight": 2,
            "ref": "Gorce2024",
            "note": "Coude fléchi au point bas de la raquette"
        },
    },

    # ── Ball Impact ──────────────────────────
    "ball_impact": {
        "shoulder_elevation_right": {
            "target": 110.7, "std": 16.9,
            "look": "near", "weight": 3,
            "alert_min": 77.0, "alert_max": 144.5,
            "ref": "Gorce2024",
            "note": "Élévation épaule au contact balle"
        },
        "elbow_right": {
            "target": 30.1, "std": 15.9,
            "look": "min", "weight": 3,
            "alert_min": 0.0, "alert_max": 62.0,
            "ref": "Gorce2024",
            "note": "Extension quasi-complète du coude à l'impact"
        },
        "knee_flexion_right": {
            "target": 160.0, "std": 15.0,
            "look": "near", "weight": 1,
            "ref": "Gorce2024",
            "note": "Extension des jambes à l'impact"
        },
    },
}

# ─────────────────────────────────────────────
# FOREHAND (COUP DROIT)
# ─────────────────────────────────────────────

FOREHAND_NORMATIVES: Dict[str, Dict] = {

    # ── Preparation (backswing maximal) ──────
    "preparation": {
        "trunk_rotation": {
            "target": 60.0, "std": 15.0,
            "look": "max", "weight": 3,
            "alert_min": 45.0,
            "ref": "Knudson2001",
            "note": "Rotation tronc vers l'arrière — < 60° = surcharge lombaire compensatoire"
        },
        "knee_flexion_right": {
            "target": 140.0, "std": 15.0,
            "look": "min", "weight": 2,
            "ref": "Knudson2001",
            "note": "Flexion genou — transfert de puissance"
        },
    },

    # ── Acceleration (séparation bassin-épaules) ──
    "acceleration": {
        "trunk_rotation": {
            "target": 97.6, "std": 20.0,
            "look": "max", "weight": 3,
            "alert_min": 60.0,
            "ref": "Landlinger",
            "note": "Séparation bassin-épaules maximale — indicateur élite"
        },
        "hip_right": {
            "target": 60.3, "std": 15.0,
            "look": "near", "weight": 2,
            "ref": "Landlinger",
            "note": "Rotation pelvienne — séquence proximale-distale"
        },
        "shoulder_elevation_right": {
            "target": 90.0, "std": 20.0,
            "look": "near", "weight": 1,
            "ref": "Knudson2001",
            "note": "Élévation épaule en phase d'accélération"
        },
    },

    # ── Ball Impact ──────────────────────────
    "ball_impact": {
        "elbow_right": {
            "target": 150.0, "std": 20.0,
            "look": "near", "weight": 3,
            "alert_min": 110.0,
            "ref": "Knudson2001",
            "note": "Coude semi-étendu à l'impact"
        },
        "trunk_rotation": {
            "target": 80.0, "std": 20.0,
            "look": "near", "weight": 2,
            "ref": "Knudson2001",
            "note": "Rotation tronc complète vers l'avant à l'impact"
        },
        "hip_right": {
            "target": 150.0, "std": 20.0,
            "look": "near", "weight": 1,
            "ref": "Creveaux2009",
            "note": "Hanche — différencie stance ouvert vs neutre"
        },
    },
}

# ─────────────────────────────────────────────
# BACKHAND 1 MAIN
# ─────────────────────────────────────────────

BACKHAND_1H_NORMATIVES: Dict[str, Dict] = {

    # ── Preparation ──────────────────────────
    "preparation": {
        "trunk_rotation": {
            "target": 90.0, "std": 20.0,
            "look": "max", "weight": 3,
            "alert_min": 50.0,
            "ref": "Elliott2008",
            "note": "Rotation tronc vers l'arrière — initie la chaîne cinétique"
        },
        "elbow_right": {
            "target": 120.0, "std": 20.0,
            "look": "near", "weight": 2,
            "ref": "Elliott2008",
            "note": "Coude fléchi en backswing"
        },
    },

    # ── Ball Impact ──────────────────────────
    "ball_impact": {
        "elbow_right": {
            "target": 40.0, "std": 10.0,
            "look": "near", "weight": 3,
            "alert_min": 25.0, "alert_max": 60.0,
            "ref": "Elliott2008",
            "note": "Flexion coude 35–46° à l'impact (1 main)"
        },
        "trunk_rotation": {
            "target": 60.0, "std": 15.0,
            "look": "near", "weight": 2,
            "alert_min": 40.0,
            "ref": "Elliott2008",
            "note": "Rotation tronc vers l'avant — transfert énergie"
        },
        # Poignet droit — indicateur clé épicondylite
        # MediaPipe ne calcule pas directement l'extension du poignet
        # → approximé via l'angle coude-poignet-métacarpe (si disponible)
        # → pour l'instant : surveillé via la position relative du poignet
    },

    # ── Follow Through ───────────────────────
    "follow_through": {
        "shoulder_elevation_right": {
            "target": 100.0, "std": 20.0,
            "look": "near", "weight": 2,
            "ref": "Elliott2008",
            "note": "Élévation épaule en suivi de frappe"
        },
        "trunk_rotation": {
            "target": 120.0, "std": 20.0,
            "look": "max", "weight": 3,
            "alert_min": 80.0,
            "ref": "Elliott2008",
            "note": "Rotation complète — indicateur de qualité technique"
        },
    },
}

# ─────────────────────────────────────────────
# BACKHAND 2 MAINS
# ─────────────────────────────────────────────

BACKHAND_2H_NORMATIVES: Dict[str, Dict] = {

    # ── Preparation ──────────────────────────
    "preparation": {
        "trunk_rotation": {
            "target": 90.0, "std": 20.0,
            "look": "max", "weight": 3,
            "alert_min": 50.0,
            "ref": "Elliott2008",
            "note": "Rotation tronc — identique au 1 main en préparation"
        },
        "elbow_right": {
            "target": 100.0, "std": 20.0,
            "look": "near", "weight": 2,
            "ref": "Elliott2008",
            "note": "Coude droit fléchi en backswing 2 mains"
        },
        "elbow_left": {
            "target": 100.0, "std": 20.0,
            "look": "near", "weight": 2,
            "ref": "Elliott2008",
            "note": "Coude gauche fléchi — bras non-dominant actif"
        },
    },

    # ── Ball Impact ──────────────────────────
    "ball_impact": {
        "elbow_right": {
            "target": 40.0, "std": 10.0,
            "look": "near", "weight": 3,
            "alert_min": 25.0, "alert_max": 60.0,
            "ref": "Elliott2008",
            "note": "Flexion coude droit à l'impact — réduit vs 1 main"
        },
        "elbow_left": {
            "target": 40.0, "std": 10.0,
            "look": "near", "weight": 3,
            "ref": "Elliott2008",
            "note": "Coude gauche — répartit la charge sur les 2 bras"
        },
        "trunk_rotation": {
            "target": 60.0, "std": 15.0,
            "look": "near", "weight": 2,
            "alert_min": 40.0,
            "ref": "Elliott2008",
            "note": "Rotation tronc à l'impact 2 mains"
        },
    },

    # ── Follow Through ───────────────────────
    "follow_through": {
        "shoulder_elevation_right": {
            "target": 90.0, "std": 20.0,
            "look": "near", "weight": 2,
            "ref": "Elliott2008",
            "note": "Élévation épaule droite en suivi 2 mains"
        },
        "shoulder_elevation_left": {
            "target": 90.0, "std": 20.0,
            "look": "near", "weight": 2,
            "ref": "Elliott2008",
            "note": "Élévation épaule gauche — symétrie en 2 mains"
        },
        "trunk_rotation": {
            "target": 110.0, "std": 20.0,
            "look": "max", "weight": 3,
            "alert_min": 70.0,
            "ref": "Elliott2008",
            "note": "Rotation complète — légèrement inférieure au 1 main"
        },
    },
}

# ─────────────────────────────────────────────
# ACCÈS UNIFIÉ
# ─────────────────────────────────────────────

NORMATIVES_BY_GESTURE = {
    "serve":        SERVE_NORMATIVES,
    "forehand":     FOREHAND_NORMATIVES,
    "backhand_1h":  BACKHAND_1H_NORMATIVES,
    "backhand_2h":  BACKHAND_2H_NORMATIVES,
}

# Écarts-types normatifs pour le calcul de qualité de signal
# Utilisés dans find_phases_gestures.py
NORMATIVE_STD_BY_JOINT = {
    # Service
    "knee_flexion_right":       9.7,
    "trunk_inclination":        7.1,
    "shoulder_rotation_right":  26.5,
    "shoulder_elevation_right": 16.9,
    "elbow_right":              15.9,
    # Forehand / Backhand
    "trunk_rotation":           20.0,
    "hip_right":                15.0,
    "elbow_left":               15.0,
    "shoulder_elevation_left":  16.9,
    "knee_flexion_left":        9.7,
}


def get_normatives(gesture: str, variant: str = None) -> Dict:
    """
    Retourne les normatives pour un geste donné.

    Args:
        gesture : "serve" | "forehand" | "backhand"
        variant : "1h" | "2h" (uniquement pour backhand)

    Returns:
        Dict des normatives par phase
    """
    if gesture == "backhand":
        key = f"backhand_{variant}" if variant in ("1h", "2h") else "backhand_1h"
    else:
        key = gesture

    if key not in NORMATIVES_BY_GESTURE:
        raise ValueError(f"Geste inconnu : '{gesture}'. "
                         f"Valeurs acceptées : serve, forehand, backhand")

    return NORMATIVES_BY_GESTURE[key]