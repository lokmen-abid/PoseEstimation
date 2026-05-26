"""
=============================================================================
IA/Serve — Pose Estimation Pipeline
Exemple 3 : Angles 3D (pose_world_landmarks) + Savitzky-Golay
Détection de phase : Option C — annotation manuelle des frames clés
API : MediaPipe Tasks (compatible Python 3.13 + mediapipe 0.10.35)
=============================================================================

Workflow en 2 passes :
  Passe 1 — exploration (sans phases, sans mongo) :
    python src/pipeline/pose_ex3_3d_savgol.py --video "video.mp4" --session_id "test01" --no_mongo

  Passe 2 — avec annotation :
    python src/pipeline/pose_ex3_3d_savgol.py --video "video.mp4" --session_id "test01"
        --phases "trophy_position=145,racket_low_point=210,ball_impact=240"
=============================================================================
"""

import argparse
import asyncio
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional

import cv2
import mediapipe as mp
import numpy as np
from dotenv import load_dotenv
from scipy.signal import savgol_filter

load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
PIPELINE_MODE    = "ex3_3d_savgol"
SAVGOL_WINDOW    = 11
SAVGOL_POLYORDER = 3
MIN_VISIBILITY   = 0.5
PHASE_WINDOW     = 15
MONGO_URI        = os.getenv("MONGODB_ATLAS_URI", "")
DB_NAME          = "postural_db"
TASK_MODEL_PATH  = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "pose_landmarker_heavy.task"))

KEYPOINT_MAP = {
    "left_shoulder":   11, "right_shoulder":  12,
    "left_elbow":      13, "right_elbow":     14,
    "left_wrist":      15, "right_wrist":     16,
    "left_hip":        23, "right_hip":       24,
    "left_knee":       25, "right_knee":      26,
    "left_ankle":      27, "right_ankle":     28,
    "left_heel":       29, "right_heel":      30,
    "left_foot_index": 31, "right_foot_index":32,
}

NORMATIVE_VALUES = {
    "knee_flexion_right":       {"mean": 64.5,  "std": 9.7,  "phase": "trophy_position",  "ref": "Gorce2024"},
    "trunk_inclination":        {"mean": 25.0,  "std": 7.1,  "phase": "trophy_position",  "ref": "Gorce2024"},
    "shoulder_rotation_right":  {"mean": 130.1, "std": 26.5, "phase": "racket_low_point", "ref": "Gorce2024"},
    "shoulder_elevation_right": {"mean": 110.7, "std": 16.9, "phase": "ball_impact",      "ref": "Gorce2024"},
    "elbow_right":              {"mean": 30.1,  "std": 15.9, "phase": "ball_impact",      "ref": "Gorce2024"},
}

# ─────────────────────────────────────────────
# GÉOMÉTRIE 3D
# ─────────────────────────────────────────────

def angle_3d(a: np.ndarray, vertex: np.ndarray, b: np.ndarray) -> Optional[float]:
    v1 = a - vertex
    v2 = b - vertex
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    return math.degrees(math.acos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))


def trunk_inclination_angle(mid_hip: np.ndarray, mid_shoulder: np.ndarray) -> Optional[float]:
    vec = mid_shoulder - mid_hip
    n   = np.linalg.norm(vec)
    if n < 1e-6:
        return None
    return math.degrees(math.acos(np.clip(np.dot(vec / n, np.array([0., -1., 0.])), -1., 1.)))

# ─────────────────────────────────────────────
# KEYPOINTS
# ─────────────────────────────────────────────

def extract_keypoints(world_landmarks) -> Dict:
    """Extrait les keypoints 3D depuis pose_world_landmarks (pour calcul des angles)."""
    kps = {}
    for name, idx in KEYPOINT_MAP.items():
        lm = world_landmarks[idx]
        kps[name] = np.array([lm.x, lm.y, lm.z]) if lm.visibility >= MIN_VISIBILITY else None
    lh, rh = kps.get("left_hip"), kps.get("right_hip")
    ls, rs  = kps.get("left_shoulder"), kps.get("right_shoulder")
    kps["mid_hip"]      = (lh + rh) / 2.0 if (lh is not None and rh is not None) else None
    kps["mid_shoulder"] = (ls + rs) / 2.0 if (ls is not None and rs is not None) else None
    return kps


def extract_image_keypoints(image_landmarks) -> Dict:
    """
    Extrait les coordonnées image normalisées (0.0 → 1.0) depuis pose_landmarks.
    Utilisées UNIQUEMENT pour l'affichage du squelette dans visualize_pose.py.
    N'affecte pas les calculs d'angles (qui utilisent pose_world_landmarks).
    """
    image_kps = {}
    for name, idx in KEYPOINT_MAP.items():
        lm = image_landmarks[idx]
        if lm.visibility >= MIN_VISIBILITY:
            image_kps[name] = {
                "x":          float(lm.x),
                "y":          float(lm.y),
                "visibility": float(lm.visibility)
            }
        else:
            image_kps[name] = None
    return image_kps


def keypoints_to_dict(kps: Dict) -> Dict:
    return {
        name: {"x": float(v[0]), "y": float(v[1]), "z": float(v[2]), "visibility": 1.0}
        if isinstance(v, np.ndarray) else None
        for name, v in kps.items()
    }

# ─────────────────────────────────────────────
# ANGLES
# ─────────────────────────────────────────────

def compute_angles(kps: Dict) -> Dict:
    angles = {}

    def safe(name, a, v, b):
        angles[name] = angle_3d(a, v, b) if all(x is not None for x in [a, v, b]) else None

    safe("knee_flexion_right", kps.get("right_hip"),      kps.get("right_knee"),  kps.get("right_ankle"))
    safe("knee_flexion_left",  kps.get("left_hip"),       kps.get("left_knee"),   kps.get("left_ankle"))
    safe("elbow_right",        kps.get("right_shoulder"), kps.get("right_elbow"), kps.get("right_wrist"))
    safe("elbow_left",         kps.get("left_shoulder"),  kps.get("left_elbow"),  kps.get("left_wrist"))
    safe("hip_right",          kps.get("right_shoulder"), kps.get("right_hip"),   kps.get("right_knee"))
    safe("hip_left",           kps.get("left_shoulder"),  kps.get("left_hip"),    kps.get("left_knee"))
    safe("shoulder_rotation_right",
         kps.get("right_elbow"), kps.get("right_shoulder"), kps.get("right_hip"))

    ms = kps.get("mid_shoulder")
    angles["shoulder_elevation_right"] = (
        angle_3d(kps["right_elbow"], kps["right_shoulder"], ms)
        if ms is not None and kps.get("right_shoulder") is not None
        and kps.get("right_elbow") is not None else None)
    angles["shoulder_elevation_left"] = (
        angle_3d(kps["left_elbow"], kps["left_shoulder"], ms)
        if ms is not None and kps.get("left_shoulder") is not None
        and kps.get("left_elbow") is not None else None)

    mh, ms2 = kps.get("mid_hip"), kps.get("mid_shoulder")
    angles["trunk_inclination"] = (
        trunk_inclination_angle(mh, ms2) if mh is not None and ms2 is not None else None)

    lh, rh = kps.get("left_hip"), kps.get("right_hip")
    ls, rs  = kps.get("left_shoulder"), kps.get("right_shoulder")
    if all(v is not None for v in [lh, rh, ls, rs]):
        hv = np.array([rh[0]-lh[0], 0., rh[2]-lh[2]])
        sv = np.array([rs[0]-ls[0], 0., rs[2]-ls[2]])
        nh, ns = np.linalg.norm(hv), np.linalg.norm(sv)
        angles["trunk_rotation"] = (
            math.degrees(math.acos(np.clip(np.dot(hv/nh, sv/ns), -1., 1.)))
            if nh > 1e-6 and ns > 1e-6 else None)
    else:
        angles["trunk_rotation"] = None

    return angles

# ─────────────────────────────────────────────
# SAVITZKY-GOLAY
# ─────────────────────────────────────────────

def apply_savgol(angle_series: List[Dict]) -> List[Dict]:
    if len(angle_series) < SAVGOL_WINDOW:
        print(f"[WARN] Trop peu de frames ({len(angle_series)}) pour SavGol — retour brut")
        return angle_series

    keys            = list(angle_series[0].keys())
    smoothed_series = [{k: None for k in keys} for _ in angle_series]

    for key in keys:
        raw = [a.get(key) for a in angle_series]
        idx = [i for i, v in enumerate(raw) if v is not None]
        if len(idx) < SAVGOL_WINDOW:
            continue
        full = np.interp(range(len(raw)), idx, [raw[i] for i in idx])
        try:
            smoothed = savgol_filter(full, window_length=SAVGOL_WINDOW,
                                     polyorder=SAVGOL_POLYORDER)
        except ValueError:
            smoothed = full
        for i, v in enumerate(raw):
            smoothed_series[i][key] = float(smoothed[i]) if v is not None else None

    return smoothed_series

# ─────────────────────────────────────────────
# PHASES — Option C (annotation manuelle)
# ─────────────────────────────────────────────

def parse_phase_annotations(phases_str: Optional[str]) -> Dict[str, int]:
    if not phases_str:
        return {}
    result = {}
    for item in phases_str.split(","):
        if "=" not in item:
            continue
        name, frame_str = item.strip().split("=", 1)
        try:
            result[name.strip()] = int(frame_str.strip())
        except ValueError:
            pass
    return result


def assign_phases_to_frames(frames_data: List[Dict],
                             annotations: Dict[str, int]) -> List[Dict]:
    if not annotations:
        return frames_data
    for doc in frames_data:
        fn = doc["frame_number"]
        closest, min_dist = None, float("inf")
        for phase, target in annotations.items():
            dist = abs(fn - target)
            if dist <= PHASE_WINDOW and dist < min_dist:
                min_dist, closest = dist, phase
        doc["phase"] = closest
    return frames_data


def extract_phase_angles(smoothed: List[Dict], annotations: Dict[str, int]) -> Dict:
    phase_angles = {}
    for phase, target in annotations.items():
        start  = max(0, target - PHASE_WINDOW)
        end    = min(len(smoothed), target + PHASE_WINDOW + 1)
        window = smoothed[start:end]
        if not window:
            continue
        phase_angles[phase] = {}
        for key in window[0].keys():
            vals = [a[key] for a in window if a.get(key) is not None]
            if vals:
                arr = np.array(vals)
                phase_angles[phase][key] = {
                    "mean": float(np.mean(arr)), "std": float(np.std(arr)),
                    "min":  float(np.min(arr)),  "max": float(np.max(arr)),
                }
    return phase_angles

# ─────────────────────────────────────────────
# DOWNLOAD
# ─────────────────────────────────────────────

def download_youtube(url: str, output_dir: str) -> str:
    out = os.path.join(output_dir, "input_video_ex3.mp4")
    cmd = ["yt-dlp", "-f", "best[ext=mp4]/best", "-o", out, "--no-playlist", url]
    print(f"[yt-dlp] Téléchargement : {url}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"yt-dlp error:\n{r.stderr}")
    for f in os.listdir(output_dir):
        if "ex3" in f and f.endswith(".mp4"):
            return os.path.abspath(os.path.join(output_dir, f))
    raise RuntimeError("Aucun fichier mp4 trouvé")

# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def aggregate_metrics(smoothed: List[Dict], annotations: Dict,
                      phase_angles: Dict, session_id: str) -> Dict:
    joint_metrics = {}
    for key in (smoothed[0].keys() if smoothed else []):
        vals = [a[key] for a in smoothed if a.get(key) is not None]
        if vals:
            arr = np.array(vals)
            joint_metrics[key] = {
                "min":  float(np.min(arr)), "max":  float(np.max(arr)),
                "mean": float(np.mean(arr)), "std": float(np.std(arr)),
            }
    norm_cmp = {}
    for joint, norm in NORMATIVE_VALUES.items():
        measured = None
        source   = "session_mean"
        tp = norm["phase"]
        if tp in phase_angles and joint in phase_angles[tp]:
            measured = phase_angles[tp][joint]["mean"]
            source   = f"phase:{tp}"
        elif joint in joint_metrics:
            measured = joint_metrics[joint]["mean"]
        if measured is not None:
            delta = measured - norm["mean"]
            norm_cmp[joint] = {
                "measured_mean":  measured,
                "normative_mean": norm["mean"],
                "normative_std":  norm["std"],
                "delta_degrees":  round(delta, 2),
                "within_1std":    abs(delta) <= norm["std"],
                "reference":      norm["ref"],
                "phase":          norm["phase"],
                "source":         source,
            }
    return {
        "session_id":           session_id,
        "gesture_type":         "serve",
        "pipeline_mode":        PIPELINE_MODE,
        "total_frames":         len(smoothed),
        "phases_detected":      annotations,
        "phase_angles":         phase_angles,
        "joint_metrics":        joint_metrics,
        "normative_comparison": norm_cmp,
        "alerts":               [],
        "computed_at":          datetime.now(timezone.utc).isoformat(),
    }

# ─────────────────────────────────────────────
# MONGODB
# ─────────────────────────────────────────────

async def save_to_mongodb(frames_data: List[Dict], metrics: Dict):
    try:
        import motor.motor_asyncio
    except ImportError:
        print("[WARN] motor non installé"); return
    if not MONGO_URI:
        print("[WARN] MONGODB_ATLAS_URI non défini"); return
    client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
    db = client[DB_NAME]
    if frames_data:
        await db["frames"].insert_many(frames_data)
    await db["metrics"].insert_one(metrics)
    client.close()
    print("[MongoDB] Sauvegarde terminée.")

# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────

async def run_pipeline(video_path: str, session_id: str,
                       phase_annotations: Dict[str, int],
                       save_mongo: bool = True):

    if not os.path.exists(TASK_MODEL_PATH):
        raise RuntimeError(f"Modèle introuvable : {TASK_MODEL_PATH}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir : {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Pipeline Ex3] {total_frames} frames @ {fps:.1f} FPS")

    if phase_annotations:
        print(f"[Pipeline Ex3] Phases annotées : {phase_annotations}")
    else:
        print("[Pipeline Ex3] Passe 1 — exploration sans annotation")
        print("  → Identifie les frames clés dans le JSON, puis relance avec --phases")

    BaseOptions           = mp.tasks.BaseOptions
    PoseLandmarker        = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode     = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=TASK_MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    raw_frames_data: List[Dict] = []
    raw_angles_list: List[Dict] = []

    # ── Passe 1 : extraction brute ──
    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_number = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int(frame_number * (1000.0 / fps))
            mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB,
                                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result       = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_world_landmarks:
                # ── Calcul des angles (world landmarks — inchangé) ──
                kps        = extract_keypoints(result.pose_world_landmarks[0])
                raw_angles = compute_angles(kps)
                raw_angles_list.append(raw_angles)

                # ── Coordonnées image pour le squelette (n'affecte pas les angles) ──
                image_kps = {}
                if result.pose_landmarks:
                    image_kps = extract_image_keypoints(result.pose_landmarks[0])

                raw_frames_data.append({
                    "session_id":    session_id,
                    "frame_number":  frame_number,
                    "timestamp_ms":  timestamp_ms,
                    "phase":         None,
                    "pipeline_mode": PIPELINE_MODE,
                    "keypoints":     keypoints_to_dict(kps),  # world coords (angles)
                    "image_kps":     image_kps,               # image coords (squelette)
                    "angles":        {},                       # rempli après SavGol
                })

            if frame_number % 50 == 0:
                print(f"  Extraction frame {frame_number}/{total_frames}...")

            frame_number += 1

    cap.release()
    print(f"[Ex3] Extraction terminée : {len(raw_angles_list)} frames valides")

    # ── Post-traitement : Savitzky-Golay ──
    print(f"[Ex3] Application Savitzky-Golay (window={SAVGOL_WINDOW}, poly={SAVGOL_POLYORDER})...")
    smoothed_angles = apply_savgol(raw_angles_list)

    for i, doc in enumerate(raw_frames_data):
        doc["angles"] = {k: (round(v, 3) if v is not None else None)
                         for k, v in smoothed_angles[i].items()}

    # ── Phases + métriques ──
    frames_data  = assign_phases_to_frames(raw_frames_data, phase_annotations)
    phase_angles = extract_phase_angles(smoothed_angles, phase_annotations)
    metrics      = aggregate_metrics(smoothed_angles, phase_annotations,
                                     phase_angles, session_id)

    # ── Résumé console ──
    print("\n" + "="*60)
    print(f"RÉSUMÉ — {PIPELINE_MODE}")
    print("="*60)
    print(f"Frames analysées : {metrics['total_frames']}")
    print(f"Phases annotées  : {phase_annotations or 'aucune'}")

    if phase_angles:
        print("\nAngles aux phases clés (SavGol) :")
        for phase_name, joints in phase_angles.items():
            print(f"\n  [{phase_name}]")
            for joint, stats in joints.items():
                print(f"    {joint:35s} : {stats['mean']:6.1f}° ± {stats['std']:4.1f}°")

    print("\nComparaison normative (Gorce 2024) :")
    for joint, comp in metrics.get("normative_comparison", {}).items():
        status = "✓ dans 1σ" if comp["within_1std"] else "✗ hors 1σ"
        print(f"  {joint:35s} : mesuré={comp['measured_mean']:6.1f}° | "
              f"normative={comp['normative_mean']:5.1f}±{comp['normative_std']:4.1f}° | "
              f"Δ={comp['delta_degrees']:+.1f}° | {status} [{comp.get('source','')}]")

    # ── Export JSON métriques ──
    out_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "../../output/results"))
    os.makedirs(out_dir, exist_ok=True)

    out_json = os.path.join(out_dir, f"results_{PIPELINE_MODE}_{session_id}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n[Output] JSON : {out_json}")

    # ── Export JSON frames (angles + image_kps pour visualisation) ──
    frames_json = os.path.join(out_dir, f"frames_angles_{session_id}.json")
    with open(frames_json, "w", encoding="utf-8") as f:
        json.dump([
            {
                "frame_number": d["frame_number"],
                "angles":       d["angles"],
                "keypoints":    d["keypoints"],  # world coords (pour angles)
                "image_kps":    d["image_kps"],  # image coords (pour squelette)
            }
            for d in frames_data
        ], f, ensure_ascii=False, indent=2)
    print(f"[Output] Frames angles : {frames_json}")

    if save_mongo:
        await save_to_mongodb(frames_data, metrics)
    return metrics

# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--youtube", type=str)
    g.add_argument("--video",   type=str)
    p.add_argument("--session_id", required=True)
    p.add_argument("--phases", type=str, default=None,
                   help="Ex: 'trophy_position=145,racket_low_point=210,ball_impact=240'")
    p.add_argument("--no_mongo", action="store_true")
    return p.parse_args()


async def main():
    args              = parse_args()
    phase_annotations = parse_phase_annotations(args.phases)

    if args.youtube:
        tmpdir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "../../output/tmp_video"))
        os.makedirs(tmpdir, exist_ok=True)
        video_path = download_youtube(args.youtube, tmpdir)
    else:
        if not os.path.exists(args.video):
            sys.exit(f"Fichier introuvable : {args.video}")
        video_path = args.video

    await run_pipeline(video_path, args.session_id,
                       phase_annotations, save_mongo=not args.no_mongo)


if __name__ == "__main__":
    asyncio.run(main())