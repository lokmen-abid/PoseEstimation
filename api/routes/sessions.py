from fastapi import APIRouter, HTTPException, Depends, Query, UploadFile, File
from pydantic import BaseModel, Field
from typing import Optional, Dict, Literal
from api.models import Session, Athlete, Frame, Metrics
from api.auth import get_current_user

router = APIRouter(tags=["Sessions"])

# ── Constantes ──────────────────────────────────────────────

GESTURE_TYPES = Literal["service", "coup_droit", "revers"]

STATUS_TRANSITIONS = {
    "created":    ["processing"],
    "processing": ["completed", "error"],
    "completed":  ["processing"],
    "error":      ["processing"],   # permettre un re-run
}

# ── Schémas ─────────────────────────────────────────────────

class SessionCreate(BaseModel):
    """
    Payload de création d'une session.

    phase_annotations : frames clés annotées manuellement par le spécialiste.
      Ces annotations sont transmises telles quelles au pipeline IA
      (pose_ex3_3d_savgol.py --phases "trophy_position=145,racket_low_point=210,...").
      Elles peuvent être nulles à la création et enrichies ultérieurement
      via PUT /sessions/{id}/annotations.

    Exemple :
      {
        "athlete_id": "664abc...",
        "gesture_type": "service",
        "fps": 30,
        "phase_annotations": {
          "trophy_position": 145,
          "racket_low_point": 210,
          "ball_impact": 240
        }
      }
    """
    athlete_id:        str              = Field(..., description="ID Beanie de l'athlète")
    gesture_type:      GESTURE_TYPES    = Field(..., description="Type de geste analysé")
    fps:               int              = Field(30, ge=1, le=240, description="FPS de la vidéo source")
    phase_annotations: Optional[Dict[str, int]] = Field(
        None,
        description=(
            "Frames clés annotées manuellement. "
            "Clés attendues selon le geste : "
            "service → trophy_position, racket_low_point, ball_impact ; "
            "coup_droit → preparation, acceleration, follow_through ; "
            "revers → preparation, racket_low_point, ball_impact"
        )
    )


class SessionUpdate(BaseModel):
    """
    Mise à jour partielle — seules les annotations et le fps sont modifiables.
    Le status ne se modifie pas directement par le spécialiste.
    """
    fps:               Optional[int]             = Field(None, ge=1, le=240)
    phase_annotations: Optional[Dict[str, int]]  = Field(None)


class SessionResponse(BaseModel):
    id:                str
    athlete_id:        str
    specialist_id:     str
    gesture_type:      str
    status:            str
    fps:               int
    total_frames:      Optional[int]
    video_url:         str
    phase_annotations: Optional[Dict[str, int]]
    created_at:        str


# ── Helpers ─────────────────────────────────────────────────

def _serialize(s: Session) -> dict:
    """Sérialise une session en dict JSON-safe."""
    return {
        "id":                str(s.id),
        "athlete_id":        s.athlete_id,
        "specialist_id":     s.specialist_id,
        "gesture_type":      s.gesture_type,
        "status":            s.status,
        "fps":               s.fps,
        "total_frames":      s.total_frames,
        "video_url":         s.video_url,
        "phase_annotations": s.phase_annotations,
        "created_at":        s.created_at.isoformat() if s.created_at else None,
    }


async def _get_session_or_403(session_id: str, specialist_id: str) -> Session:
    """
    Récupère la session et vérifie l'appartenance au spécialiste connecté.
    Lève 404 si introuvable, 403 si cross-specialist.
    """
    session = await Session.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session introuvable")
    if session.specialist_id != specialist_id:
        raise HTTPException(status_code=403, detail="Accès refusé")
    return session


async def _get_athlete_or_403(athlete_id: str, specialist_id: str) -> Athlete:
    """
    Double isolation : vérifie que l'athlète appartient bien au spécialiste
    avant de créer ou lister des sessions sur cet athlète.
    """
    athlete = await Athlete.get(athlete_id)
    if not athlete:
        raise HTTPException(status_code=404, detail="Athlète introuvable")
    if athlete.specialist_id != specialist_id:
        raise HTTPException(status_code=403, detail="Accès refusé — athlète d'un autre spécialiste")
    return athlete


# ── Endpoints CRUD ──────────────────────────────────────────

@router.post(
    "/",
    summary="Créer une session",
    status_code=201,
)
async def create_session(
    data: SessionCreate,
    current_user=Depends(get_current_user),
):
    """
    Crée une session liée à un athlète du spécialiste connecté.

    - Vérifie que l'athlète appartient au spécialiste (double isolation).
    - La session est créée avec status='created' et video_url vide.
    - Les phase_annotations peuvent être fournies dès la création
      ou ajoutées plus tard via PUT /sessions/{id}/annotations.
    """
    await _get_athlete_or_403(data.athlete_id, str(current_user.id))

    session = Session(
        athlete_id=data.athlete_id,
        specialist_id=str(current_user.id),
        gesture_type=data.gesture_type,
        video_url="",           # sera rempli par POST /sessions/{id}/upload
        status="created",
        fps=data.fps,
        phase_annotations=data.phase_annotations,
    )
    await session.insert()
    return {
        "message": "Session créée avec succès",
        "session": _serialize(session),
    }


@router.get(
    "/",
    summary="Lister mes sessions (paginé)",
)
async def list_sessions(
    skip:         int            = Query(default=0,  ge=0),
    limit:        int            = Query(default=20, ge=1, le=100),
    gesture_type: Optional[str]  = Query(default=None, description="Filtrer par type de geste"),
    status:       Optional[str]  = Query(default=None, description="Filtrer par statut"),
    current_user=Depends(get_current_user),
):
    """
    Retourne toutes les sessions du spécialiste connecté.
    Filtres optionnels : gesture_type, status.
    """
    query = Session.find(Session.specialist_id == str(current_user.id))

    if gesture_type:
        query = query.find(Session.gesture_type == gesture_type)
    if status:
        query = query.find(Session.status == status)

    total   = await query.count()
    sessions = await query.skip(skip).limit(limit).to_list()

    return {
        "total": total,
        "skip":  skip,
        "limit": limit,
        "data":  [_serialize(s) for s in sessions],
    }


@router.get(
    "/{session_id}",
    summary="Détail d'une session",
)
async def get_session(
    session_id: str,
    current_user=Depends(get_current_user),
):
    """
    Retourne le détail complet d'une session, y compris les phase_annotations
    et le statut courant du pipeline IA.
    """
    session = await _get_session_or_403(session_id, str(current_user.id))
    return _serialize(session)


@router.put(
    "/{session_id}",
    summary="Modifier fps ou annotations de phase",
)
async def update_session(
    session_id: str,
    data:       SessionUpdate,
    current_user=Depends(get_current_user),
):
    """
    Modification partielle : fps et/ou phase_annotations.

    Point critique (rapport de synthèse) :
    Le pipeline IA reçoit les phases via --phases "trophy_position=145,...".
    Cet endpoint permet au spécialiste de saisir ces annotations depuis
    le frontend React après une première passe d'exploration (--no_mongo),
    puis de lancer l'analyse complète via POST /sessions/{id}/analyze.
    """
    session = await _get_session_or_403(session_id, str(current_user.id))

    if data.fps               is not None: session.fps               = data.fps
    if data.phase_annotations is not None: session.phase_annotations = data.phase_annotations

    await session.save()
    return {
        "message": "Session mise à jour",
        "session": _serialize(session),
    }


@router.delete(
    "/{session_id}",
    summary="Supprimer une session (cascade frames + metrics)",
)
async def delete_session(
    session_id: str,
    current_user=Depends(get_current_user),
):
    """
    Supprime la session et tous les documents liés en cascade :
      - Collection frames  (keypoints par frame)
      - Collection metrics (résumé clinique)

    La suppression est irréversible.
    """
    session = await _get_session_or_403(session_id, str(current_user.id))

    # Cascade
    await Frame.find(Frame.session_id   == session_id).delete()
    await Metrics.find(Metrics.session_id == session_id).delete()

    athlete_name = session.gesture_type
    await session.delete()

    return {
        "message": f"Session ({athlete_name}) et toutes les données associées supprimées."
    }


# ── Endpoint par athlète ─────────────────────────────────────

@router.get(
    "/athlete/{athlete_id}",
    summary="Historique des sessions d'un athlète",
)
async def get_sessions_by_athlete(
    athlete_id: str,
    skip:  int = Query(default=0,  ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    current_user=Depends(get_current_user),
):
    """
    Liste toutes les sessions d'un athlète spécifique.
    Vérifie que l'athlète appartient bien au spécialiste connecté.
    """
    await _get_athlete_or_403(athlete_id, str(current_user.id))

    query    = Session.find(Session.athlete_id == athlete_id)
    total    = await query.count()
    sessions = await query.skip(skip).limit(limit).to_list()

    return {
        "total": total,
        "skip":  skip,
        "limit": limit,
        "data":  [_serialize(s) for s in sessions],
    }


# ── Upload vidéo ─────────────────────────────────────────────

@router.post(
    "/{session_id}/upload",
    summary="Uploader la vidéo d'une session",
)
async def upload_video(
    session_id: str,
    file: UploadFile = File(..., description="Fichier vidéo (mp4, mov, avi)"),
    current_user=Depends(get_current_user),
):
    """
    Upload la vidéo liée à une session et met à jour video_url.

    Phase actuelle : sauvegarde locale dans output/tmp_video/.
    Phase 4 : remplacer par un upload vers S3 / Cloudflare R2
    et stocker l'URL cloud dans video_url.

    La session passe en status='processing' dès que la vidéo est reçue.
    """
    import os, aiofiles

    session = await _get_session_or_403(session_id, str(current_user.id))

    ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
    ext = os.path.splitext(file.filename or "")[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Format non supporté : {ext}. Formats acceptés : {ALLOWED_EXTENSIONS}",
        )

    save_dir  = os.path.abspath(os.path.join("output", "tmp_video"))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{session_id}{ext}")

    async with aiofiles.open(save_path, "wb") as out:
        content = await file.read()
        await out.write(content)

    session.video_url = save_path
    session.status    = "processing"
    await session.save()

    return {
        "message":   "Vidéo uploadée avec succès",
        "video_url": save_path,
        "status":    session.status,
    }


# ── Lancement pipeline IA ────────────────────────────────────

@router.post(
    "/{session_id}/analyze",
    summary="Lancer l'analyse IA (pipeline pose_ex3_3d_savgol)",
)
async def analyze_session(
    session_id: str,
    current_user=Depends(get_current_user),
):
    """
    Déclenche le pipeline IA sur la vidéo uploadée.

    Prérequis :
      - La session doit avoir un video_url non vide (POST /upload effectué).
      - phase_annotations recommandées pour une analyse par phase correcte.
        Sans annotations, le pipeline tourne en mode exploration (passe 1).
        Avec annotations, il effectue la comparaison normative complète (passe 2).

    Point critique (rapport de synthèse) :
      La comparaison session_mean vs normative_phase était méthodologiquement
      incorrecte. Le pipeline Ex3 compare uniquement les angles mesurés
      dans la fenêtre de phase (PHASE_WINDOW=15 frames autour de la frame annotée),
      pas la moyenne globale de la session. Les phase_annotations sont donc
      indispensables pour des résultats cliniquement valides.

    Phase actuelle (Phase 3) :
      Lance le pipeline de façon synchrone via subprocess pour la démo.
      Le status passe à 'processing' puis 'completed' ou 'error'.

    Phase 4 :
      Remplacer subprocess par asyncio.create_task() ou
      FastAPI BackgroundTasks pour un traitement non-bloquant.
    """
    import subprocess, sys, os

    session = await _get_session_or_403(session_id, str(current_user.id))

    if not session.video_url:
        raise HTTPException(
            status_code=422,
            detail="Aucune vidéo uploadée. Utilisez d'abord POST /sessions/{id}/upload",
        )

    if session.status not in ("processing","completed", "error"):
        raise HTTPException(
            status_code=422,
            detail=f"Impossible de lancer l'analyse depuis le statut '{session.status}'",
        )

    # Construction de l'argument --phases depuis les annotations stockées
    phases_arg = None
    if session.phase_annotations:
        phases_arg = ",".join(
            f"{k}={v}" for k, v in session.phase_annotations.items()
        )
        # Ex : "trophy_position=145,racket_low_point=210,ball_impact=240"

    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    pipeline_script = os.path.join(project_root, "src", "pipeline", "pose_ex3_3d_savgol.py")

    cmd = [
        sys.executable,
        pipeline_script,
        "--video", session.video_url,
        "--session_id", session_id,
    ]
    if phases_arg:
        cmd += ["--phases", phases_arg]
    else:
        cmd += ["--no_mongo"]

    session.status = "processing"
    await session.save()

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=600,
            encoding="utf-8",
            errors="replace",
            cwd=project_root,
            env=env,
        )

        stderr_tail = (result.stderr or "")[-500:]
        stdout_tail = (result.stdout or "")[-200:]

        if result.returncode == 0:
            session.status = "completed"
            await session.save()
            return {
                "message": "Analyse terminee avec succes",
                "session_id": session_id,
                "phases_used": session.phase_annotations,
                "has_annotations": phases_arg is not None,
                "hint": (
                    None if phases_arg
                    else "Analyse exploratoire terminee. Annotez les frames cles puis relancez."
                ),
            }
        else:
            session.status = "error"
            await session.save()
            detail = stderr_tail or stdout_tail or "Erreur pipeline inconnue"
            raise HTTPException(status_code=500, detail=f"Erreur pipeline : {detail}")

    except subprocess.TimeoutExpired:
        session.status = "error"
        await session.save()
        raise HTTPException(status_code=504, detail="Timeout : pipeline > 10 min")


# ── Résultats ────────────────────────────────────────────────

@router.get(
    "/{session_id}/results",
    summary="Métriques et alertes cliniques d'une session",
)
async def get_session_results(
    session_id: str,
    current_user=Depends(get_current_user),
):
    """
    Retourne les métriques calculées par le pipeline IA :
      - joint_metrics : min/max/mean/std par articulation
      - normative_comparison : écart vs normatives (Gorce 2024, Elliott 2008)
      - alerts : alertes cliniques (epicondylite, surcharge lombaire, asymétrie)
      - phases_detected : frames clés annotées utilisées

    Retourne 404 si l'analyse n'a pas encore été lancée ou est en cours.
    """
    session = await _get_session_or_403(session_id, str(current_user.id))

    if session.status != "completed":
        raise HTTPException(
            status_code=404,
            detail=(
                f"Résultats indisponibles — statut actuel : '{session.status}'. "
                "Attendez la fin de l'analyse (status='completed')."
            ),
        )

    metrics = await Metrics.find_one(Metrics.session_id == session_id)
    if not metrics:
        raise HTTPException(
            status_code=404,
            detail="Métriques introuvables en base. Relancez l'analyse.",
        )

    return {
        "session_id":           session_id,
        "gesture_type":         metrics.gesture_type,
        "pipeline_mode":        metrics.pipeline_mode,
        "total_frames":         metrics.total_frames,
        "phases_detected":      metrics.phases_detected,
        "phase_annotations":    session.phase_annotations,
        "joint_metrics":        {
            k: {
                "min":  v.min,
                "max":  v.max,
                "mean": v.mean,
                "std":  v.std,
            }
            for k, v in metrics.joint_metrics.items()
        },
        "normative_comparison": metrics.normative_comparison,
        "alerts": [
            {
                "joint":     a.joint,
                "value":     a.value,
                "threshold": a.threshold,
                "reference": a.reference,
                "severity":  a.severity,
            }
            for a in metrics.alerts
        ],
        "computed_at": metrics.computed_at.isoformat() if metrics.computed_at else None,
    }


@router.get(
    "/{session_id}/candidates",
    summary="Frames candidates pour annotation (screenshots + scores)",
)
async def get_phase_candidates(
    session_id: str,
    current_user=Depends(get_current_user),
):
    """
    Analyse le JSON frames_angles_{session_id}.json produit par la Passe 1
    et retourne pour chaque phase :
      - frame suggérée (meilleure combinaison temporellement cohérente)
      - top 3 candidats avec score et confiance
      - screenshot base64 avec squelette dessiné (OpenCV)
      - angles clés à cette frame

    Prérequis : Passe 1 effectuée (frames_angles_{id}.json doit exister).
    """
    import os, sys, base64, cv2, numpy as np

    session = await _get_session_or_403(session_id, str(current_user.id))

    # ── Chemins ──────────────────────────────────────────────
    project_root  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    pipeline_dir  = os.path.join(project_root, "src", "pipeline")
    results_dir   = os.path.join(project_root, "output", "results")
    frames_json   = os.path.join(results_dir, f"frames_angles_{session_id}.json")

    if not os.path.exists(frames_json):
        raise HTTPException(
            status_code=404,
            detail=(
                "Fichier frames_angles introuvable. "
                "Lancez d'abord la Passe 1 via POST /sessions/{id}/analyze."
            ),
        )

    # ── Import pipeline (chemin absolu) ──────────────────────
    if pipeline_dir not in sys.path:
        sys.path.insert(0, pipeline_dir)

    try:
        from find_phases_gestures import (
            load_frames, detect_fps, adapt_to_fps,
            find_candidate_frames, select_best_combination,
            PHASE_ZONES, get_key_angles_str,
        )
        from normatives import get_normatives
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Import pipeline échoué : {e}")

    # ── Chargement frames JSON ────────────────────────────────
    try:
        frames = load_frames(frames_json)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lecture frames JSON échouée : {e}")

    if not frames:
        raise HTTPException(status_code=422, detail="JSON frames vide.")

    # ── Mapping geste frontend → pipeline ────────────────────
    GESTURE_MAP = {"service": "serve", "coup_droit": "forehand", "revers": "backhand"}
    gesture = GESTURE_MAP.get(session.gesture_type, "serve")

    # ── Détection FPS + candidats ─────────────────────────────
    fps          = detect_fps(frames)
    local_window, min_frames = adapt_to_fps(fps)
    normatives   = get_normatives(gesture)
    zones        = PHASE_ZONES.get(gesture, {})
    candidates   = find_candidate_frames(frames, gesture, normatives, zones, local_window)
    best, _      = select_best_combination(candidates, gesture, min_frames)

    # ── Lookup frame par numéro ───────────────────────────────
    frame_lookup = {f["frame_number"]: f for f in frames}

    # ── Extraction screenshots OpenCV ────────────────────────
    SKELETON_CONNECTIONS = [
        (11, 12), (11, 23), (12, 24), (23, 24),
        (11, 13), (13, 15), (12, 14), (14, 16),
        (23, 25), (25, 27), (24, 26), (26, 28),
    ]
    KEY_LANDMARKS = {11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28}
    KP_NAME_TO_IDX = {
        "left_shoulder": 11, "right_shoulder": 12,
        "left_elbow": 13,    "right_elbow": 14,
        "left_wrist": 15,    "right_wrist": 16,
        "left_hip": 23,      "right_hip": 24,
        "left_knee": 25,     "right_knee": 26,
        "left_ankle": 27,    "right_ankle": 28,
    }

    CONF_COLORS = {
        "HIGH":       (60, 210, 60),
        "MEDIUM":     (0,  200, 255),
        "LOW":        (0,  165, 255),
        "UNRELIABLE": (50,  50, 220),
    }

    def extract_frame_screenshot(video_path: str, frame_number: int,
                                  frame_data: dict, phase_name: str,
                                  confidence: str) -> str:
        """Extrait la frame de la vidéo, dessine le squelette, retourne base64."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return ""
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, img = cap.read()
        cap.release()
        if not ret:
            return ""

        h, w = img.shape[:2]
        image_kps = frame_data.get("image_kps", {})

        # Projeter keypoints
        positions = {}
        for name, idx in KP_NAME_TO_IDX.items():
            kp = image_kps.get(name)
            if kp:
                positions[idx] = (int(kp["x"] * w), int(kp["y"] * h))

        # Connexions
        for (i, j) in SKELETON_CONNECTIONS:
            if i in positions and j in positions:
                cv2.line(img, positions[i], positions[j], (220, 220, 220), 2, cv2.LINE_AA)

        # Points
        for idx, (px, py) in positions.items():
            color = (0, 200, 255) if idx in KEY_LANDMARKS else (0, 220, 160)
            cv2.circle(img, (px, py), 7, color, -1, cv2.LINE_AA)
            cv2.circle(img, (px, py), 8, (15, 15, 15), 1, cv2.LINE_AA)

        # Label phase + frame en haut
        conf_color = CONF_COLORS.get(confidence, (255, 255, 255))
        label_text = f"{phase_name.replace('_', ' ').upper()}  |  Frame {frame_number}  |  {confidence}"
        cv2.rectangle(img, (0, 0), (w, 30), (10, 10, 10), -1)
        cv2.putText(img, label_text, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 1, cv2.LINE_AA)

        # Redimensionner pour le frontend (max 640px de large)
        if w > 640:
            scale = 640 / w
            img = cv2.resize(img, (640, int(h * scale)), interpolation=cv2.INTER_AREA)

        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        return base64.b64encode(buf).decode("utf-8")

    # ── Construction réponse ──────────────────────────────────
    JOINT_SHORT = {
        "knee_flexion_right":       "Genou D",
        "trunk_inclination":        "Incl. tronc",
        "shoulder_rotation_right":  "Rot. épaule D",
        "shoulder_elevation_right": "Elv. épaule D",
        "elbow_right":              "Coude D",
        "knee_flexion_left":        "Genou G",
        "trunk_rotation":           "Rot. tronc",
        "hip_right":                "Hanche D",
        "elbow_left":               "Coude G",
    }

    has_video = bool(session.video_url and os.path.exists(session.video_url))

    result = {
        "session_id":  session_id,
        "gesture":     gesture,
        "total_frames": len(frames),
        "fps":         fps,
        "best":        best,          # {phase: frame_number} sélection optimale
        "has_video":   has_video,
        "phases":      {},
    }

    for phase_name, frame_list in candidates.items():
        top3 = frame_list[:3]
        best_fn   = best.get(phase_name) if best else (top3[0][0] if top3 else None)
        best_conf = top3[0][2] if top3 else "UNRELIABLE"

        # Screenshot de la frame suggérée
        screenshot_b64 = ""
        if has_video and best_fn is not None and best_fn in frame_lookup:
            screenshot_b64 = extract_frame_screenshot(
                session.video_url, best_fn,
                frame_lookup[best_fn], phase_name, best_conf
            )

        # Angles clés à cette frame
        key_angles = {}
        if best_fn is not None and best_fn in frame_lookup:
            angles = frame_lookup[best_fn].get("angles", {})
            phase_joints = list(normatives.get(phase_name, {}).keys())[:5]
            for joint in phase_joints:
                val = angles.get(joint)
                if val is not None:
                    key_angles[JOINT_SHORT.get(joint, joint)] = round(val, 1)

        result["phases"][phase_name] = {
            "suggested_frame": best_fn,
            "confidence":      best_conf,
            "screenshot_b64":  screenshot_b64,
            "key_angles":      key_angles,
            "top3": [
                {
                    "frame":      fn,
                    "score":      round(score, 3),
                    "confidence": conf,
                }
                for fn, score, conf in top3
            ],
        }

    return result


# ── Annotations seulement ───────────────────────────────────

@router.put(
    "/{session_id}/annotations",
    summary="Mettre à jour les annotations de phase uniquement",
)
async def update_phase_annotations(
    session_id: str,
    annotations: Dict[str, int],
    current_user=Depends(get_current_user),
):
    """
    Endpoint dédié à la saisie des frames clés par le spécialiste
    depuis le frontend React, après la passe d'exploration.

    Exemple de body :
      { "trophy_position": 145, "racket_low_point": 210, "ball_impact": 240 }

    Ces valeurs seront passées au pipeline via --phases lors du prochain
    appel à POST /sessions/{id}/analyze.
    """
    session = await _get_session_or_403(session_id, str(current_user.id))

    session.phase_annotations = annotations
    await session.save()

    return {
        "message":           "Annotations mises à jour",
        "phase_annotations": session.phase_annotations,
        "next_step":         f"POST /api/sessions/{session_id}/analyze pour lancer l'analyse complète",
    }