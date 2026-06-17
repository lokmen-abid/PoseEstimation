from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends, Query, UploadFile, File
from pydantic import BaseModel, Field
from typing import Optional, Dict, Literal
from api.models import Session, Athlete, Frame, Metrics
from api.auth import get_current_user
from fastapi.responses import StreamingResponse
import io
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
        "error_message":     s.error_message,
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


@router.get(
    "/athlete/{athlete_id}/evolution",
    summary="Évolution longitudinale des angles d'un athlète",
)
async def get_athlete_evolution(
        athlete_id: str,
        current_user=Depends(get_current_user),
):
    """
    Agrège les métriques de toutes les sessions 'completed' d'un athlète
    et retourne une série temporelle par articulation.

    Format retourné :
    {
      "athlete_id": "...",
      "sessions_count": 3,
      "series": [
        {
          "session_id": "...",
          "date": "2026-05-10",
          "gesture_type": "service",
          "joints": {
            "knee_flexion_right": { "mean": 130.2, "std": 3.4 },
            ...
          },
          "alerts_count": 2
        },
        ...
      ]
    }
    """
    await _get_athlete_or_403(athlete_id, str(current_user.id))

    # Toutes les sessions complétées, triées par date croissante
    sessions = await Session.find(
        Session.athlete_id == athlete_id,
        Session.status == "completed",
    ).sort("+created_at").to_list()

    if not sessions:
        return {"athlete_id": athlete_id, "sessions_count": 0, "series": []}

    series = []
    for s in sessions:
        sid = str(s.id)
        metrics = await Metrics.find_one(Metrics.session_id == sid)
        if not metrics:
            continue

        # Extraire uniquement mean + std par joint (suffisant pour graphiques)
        joints = {
            joint: {
                "mean": round(m.mean, 1),
                "std": round(m.std, 1),
            }
            for joint, m in (metrics.joint_metrics or {}).items()
        }

        series.append({
            "session_id": sid,
            "date": s.created_at.strftime("%Y-%m-%d") if s.created_at else "",
            "gesture_type": s.gesture_type,
            "joints": joints,
            "alerts_count": len(metrics.alerts or []),
        })

    return {
        "athlete_id": athlete_id,
        "sessions_count": len(series),
        "series": series,
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
    session.status    = "ready"   # vidéo disponible, pipeline pas encore lancé
    await session.save()

    return {
        "message":   "Vidéo uploadée avec succès",
        "video_url": save_path,
        "status":    session.status,
    }


# ── Lancement pipeline IA ────────────────────────────────────

@router.post(
    "/{session_id}/analyze",
    status_code=202,
    summary="Lancer l'analyse IA (pipeline pose_ex3_3d_savgol)",
)
async def analyze_session(
    session_id: str,
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_user),
):
    """
    Déclenche le pipeline IA sur la vidéo uploadée de façon non-bloquante.

    La route répond immédiatement avec HTTP 202 Accepted.
    Le pipeline s'exécute en tâche de fond via asyncio.create_subprocess_exec()
    sans bloquer le serveur FastAPI.

    Suivi du statut :
      - created    → processing  (dès cet appel)
      - processing → completed   (fin pipeline OK)
      - processing → error       (échec pipeline)

    Le client doit poller GET /sessions/{id} jusqu'à status != 'processing',
    puis appeler GET /sessions/{id}/results pour les métriques.

    Prérequis :
      - video_url non vide (POST /upload effectué).
      - phase_annotations recommandées pour les résultats cliniques complets.
        Sans annotations → passe 1 exploration (pas de sauvegarde MongoDB).
        Avec annotations → passe 2 complète (métriques + alertes en base).
    """
    import sys, os

    session = await _get_session_or_403(session_id, str(current_user.id))

    if not session.video_url:
        raise HTTPException(
            status_code=422,
            detail="Aucune vidéo uploadée. Utilisez d'abord POST /sessions/{id}/upload",
        )

    if session.status == "processing":
        raise HTTPException(
            status_code=409,
            detail="Une analyse est déjà en cours pour cette session.",
        )

    if session.status not in ("created", "ready", "completed", "error"):
        raise HTTPException(
            status_code=422,
            detail=f"Impossible de lancer l'analyse depuis le statut '{session.status}'",
        )

    # ── Construction de la commande ──────────────────────────
    phases_arg = None
    if session.phase_annotations:
        # Exclure la clé "variant" des phases passées au pipeline
        phase_only = {k: v for k, v in session.phase_annotations.items()
                      if k != "variant"}
        if phase_only:
            phases_arg = ",".join(f"{k}={v}" for k, v in phase_only.items())

    project_root    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    pipeline_script = os.path.join(project_root, "src", "pipeline", "pose_ex3_3d_savgol.py")

    GESTURE_MAP = {"service": "serve", "coup_droit": "forehand", "revers": "backhand"}
    gesture_arg = GESTURE_MAP.get(session.gesture_type, "serve")

    variant_arg = None
    if gesture_arg == "backhand" and session.phase_annotations:
        raw_variant = session.phase_annotations.get("variant")
        if raw_variant in (1, 2):
            variant_arg = f"{raw_variant}h"

    cmd = [
        sys.executable,
        pipeline_script,
        "--video",      session.video_url,
        "--session_id", session_id,
        "--gesture",    gesture_arg,
    ]
    if variant_arg:
        cmd += ["--variant", variant_arg]
    if phases_arg:
        cmd += ["--phases", phases_arg]
    else:
        cmd += ["--no_mongo"]

    # ── Passer en processing avant de rendre la main ────────
    session.status = "processing"
    await session.save()

    # ── Tâche de fond ────────────────────────────────────────
    background_tasks.add_task(
        _run_pipeline_background,
        session_id=session_id,
        cmd=cmd,
        project_root=project_root,
        has_annotations=phases_arg is not None,
    )

    return {
        "message":         "Analyse lancée en arrière-plan",
        "session_id":      session_id,
        "status":          "processing",
        "has_annotations": phases_arg is not None,
        "hint": (
            None if phases_arg
            else "Passe 1 lancée. Annotez les frames clés via PUT /annotations puis relancez."
        ),
    }


async def _run_pipeline_background(
    session_id: str,
    cmd: list,
    project_root: str,
    has_annotations: bool,
) -> None:
    """
    Exécute le pipeline IA de façon non-bloquante via run_in_executor().

    asyncio.create_subprocess_exec() n'est pas supporté sur Windows avec
    SelectorEventLoop (utilisé par Uvicorn). On délègue subprocess.run()
    à un ThreadPoolExecutor — le thread est bloquant mais la boucle asyncio
    reste libre pour traiter les autres requêtes.

    Met à jour session.status en base ('completed' ou 'error') à la fin.
    """
    import asyncio, os, subprocess, functools

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"]       = "1"

    session = await Session.get(session_id)
    if not session:
        print(f"[BG] Session {session_id} introuvable — abandon.")
        return

    def _run_sync() -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd,
            capture_output=True,
            timeout=600,
            cwd=project_root,
            env=env,
        )

    try:
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, _run_sync),
            timeout=620,
        )

        stdout = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
        stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""

        if stdout:
            print(f"[BG/{session_id}] stdout:\n{stdout[-1000:]}")
        if stderr:
            print(f"[BG/{session_id}] stderr:\n{stderr[-500:]}")

        if result.returncode == 0:
            session.status = "completed"
            session.error_message = None  # ← effacer erreur précédente si re-run
            print(f"[BG] Session {session_id} → completed ✓")
        else:
            session.status = "error"
            # Garder les 500 derniers caractères du stderr — utile pour le debug
            session.error_message = (stderr[-500:] or stdout[-200:] or
                                     f"Pipeline retourné code {result.returncode}").strip()
            print(f"[BG] Session {session_id} → error (code {result.returncode})")

        await session.save()

    except asyncio.TimeoutError:
        print(f"[BG] Timeout pipeline session {session_id}")
        session.status = "error"
        session.error_message = "Timeout : le pipeline a dépassé 10 minutes."
        await session.save()

    except Exception as exc:
        import traceback
        print(f"[BG] Exception pipeline session {session_id} : {exc}")
        print(f"[BG] Traceback:\n{traceback.format_exc()}")
        try:
            session.status = "error"
            session.error_message = str(exc)[:500]
            await session.save()
        except Exception:
            pass


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
    "/{session_id}/report",
    summary="Générer et télécharger le rapport PDF d'une session",
    response_class=StreamingResponse,
)
async def export_session_report(
    session_id: str,
    current_user=Depends(get_current_user),
):
    """
    Génère le rapport PDF biomécanique de la session et le retourne
    en téléchargement direct (Content-Disposition: attachment).

    Requiert :
      - La session doit exister et appartenir au spécialiste connecté.
      - Le statut doit être 'completed' (analyse terminée).
      - Un document Metrics doit exister en base pour cette session.

    Le PDF est généré en mémoire (io.BytesIO) — aucun fichier sur disque.
    """
    # ── 1. Vérifications ────────────────────────────────────────
    session = await _get_session_or_403(session_id, str(current_user.id))

    if session.status != "completed":
        raise HTTPException(
            status_code=400,
            detail="L'analyse n'est pas encore terminée. Statut actuel : " + session.status,
        )

    # ── 2. Récupérer l'athlète ──────────────────────────────────
    athlete = await Athlete.get(session.athlete_id)
    if not athlete:
        raise HTTPException(status_code=404, detail="Athlète introuvable")

    # ── 3. Récupérer les métriques ──────────────────────────────
    metrics_doc = await Metrics.find_one(Metrics.session_id == session_id)
    if not metrics_doc:
        raise HTTPException(
            status_code=404,
            detail="Aucune métrique disponible. Lancez d'abord l'analyse complète (passe 2).",
        )

    # ── 4. Normaliser normative_comparison ──────────────────────
    # Le champ peut contenir des dicts complets {measured_mean, normative_mean, ...}
    # ou des floats (format legacy).  generate_pdf() gère les deux cas.
    norm_comp = metrics_doc.normative_comparison or {}

    # ── 5. Normaliser alerts ─────────────────────────────────────
    raw_alerts = metrics_doc.alerts or []
    # Beanie retourne des objets ClinicalAlert (Pydantic) → convertir en dicts
    alerts_list = []
    for a in raw_alerts:
        if hasattr(a, "model_dump"):
            alerts_list.append(a.model_dump())
        elif hasattr(a, "dict"):
            alerts_list.append(a.dict())
        elif isinstance(a, dict):
            alerts_list.append(a)

    # ── 6. Générer le PDF ────────────────────────────────────────
    from api.routes.report_generator import generate_pdf

    pdf_bytes = generate_pdf(
        session_id           = session_id,
        gesture_type         = session.gesture_type,
        created_at           = session.created_at.isoformat() if session.created_at else None,
        pipeline_mode        = metrics_doc.pipeline_mode or "ex3_3d_savgol",
        total_frames         = metrics_doc.total_frames or 0,
        athlete_name         = athlete.name,
        athlete_age          = athlete.age,
        athlete_hand         = athlete.dominant_hand or "—",
        phase_annotations    = session.phase_annotations,
        joint_metrics        = {
            k: (v.model_dump() if hasattr(v, "model_dump") else dict(v) if not isinstance(v, dict) else v)
            for k, v in (metrics_doc.joint_metrics or {}).items()
        },
        normative_comparison = norm_comp,
        alerts               = alerts_list,
    )

    # ── 7. Nom du fichier ────────────────────────────────────────
    safe_name   = athlete.name.replace(" ", "_").replace("/", "-")
    gesture_tag = session.gesture_type
    filename    = f"rapport_{safe_name}_{gesture_tag}_{session_id[:8]}.pdf"

    # ── 8. StreamingResponse ─────────────────────────────────────
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(pdf_bytes)),
        },
    )


@router.get(
    "/{session_id}/export/csv",
    summary="Exporter les métriques de la session en CSV",
    response_class=StreamingResponse,
)
async def export_session_csv(
        session_id: str,
        current_user=Depends(get_current_user),
):
    """
    Génère un fichier CSV avec toutes les métriques articulaires
    et la comparaison normative. Téléchargement direct.

    Colonnes :
      Articulation, Min (°), Max (°), Moyenne (°), Écart-type,
      Normative (°), Δ (°), Dans 1σ, Source, Alertes
    """
    import csv

    # ── 1. Vérifications ────────────────────────────────────
    session = await _get_session_or_403(session_id, str(current_user.id))

    if session.status != "completed":
        raise HTTPException(
            status_code=400,
            detail="La session n'est pas encore complétée.",
        )

    metrics_doc = await Metrics.find_one(Metrics.session_id == session_id)
    if not metrics_doc:
        raise HTTPException(
            status_code=404,
            detail="Aucune métrique disponible pour cette session.",
        )

    athlete = await Athlete.get(session.athlete_id)

    # ── 2. Labels traduits ───────────────────────────────────
    JOINT_LABELS = {
        "knee_flexion_right": "Flexion genou D",
        "knee_flexion_left": "Flexion genou G",
        "trunk_inclination": "Inclinaison tronc",
        "trunk_rotation": "Rotation tronc",
        "shoulder_rotation_right": "Rotation épaule D",
        "shoulder_elevation_right": "Élévation épaule D",
        "shoulder_elevation_left": "Élévation épaule G",
        "elbow_right": "Flexion coude D",
        "elbow_left": "Flexion coude G",
        "hip_right": "Hanche D",
        "hip_left": "Hanche G",
    }

    GESTURE_LABELS = {
        "service": "Service",
        "coup_droit": "Coup droit",
        "revers": "Revers",
    }

    def jlabel(k: str) -> str:
        return JOINT_LABELS.get(k, k.replace("_", " ").title())

    # ── 3. Construire le CSV en mémoire ──────────────────────
    output = io.StringIO()
    writer = csv.writer(output, delimiter=",", quoting=csv.QUOTE_MINIMAL)

    # En-tête du rapport
    athlete_name = athlete.name if athlete else "—"
    gesture = GESTURE_LABELS.get(session.gesture_type, session.gesture_type)
    date_str = session.created_at.strftime("%d/%m/%Y") if session.created_at else "—"

    writer.writerow(["IA/Serve — Rapport d'analyse biomécanique"])
    writer.writerow(["Athlète", athlete_name])
    writer.writerow(["Geste", gesture])
    writer.writerow(["Date", date_str])
    writer.writerow(["Pipeline", metrics_doc.pipeline_mode or "ex3_3d_savgol"])
    writer.writerow(["Frames analysées", metrics_doc.total_frames or 0])
    writer.writerow([])  # ligne vide

    # ── Section métriques articulaires ───────────────────────
    writer.writerow(["MÉTRIQUES ARTICULAIRES"])
    writer.writerow([
        "Articulation", "Min (°)", "Max (°)", "Moyenne (°)", "Écart-type (°)"
    ])

    for joint_key, m in (metrics_doc.joint_metrics or {}).items():
        if hasattr(m, "model_dump"):
            md = m.model_dump()
        elif hasattr(m, "dict"):
            md = m.dict()
        else:
            md = dict(m) if not isinstance(m, dict) else m

        writer.writerow([
            jlabel(joint_key),
            f"{md.get('min', 0):.1f}",
            f"{md.get('max', 0):.1f}",
            f"{md.get('mean', 0):.1f}",
            f"{md.get('std', 0):.1f}",
        ])

    writer.writerow([])  # ligne vide

    # ── Section comparaison normative ────────────────────────
    writer.writerow(["COMPARAISON NORMATIVE — Gorce 2024 / Elliott 2008"])
    writer.writerow([
        "Articulation", "Mesuré (°)", "Normative (°)",
        "Δ (°)", "Dans 1σ", "Source"
    ])

    for joint_key, comp in (metrics_doc.normative_comparison or {}).items():
        if isinstance(comp, dict):
            measured = comp.get("measured_mean", 0) or 0
            normative = comp.get("normative_mean", 0) or 0
            delta = comp.get("delta_degrees", 0) or 0
            within_1sd = comp.get("within_1std", True)
            source = comp.get("source", "—")
        else:
            measured, normative, delta = 0, 0, float(comp)
            within_1sd, source = True, "—"

        writer.writerow([
            jlabel(joint_key),
            f"{measured:.1f}",
            f"{normative:.1f}",
            f"{delta:+.1f}",
            "Oui" if within_1sd else "Non",
            source,
        ])

    writer.writerow([])  # ligne vide

    # ── Section alertes cliniques ────────────────────────────
    alerts = metrics_doc.alerts or []
    if alerts:
        writer.writerow([f"ALERTES CLINIQUES ({len(alerts)})"])
        writer.writerow([
            "Articulation", "Phase", "Valeur (°)", "Seuil (°)", "Sévérité", "Référence"
        ])
        for a in alerts:
            if hasattr(a, "model_dump"):
                ad = a.model_dump()
            elif hasattr(a, "dict"):
                ad = a.dict()
            else:
                ad = dict(a) if not isinstance(a, dict) else a

            writer.writerow([
                jlabel(ad.get("joint", "—")),
                ad.get("phase", "—").replace("_", " "),
                f"{ad.get('value', 0):.1f}",
                f"{ad.get('threshold', 0):.1f}",
                ad.get("severity", "—"),
                ad.get("reference", "—"),
            ])
    else:
        writer.writerow(["ALERTES CLINIQUES"])
        writer.writerow(["Aucune alerte clinique détectée"])

    writer.writerow([])
    writer.writerow(["Session ID", session_id])
    writer.writerow(["Généré par", "IA/Serve — Plateforme d'analyse biomécanique"])

    # ── 4. Réponse ───────────────────────────────────────────
    safe_name = (athlete_name).replace(" ", "_").replace("/", "-")
    gesture_tag = session.gesture_type
    filename = f"metriques_{safe_name}_{gesture_tag}_{session_id[:8]}.csv"

    csv_bytes = output.getvalue().encode("utf-8-sig")  # BOM pour Excel FR

    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(csv_bytes)),
        },
    )

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
            load_frames, detect_fps,
            find_all_candidates, select_best_combination,
            check_camera_angle_reliability,
            PHASE_ORDER, get_key_angles_str,
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
    fps                = detect_fps(frames)
    normatives         = get_normatives(gesture)
    camera_reliability = check_camera_angle_reliability(frames, gesture)
    candidates         = find_all_candidates(frames, gesture, normatives, fps,
                                             camera_reliability)
    best, _            = select_best_combination(candidates, gesture, fps,
                                                 frames=frames,
                                                 camera_reliability=camera_reliability)

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

    phase_order = PHASE_ORDER.get(gesture, list(candidates.keys()))
    for phase_name in phase_order:
        frame_list = candidates.get(phase_name, [])
        top3 = frame_list[:3]
        # Si best=None (veto a tout rejeté ou aucune combinaison valide),
        # on affiche quand même les candidats mais on force la confiance à
        # UNRELIABLE pour signaler au spécialiste qu'une vérification manuelle
        # est nécessaire — on ne cache pas silencieusement l'échec du veto.
        if best is not None:
            best_fn   = best.get(phase_name)
            best_conf = top3[0][2] if top3 else "UNRELIABLE"
        else:
            best_fn   = top3[0][0] if top3 else None
            best_conf = "UNRELIABLE"  # veto a rejeté toutes les combinaisons

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
                    "frame":          fn,
                    "score":          round(score, 3),
                    "confidence":     conf,
                    "screenshot_b64": (
                        extract_frame_screenshot(
                            session.video_url, fn,
                            frame_lookup[fn], phase_name, conf
                        )
                        if has_video and fn in frame_lookup else ""
                    ),
                    "key_angles": (
                        {
                            JOINT_SHORT.get(joint, joint): round(v, 1)
                            for joint in list(normatives.get(phase_name, {}).keys())[:5]
                            if (v := frame_lookup[fn].get("angles", {}).get(joint)) is not None
                        }
                        if fn in frame_lookup else {}
                    ),
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