"""
Router FastAPI — Module Analyse de Jeu
Prefix  : /api/match-sessions
Tags    : Match Analysis

Endpoints :
  POST   /                          Créer une MatchSession
  GET    /                          Lister les MatchSessions d'un athlète
  GET    /{match_id}                Détail d'une MatchSession
  DELETE /{match_id}                Supprimer une MatchSession
  POST   /{match_id}/upload         Upload de la vidéo match
  POST   /{match_id}/analyze        Lancer le pipeline analyze_match.py
  GET    /{match_id}/results        Récupérer les MatchMetrics
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, BackgroundTasks
from pydantic import BaseModel, Field

from api.auth import get_current_user
from api.models import MatchSession, MatchMetrics, Athlete

router = APIRouter(tags=["Match Analysis"])

# ── Répertoire de stockage vidéos match ───────────────────────
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads/match_videos"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# SCHÉMAS PYDANTIC
# ══════════════════════════════════════════════════════════════

class MatchSessionCreate(BaseModel):
    athlete_id: str
    surface: str = Field("hard", pattern="^(hard|clay|grass)$")
    match_format: str = Field("best_of_3", pattern="^(best_of_3|best_of_5|pro_set)$")
    opponent_name: Optional[str] = Field(None, max_length=100)
    location: Optional[str] = Field(None, max_length=200)
    notes: Optional[str] = Field(None, max_length=2000)


class MatchSessionUpdate(BaseModel):
    surface: Optional[str] = Field(None, pattern="^(hard|clay|grass)$")
    match_format: Optional[str] = Field(None, pattern="^(best_of_3|best_of_5|pro_set)$")
    opponent_name: Optional[str] = Field(None, max_length=100)
    location: Optional[str] = Field(None, max_length=200)
    notes: Optional[str] = Field(None, max_length=2000)


# ── Sérialiseurs ──────────────────────────────────────────────

def _serialize_match(m: MatchSession) -> dict:
    return {
        "id": str(m.id),
        "athlete_id": m.athlete_id,
        "specialist_id": m.specialist_id,
        "video_url": m.video_url,
        "status": m.status,
        "fps": m.fps,
        "total_frames": m.total_frames,
        "duration_seconds": m.duration_seconds,
        "surface": m.surface,
        "match_format": m.match_format,
        "opponent_name": m.opponent_name,
        "location": m.location,
        "notes": m.notes,
        "created_at": m.created_at.isoformat() if m.created_at else None,
        "error_message": m.error_message,
    }


def _serialize_metrics(mx: MatchMetrics) -> dict:
    return {
        "match_session_id": mx.match_session_id,
        "athlete_id": mx.athlete_id,
        "total_frames_analyzed": mx.total_frames_analyzed,
        "total_points_detected": mx.total_points_detected,
        "total_rallies_detected": mx.total_rallies_detected,
        "sets_detected": mx.sets_detected,
        "gesture_events": [e.model_dump() for e in mx.gesture_events],
        "rallies": [r.model_dump() for r in mx.rallies],
        "sets": {k: v.model_dump() for k, v in mx.sets.items()},
        "overall_gesture_counts": mx.overall_gesture_counts,
        "overall_gesture_pct": mx.overall_gesture_pct,
        "dominant_gesture_match": mx.dominant_gesture_match,
        "avg_rally_length_seconds": mx.avg_rally_length_seconds,
        "avg_strokes_per_rally": mx.avg_strokes_per_rally,
        "strengths": mx.strengths,
        "weaknesses": mx.weaknesses,
        "pipeline_version": mx.pipeline_version,
        "computed_at": mx.computed_at.isoformat() if mx.computed_at else None,
    }


# ── Guard : vérif ownership ───────────────────────────────────

async def _get_match_owned(match_id: str, current_user) -> MatchSession:
    match = await MatchSession.get(match_id)
    if not match:
        raise HTTPException(status_code=404, detail="MatchSession introuvable")
    if match.specialist_id != str(current_user.id):
        raise HTTPException(status_code=403, detail="Accès refusé")
    return match


# ══════════════════════════════════════════════════════════════
# ENDPOINTS CRUD
# ══════════════════════════════════════════════════════════════

@router.post("/", status_code=201, summary="Créer une MatchSession")
async def create_match_session(
    data: MatchSessionCreate,
    current_user=Depends(get_current_user),
):
    # Vérifier que l'athlète appartient au spécialiste
    athlete = await Athlete.get(data.athlete_id)
    if not athlete:
        raise HTTPException(status_code=404, detail="Athlète introuvable")
    if athlete.specialist_id != str(current_user.id):
        raise HTTPException(status_code=403, detail="Athlète non autorisé")

    match = MatchSession(
        athlete_id=data.athlete_id,
        specialist_id=str(current_user.id),
        video_url="",
        surface=data.surface,
        match_format=data.match_format,
        opponent_name=data.opponent_name,
        location=data.location,
        notes=data.notes,
    )
    await match.insert()
    return {"message": "MatchSession créée", "match_session": _serialize_match(match)}


@router.get("/", summary="Lister les MatchSessions d'un athlète")
async def list_match_sessions(
    athlete_id: str = Query(..., description="ID de l'athlète"),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    current_user=Depends(get_current_user),
):
    # Vérifier ownership athlète
    athlete = await Athlete.get(athlete_id)
    if not athlete:
        raise HTTPException(status_code=404, detail="Athlète introuvable")
    if athlete.specialist_id != str(current_user.id):
        raise HTTPException(status_code=403, detail="Accès refusé")

    matches = await MatchSession.find(
        MatchSession.athlete_id == athlete_id,
        MatchSession.specialist_id == str(current_user.id),
    ).skip(skip).limit(limit).sort(-MatchSession.created_at).to_list()

    total = await MatchSession.find(
        MatchSession.athlete_id == athlete_id,
        MatchSession.specialist_id == str(current_user.id),
    ).count()

    return {"total": total, "skip": skip, "limit": limit, "data": [_serialize_match(m) for m in matches]}


@router.get("/{match_id}", summary="Détail d'une MatchSession")
async def get_match_session(
    match_id: str,
    current_user=Depends(get_current_user),
):
    match = await _get_match_owned(match_id, current_user)
    return _serialize_match(match)


@router.patch("/{match_id}", summary="Modifier les métadonnées d'un match")
async def update_match_session(
    match_id: str,
    data: MatchSessionUpdate,
    current_user=Depends(get_current_user),
):
    match = await _get_match_owned(match_id, current_user)

    if data.surface is not None:       match.surface = data.surface
    if data.match_format is not None:  match.match_format = data.match_format
    if data.opponent_name is not None: match.opponent_name = data.opponent_name
    if data.location is not None:      match.location = data.location
    if data.notes is not None:         match.notes = data.notes

    await match.save()
    return {"message": "MatchSession mise à jour", "match_session": _serialize_match(match)}


@router.delete("/{match_id}", summary="Supprimer une MatchSession et ses données")
async def delete_match_session(
    match_id: str,
    current_user=Depends(get_current_user),
):
    match = await _get_match_owned(match_id, current_user)

    # Supprimer le fichier vidéo si présent
    if match.video_url:
        video_path = Path(match.video_url)
        if video_path.exists():
            video_path.unlink(missing_ok=True)

    # Supprimer les métriques associées
    existing = await MatchMetrics.find_one(MatchMetrics.match_session_id == match_id)
    if existing:
        await existing.delete()

    await match.delete()
    return {"message": f"MatchSession {match_id} supprimée"}


# ══════════════════════════════════════════════════════════════
# UPLOAD VIDÉO
# ══════════════════════════════════════════════════════════════

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_SIZE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB


@router.post("/{match_id}/upload", summary="Upload la vidéo du match")
async def upload_match_video(
    match_id: str,
    file: UploadFile = File(...),
    current_user=Depends(get_current_user),
):
    match = await _get_match_owned(match_id, current_user)

    if match.status not in ("created", "error"):
        raise HTTPException(status_code=400, detail="Vidéo déjà uploadée ou traitement en cours")

    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Format non supporté. Acceptés : {ALLOWED_EXTENSIONS}")

    dest = UPLOAD_DIR / f"{match_id}{ext}"

    # Stream vers disque (évite de charger 2 GB en RAM)
    try:
        with dest.open("wb") as f:
            while chunk := await file.read(1024 * 1024):  # 1 MB chunks
                f.write(chunk)
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Erreur d'upload : {e}")

    match.video_url = str(dest)
    match.status = "ready"
    await match.save()

    return {"message": "Vidéo uploadée", "video_url": str(dest), "status": "ready"}


# ══════════════════════════════════════════════════════════════
# LANCEMENT DU PIPELINE
# ══════════════════════════════════════════════════════════════

async def _run_pipeline(match_id: str, video_path: str):
    """Tâche background : lance analyze_match.py et met à jour le statut."""
    match = await MatchSession.get(match_id)
    if not match:
        return

    match.status = "processing"
    await match.save()

    try:
        result = subprocess.run(
            [
                "python", "src/pipeline/analyze_match.py",
                "--video", video_path,
                "--match_id", match_id,
            ],
            capture_output=True,
            text=True,
            timeout=3600,  # 1h max pour un match complet
        )

        if result.returncode != 0:
            match.status = "error"
            match.error_message = result.stderr[-500:] if result.stderr else "Erreur inconnue"
        else:
            match.status = "completed"
            match.error_message = None

    except subprocess.TimeoutExpired:
        match.status = "error"
        match.error_message = "Timeout : traitement trop long (>1h)"
    except Exception as e:
        match.status = "error"
        match.error_message = str(e)

    await match.save()


@router.post("/{match_id}/analyze", summary="Lancer le pipeline d'analyse")
async def analyze_match(
    match_id: str,
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_user),
):
    match = await _get_match_owned(match_id, current_user)

    if match.status != "ready":
        raise HTTPException(
            status_code=400,
            detail=f"Impossible de lancer l'analyse depuis le statut '{match.status}'. Statut requis : 'ready'",
        )

    if not match.video_url or not Path(match.video_url).exists():
        raise HTTPException(status_code=400, detail="Fichier vidéo introuvable")

    background_tasks.add_task(_run_pipeline, match_id, match.video_url)

    return {"message": "Analyse lancée en arrière-plan", "status": "processing"}


# ══════════════════════════════════════════════════════════════
# RÉSULTATS
# ══════════════════════════════════════════════════════════════

@router.get("/{match_id}/results", summary="Récupérer les résultats d'analyse")
async def get_match_results(
    match_id: str,
    current_user=Depends(get_current_user),
):
    match = await _get_match_owned(match_id, current_user)

    if match.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Analyse non terminée. Statut actuel : '{match.status}'",
        )

    metrics = await MatchMetrics.find_one(MatchMetrics.match_session_id == match_id)
    if not metrics:
        raise HTTPException(status_code=404, detail="Métriques introuvables malgré le statut 'completed'")

    return _serialize_metrics(metrics)