from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from api.models import Athlete
from api.auth import get_current_user

router = APIRouter(tags=["Athlètes"])

# ── Schémas ────────────────────────────────────────────────
class AthleteCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    age: int = Field(..., ge=5, le=100)
    sex: str = "male"
    dominant_hand: str = Field("right", pattern="^(right|left)$")
    medical_notes: Optional[str] = Field(None, max_length=2000)

    @field_validator("name")
    @classmethod
    def name_no_script(cls, v: str) -> str:
        forbidden = ["<", ">", "&", "\"", "'"]
        for char in forbidden:
            if char in v:
                raise ValueError("Caractères non autorisés dans le nom")
        return v.strip()


class AthleteUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=2, max_length=100)
    age: Optional[int] = Field(None, ge=5, le=100)
    sex: Optional[str] = None
    dominant_hand: Optional[str] = Field(None, pattern="^(right|left)$")
    medical_notes: Optional[str] = Field(None, max_length=2000)

    @field_validator("name")
    @classmethod
    def name_no_script(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        forbidden = ["<", ">", "&", "\"", "'"]
        for char in forbidden:
            if char in v:
                raise ValueError("Caractères non autorisés dans le nom")
        return v.strip()


# SCHÉMAS DE RÉPONSE

class AthleteResponse(BaseModel):
    id: str
    name: str
    age: int
    dominant_hand: str
    medical_notes: Optional[str]
    created_at: str  # ISO string pour le frontend


def _serialize(a: Athlete) -> dict:
    """Sérialise un athlète en dict JSON-safe."""
    return {
        "id": str(a.id),
        "name": a.name,
        "age": a.age,
        "sex": a.sex,
        "dominant_hand": a.dominant_hand,
        "medical_notes": a.medical_notes,
        "created_at": a.created_at.isoformat() if a.created_at else None,
    }


# ── Endpoints ───────────────────────────────────────────────
@router.post(
    "/",
    summary="Créer un athlète",
    status_code=201
)
async def create_athlete(
        data: AthleteCreate,
        current_user=Depends(get_current_user)
):
    athlete = Athlete(
        specialist_id=str(current_user.id),
        name=data.name,
        age=data.age,
        dominant_hand=data.dominant_hand,
        medical_notes=data.medical_notes
    )
    await athlete.insert()
    return {
        "message": "Athlète créé avec succès",
        "athlete": _serialize(athlete)
    }


@router.get(
    "/",
    summary="Lister mes athlètes (paginé)"
)
async def get_athletes(
        skip: int = Query(default=0, ge=0, description="Nombre d'entrées à sauter"),
        limit: int = Query(default=20, ge=1, le=100, description="Nombre max de résultats"),
        current_user=Depends(get_current_user)
):
    athletes = await Athlete.find(
        Athlete.specialist_id == str(current_user.id)
    ).skip(skip).limit(limit).to_list()

    total = await Athlete.find(
        Athlete.specialist_id == str(current_user.id)
    ).count()

    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "data": [_serialize(a) for a in athletes]
    }


@router.get(
    "/{athlete_id}",
    summary="Consulter un athlète par ID"
)
async def get_athlete(
        athlete_id: str,
        current_user=Depends(get_current_user)
):
    athlete = await Athlete.get(athlete_id)

    if not athlete:
        raise HTTPException(status_code=404, detail="Athlète introuvable")

    if athlete.specialist_id != str(current_user.id):
        raise HTTPException(status_code=403, detail="Accès refusé")

    return _serialize(athlete)


@router.put(
    "/{athlete_id}",
    summary="Modifier partiellement un athlète"
)
async def update_athlete(
        athlete_id: str,
        data: AthleteUpdate,
        current_user=Depends(get_current_user)
):
    athlete = await Athlete.get(athlete_id)

    if not athlete:
        raise HTTPException(status_code=404, detail="Athlète introuvable")

    if athlete.specialist_id != str(current_user.id):
        raise HTTPException(status_code=403, detail="Accès refusé")

    if data.name is not None: athlete.name = data.name
    if data.age is not None: athlete.age = data.age
    if data.sex is not None: athlete.sex = data.sex
    if data.dominant_hand is not None: athlete.dominant_hand = data.dominant_hand
    if data.medical_notes is not None: athlete.medical_notes = data.medical_notes

    await athlete.save()
    return {
        "message": "Athlète mis à jour",
        "athlete": _serialize(athlete)
    }


@router.delete(
    "/{athlete_id}",
    summary="Supprimer un athlète"
)
async def delete_athlete(
        athlete_id: str,
        current_user=Depends(get_current_user)
):
    athlete = await Athlete.get(athlete_id)

    if not athlete:
        raise HTTPException(status_code=404, detail="Athlète introuvable")

    if athlete.specialist_id != str(current_user.id):
        raise HTTPException(status_code=403, detail="Accès refusé")

    await athlete.delete()
    return {"message": f"Athlète {athlete.name} supprimé avec succès"}