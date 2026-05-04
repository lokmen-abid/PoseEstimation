from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from api.models import Athlete
from api.auth import get_current_user

router = APIRouter()

# ── Schémas ────────────────────────────────────────────────

class AthleteCreate(BaseModel):
    name: str
    age: int
    dominant_hand: str = "right"    # "right" | "left"
    medical_notes: Optional[str] = None

class AthleteUpdate(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    dominant_hand: Optional[str] = None
    medical_notes: Optional[str] = None

# ── Endpoints ───────────────────────────────────────────────

@router.post("/")
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
        "athlete": {
            "id": str(athlete.id),
            "name": athlete.name,
            "age": athlete.age,
            "dominant_hand": athlete.dominant_hand,
            "medical_notes": athlete.medical_notes,
            "created_at": athlete.created_at
        }
    }


@router.get("/")
async def get_athletes(current_user=Depends(get_current_user)):
    # Uniquement les athlètes de CE spécialiste
    athletes = await Athlete.find(
        Athlete.specialist_id == str(current_user.id)
    ).to_list()

    return [
        {
            "id": str(a.id),
            "name": a.name,
            "age": a.age,
            "dominant_hand": a.dominant_hand,
            "medical_notes": a.medical_notes,
            "created_at": a.created_at
        }
        for a in athletes
    ]


@router.get("/{athlete_id}")
async def get_athlete(
    athlete_id: str,
    current_user=Depends(get_current_user)
):
    athlete = await Athlete.get(athlete_id)

    if not athlete:
        raise HTTPException(status_code=404, detail="Athlète introuvable")

    # Vérifier que l'athlète appartient au spécialiste connecté
    if athlete.specialist_id != str(current_user.id):
        raise HTTPException(status_code=403, detail="Accès refusé")

    return {
        "id": str(athlete.id),
        "name": athlete.name,
        "age": athlete.age,
        "dominant_hand": athlete.dominant_hand,
        "medical_notes": athlete.medical_notes,
        "created_at": athlete.created_at
    }


@router.put("/{athlete_id}")
async def update_athlete(
    athlete_id: str,
    data: AthleteUpdate,
    current_user=Depends(get_current_user)
):
    athlete = await Athlete.get(athlete_id)

    if not athlete:
        raise HTTPException(status_code=404, detail="Athlète introuvable")

    # Vérifier que l'athlète appartient au spécialiste connecté
    if athlete.specialist_id != str(current_user.id):
        raise HTTPException(status_code=403, detail="Accès refusé")

    # Mettre à jour seulement les champs envoyés
    if data.name is not None:
        athlete.name = data.name
    if data.age is not None:
        athlete.age = data.age
    if data.dominant_hand is not None:
        athlete.dominant_hand = data.dominant_hand
    if data.medical_notes is not None:
        athlete.medical_notes = data.medical_notes

    await athlete.save()
    return {
        "message": "Athlète mis à jour",
        "athlete": {
            "id": str(athlete.id),
            "name": athlete.name,
            "age": athlete.age,
            "dominant_hand": athlete.dominant_hand,
            "medical_notes": athlete.medical_notes
        }
    }


@router.delete("/{athlete_id}")
async def delete_athlete(
    athlete_id: str,
    current_user=Depends(get_current_user)
):
    athlete = await Athlete.get(athlete_id)

    if not athlete:
        raise HTTPException(status_code=404, detail="Athlète introuvable")

    #  Vérifier que l'athlète appartient au spécialiste connecté
    if athlete.specialist_id != str(current_user.id):
        raise HTTPException(status_code=403, detail="Accès refusé")

    await athlete.delete()
    return {"message": f"Athlète {athlete.name} supprimé avec succès"}