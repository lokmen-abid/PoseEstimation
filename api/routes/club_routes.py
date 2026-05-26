from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import Optional
from api.models import Club, User, Athlete
from api.auth import get_current_user

router = APIRouter(tags=["Clubs"])

# ── Schémas ─────────────────────────────────────────────────

class ClubCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    city: Optional[str] = Field(None, max_length=100)

# ── Endpoints ───────────────────────────────────────────────
@router.get(
    "/overview",
    summary="Admin — vue globale : tous les clubs + stats"
)
async def get_clubs_overview(current_user: User = Depends(get_current_user)):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Accès admin requis")

    clubs = await Club.find_all().to_list()
    result = []

    for club in clubs:
        specialists = await User.find(
            User.club_id == str(club.id),
            User.role == "specialist"
        ).to_list()

        specialist_ids = [str(s.id) for s in specialists]
        athlete_count = await Athlete.find(
            {"specialist_id": {"$in": specialist_ids}}
        ).count()

        result.append({
            "id": str(club.id),
            "name": club.name,
            "city": club.city,
            "specialist_count": len(specialists),
            "athlete_count": athlete_count
        })

    # Indépendants (sans club)
    independents = await User.find(
        User.club_id == None,
        User.role == "specialist"
    ).to_list()

    independent_ids = [str(s.id) for s in independents]
    independent_athletes = await Athlete.find(
        {"specialist_id": {"$in": independent_ids}}
    ).count() if independent_ids else 0

    result.append({
        "id": None,
        "name": "Indépendants",
        "city": None,
        "specialist_count": len(independents),
        "athlete_count": independent_athletes  # Fix : calculé au lieu de 0 fixe
    })

    return result


@router.get(
    "/",
    summary="Lister tous les clubs (public — pour le formulaire d'inscription)"
)
async def list_clubs(
        skip: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=200)
):
    clubs = await Club.find_all().skip(skip).limit(limit).to_list()
    return [
        {"id": str(c.id), "name": c.name, "city": c.city}
        for c in clubs
    ]


@router.post(
    "/",
    summary="Admin — créer un club",
    status_code=201
)
async def create_club(
        data: ClubCreate,
        current_user: User = Depends(get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Accès admin requis")

    # Vérifie qu'un club avec le même nom n'existe pas déjà
    existing = await Club.find_one(Club.name == data.name)
    if existing:
        raise HTTPException(status_code=400, detail="Un club avec ce nom existe déjà")

    club = Club(name=data.name, city=data.city)
    await club.insert()

    return {
        "message": "Club créé avec succès",
        "club": {
            "id": str(club.id),
            "name": club.name,
            "city": club.city
        }
    }


@router.get(
    "/{club_id}/specialists",
    summary="Admin — spécialistes d'un club"
)
async def get_club_specialists(
        club_id: str,
        current_user: User = Depends(get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Accès admin requis")

    club = await Club.get(club_id)
    if not club:
        raise HTTPException(status_code=404, detail="Club introuvable")

    specialists = await User.find(
        User.club_id == club_id,
        User.role == "specialist"
    ).to_list()

    return {
        "club": {
            "id": str(club.id),
            "name": club.name,
            "city": club.city
        },
        "specialists": [
            {
                "id": str(s.id),
                "full_name": s.full_name,
                "email": s.email,
                "status": s.status
            }
            for s in specialists
        ]
    }