from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from api.models import Club, User, Athlete
from api.auth import get_current_user

router = APIRouter()

# ── Schémas ─────────────────────────────────────────────────

class ClubCreate(BaseModel):
    name: str
    city: Optional[str] = None

# ── Endpoints ───────────────────────────────────────────────

@router.post("/")
async def create_club(
    data: ClubCreate,
    current_user: User = Depends(get_current_user)
):
    # Seulement l'admin peut créer un club
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Accès admin requis")

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


@router.get("/")
async def list_clubs(current_user: User = Depends(get_current_user)):
    # Accessible par tous — spécialistes en ont besoin au register
    clubs = await Club.find_all().to_list()
    return [
        {"id": str(c.id), "name": c.name, "city": c.city}
        for c in clubs
    ]


@router.get("/{club_id}/specialists")
async def get_club_specialists(
    club_id: str,
    current_user: User = Depends(get_current_user)
):
    # Seulement l'admin peut voir les spécialistes d'un club
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
        "club": {"id": str(club.id), "name": club.name, "city": club.city},
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


@router.get("/overview")
async def get_clubs_overview(current_user: User = Depends(get_current_user)):
    # Vue admin : tous les clubs + leurs spécialistes + nb athlètes
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

    # Ajouter les indépendants (sans club)
    independents = await User.find(
        User.club_id == None,
        User.role == "specialist"
    ).to_list()

    result.append({
        "id": None,
        "name": "Indépendants",
        "city": None,
        "specialist_count": len(independents),
        "athlete_count": 0   # à calculer si besoin
    })

    return result