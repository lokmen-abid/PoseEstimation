from fastapi import APIRouter, HTTPException, Depends , Request
from pydantic import BaseModel, EmailStr ,  Field, field_validator
from api.models import User, Club
from api.auth import (
    hash_password, verify_password, create_token,
    get_current_user, _DUMMY_HASH
)
from typing import Optional
import time


router = APIRouter(tags=["Authentification"])

_login_attempts: dict = {}
_MAX_ATTEMPTS   = 5     # tentatives max
_WINDOW_SECONDS = 60    # par fenêtre de 60 secondes


def _check_rate_limit(ip: str):
    """Bloque une IP après MAX_ATTEMPTS tentatives dans WINDOW_SECONDS."""
    now = time.time()
    attempts = _login_attempts.get(ip, [])
    # Nettoie les tentatives hors fenêtre
    attempts = [t for t in attempts if now - t < _WINDOW_SECONDS]

    if len(attempts) >= _MAX_ATTEMPTS:
        remaining = int(_WINDOW_SECONDS - (now - attempts[0]))
        raise HTTPException(
            status_code=429,
            detail=f"Trop de tentatives. Réessayez dans {remaining} secondes."
        )

    attempts.append(now)
    _login_attempts[ip] = attempts


def _clear_rate_limit(ip: str):
    """Réinitialise le compteur après une connexion réussie."""
    _login_attempts.pop(ip, None)

# ── Schémas de requête ──────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    full_name: str = Field(..., min_length=2, max_length=100)
    club_id: Optional[str] = None

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError("Le mot de passe doit contenir au moins une majuscule")
        if not any(c.isdigit() for c in v):
            raise ValueError("Le mot de passe doit contenir au moins un chiffre")
        return v

    @field_validator("full_name")
    @classmethod
    def full_name_no_script(cls, v: str) -> str:
        # Prévention basique XSS dans les champs texte
        forbidden = ["<", ">", "&", "\"", "'"]
        for char in forbidden:
            if char in v:
                raise ValueError("Caractères non autorisés dans le nom")
        return v.strip()


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=128)

class ApproveRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    action: str = Field(..., pattern="^(approve|block)$")


# SCHÉMAS DE RÉPONSE


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    role: str
    club_id: Optional[str]


class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse


class PendingUserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    club_id: Optional[str]


class FullUserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    status: str
    club_id: Optional[str]

# ── Endpoints ───────────────────────────────────────────────

@router.post(
    "/register",
    summary="Créer un compte spécialiste",
    status_code=201
)
async def register(data: RegisterRequest):
    existing = await User.find_one(User.email == data.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email déjà utilisé")

    if data.club_id:
        club = await Club.get(data.club_id)
        if not club:
            raise HTTPException(status_code=404, detail="Club introuvable")

    user = User(
        email=data.email,
        password_hash=hash_password(data.password),
        full_name=data.full_name,
        club_id=data.club_id,
        status="pending"
    )
    await user.insert()

    return {
        "message": "Compte créé. En attente de validation par l'administrateur.",
        "email": user.email,
        "club_id": user.club_id
    }


@router.post(
    "/login",
    response_model=LoginResponse,
    summary="Connexion — retourne un token JWT"
)
async def login(request: Request, data: LoginRequest):
    # ── Rate limiting ──
    client_ip = request.client.host if request.client else "unknown"
    _check_rate_limit(client_ip)

    user = await User.find_one(User.email == data.email)

    # ── Fix timing attack : toujours appeler verify_password ──
    hash_to_check = user.password_hash if user else _DUMMY_HASH
    is_valid = verify_password(data.password, hash_to_check)

    if not user or not is_valid:
        raise HTTPException(status_code=401, detail="Email ou mot de passe incorrect")

    if user.status == "pending":
        raise HTTPException(status_code=403, detail="Compte en attente de validation admin")

    if user.status == "blocked":
        raise HTTPException(status_code=403, detail="Compte bloqué")

    # Connexion réussie — réinitialise le compteur
    _clear_rate_limit(client_ip)

    token = create_token(str(user.id))
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": str(user.id),
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
            "club_id": user.club_id
        }
    }


@router.get(
    "/pending",
    response_model=list[PendingUserResponse],
    summary="Admin — liste des comptes en attente de validation"
)
async def get_pending_users(current_user: User = Depends(get_current_user)):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Accès admin requis")

    pending = await User.find(User.status == "pending").to_list()
    return [
        {
            "id": str(u.id),
            "email": u.email,
            "full_name": u.full_name,
            "club_id": u.club_id
        }
        for u in pending
    ]


@router.post(
    "/approve",
    summary="Admin — approuver ou bloquer un compte"
)
async def approve_user(
        data: ApproveRequest,
        current_user: User = Depends(get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Accès admin requis")

    user = await User.get(data.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable")

    # Empêche l'admin de se bloquer lui-même
    if str(user.id) == str(current_user.id):
        raise HTTPException(status_code=400, detail="Impossible de modifier son propre compte")

    if data.action == "approve":
        user.status = "active"
        msg = f"{user.full_name} est maintenant actif"
    else:  # "block" — validé par le pattern Pydantic
        user.status = "blocked"
        msg = f"{user.full_name} est bloqué"

    await user.save()
    return {"message": msg}


@router.get(
    "/users",
    response_model=list[FullUserResponse],
    summary="Admin — liste de tous les spécialistes"
)
async def get_all_users(current_user: User = Depends(get_current_user)):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Accès admin requis")

    users = await User.find(User.role == "specialist").to_list()
    return [
        {
            "id": str(u.id),
            "email": u.email,
            "full_name": u.full_name,
            "status": u.status,
            "club_id": u.club_id
        }
        for u in users
    ]