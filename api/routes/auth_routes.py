from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, EmailStr
from api.models import User
from api.auth import hash_password, verify_password, create_token, get_current_user

router = APIRouter()

# ── Schémas de requête ──────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class ApproveRequest(BaseModel):
    user_id: str
    action: str   # "approve" | "block"

# ── Endpoints ───────────────────────────────────────────────

@router.post("/register")
async def register(data: RegisterRequest):
    # Vérifier si l'email existe déjà
    existing = await User.find_one(User.email == data.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email déjà utilisé")

    user = User(
        email=data.email,
        password_hash=hash_password(data.password),
        full_name=data.full_name,
        status="pending"     # ← en attente validation admin
    )
    await user.insert()

    return {
        "message": "Compte créé. En attente de validation par l'administrateur.",
        "email": user.email
    }


@router.post("/login")
async def login(data: LoginRequest):
    user = await User.find_one(User.email == data.email)

    if not user or not verify_password(data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Email ou mot de passe incorrect")

    if user.status == "pending":
        raise HTTPException(status_code=403, detail="Compte en attente de validation admin")

    if user.status == "blocked":
        raise HTTPException(status_code=403, detail="Compte bloqué")

    token = create_token(str(user.id))
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": str(user.id),
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role
        }
    }


@router.get("/pending")
async def get_pending_users(current_user: User = Depends(get_current_user)):
    # Seulement l'admin peut voir les comptes en attente
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Accès admin requis")

    pending = await User.find(User.status == "pending").to_list()
    return [{"id": str(u.id), "email": u.email, "full_name": u.full_name} for u in pending]


@router.post("/approve")
async def approve_user(data: ApproveRequest, current_user: User = Depends(get_current_user)):
    # Seulement l'admin peut approuver
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Accès admin requis")

    user = await User.get(data.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable")

    if data.action == "approve":
        user.status = "active"
        msg = f"{user.full_name} est maintenant actif"
    elif data.action == "block":
        user.status = "blocked"
        msg = f"{user.full_name} est bloqué"
    else:
        raise HTTPException(status_code=400, detail="Action invalide")

    await user.save()
    return {"message": msg}