from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta , timezone
from fastapi import HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends
from api.models import User
import os

security = HTTPBearer()

pwd_context = CryptContext(schemes=["bcrypt"])

_JWT_SECRET = os.getenv("JWT_SECRET_KEY")
if not _JWT_SECRET:
    raise RuntimeError(
        "JWT_SECRET_KEY manquant dans .env — démarrage refusé. "
        "Générez une clé avec : python -c \"import secrets; print(secrets.token_hex(32))\""
    )

_JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
_JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", 60))

# Hash factice utilisé pour prévenir les timing attacks
# Même longueur et format qu'un vrai hash bcrypt
_DUMMY_HASH = "$2b$12$dummy.hash.for.timing.attack.prevention.xxxxxxxxx"

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def create_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=_JWT_EXPIRE_MINUTES)

    return jwt.encode(
        _JWT_SECRET,
        algorithm=_JWT_ALGORITHM
    )

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security) ) -> User:
    try:
        token = credentials.credentials
        payload = jwt.decode(
            token,
            _JWT_SECRET,
            algorithms=[_JWT_ALGORITHM]
        )
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Token invalide")

        user = await User.get(user_id)
        if not user:
            raise HTTPException(status_code=401, detail="Utilisateur introuvable")
        if user.status != "active":
            raise HTTPException(status_code=403, detail="Compte non activé")

        return user

    except JWTError:
        raise HTTPException(status_code=401, detail="Token expiré ou invalide")