from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from fastapi import HTTPException, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends
from api.models import User
import os

security = HTTPBearer()

pwd_context = CryptContext(schemes=["bcrypt"])

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def create_token(user_id: str) -> str:
    expire = datetime.utcnow() + timedelta(
        minutes=int(os.getenv("JWT_EXPIRE_MINUTES", 60))
    )
    return jwt.encode(
        {"sub": user_id, "exp": expire},
        os.getenv("JWT_SECRET_KEY"),
        algorithm=os.getenv("JWT_ALGORITHM", "HS256")
    )

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security) ) -> User:
    try:
        token = credentials.credentials
        payload = jwt.decode(
            token,
            os.getenv("JWT_SECRET_KEY"),
            algorithms=[os.getenv("JWT_ALGORITHM", "HS256")]
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