from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from contextlib import asynccontextmanager
from api.database import connect_db
from api.routes import athletes, sessions, auth_routes, club_routes

security = HTTPBearer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_db(app)          # connexion Atlas au démarrage
    yield                          # app tourne ici
    print("Déconnexion propre")

app = FastAPI(
    title="Postural Platform API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # ton futur frontend React
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_routes.router,  prefix="/api/auth",     tags=["Auth"])
app.include_router(athletes.router,     prefix="/api/athletes", tags=["Athletes"])
app.include_router(club_routes.router,  prefix="/api/clubs",    tags=["Clubs"])
#app.include_router(sessions.router,     prefix="/api/sessions", tags=["Sessions"])

@app.get("/")
async def root():
    return {"status": "ok", "message": "Postural Platform API"}