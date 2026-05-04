from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from dotenv import load_dotenv
from api.models import User, Athlete, Session, Frame, Metrics
import os

load_dotenv()


async def connect_db(app):
    client = AsyncIOMotorClient(os.getenv("MONGODB_ATLAS_URI"))

    await init_beanie(
        database=client.postural_db,  # ← notation point, pas client["postural_db"]
        document_models=[User, Athlete, Session, Frame, Metrics]
    )
    print("✅ Connecté à MongoDB Atlas")