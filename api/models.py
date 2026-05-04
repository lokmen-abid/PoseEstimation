from beanie import Document
from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict
from datetime import datetime

class User(Document):
    email: EmailStr
    password_hash: str
    role: str = "specialist"       # "admin" | "specialist"
    full_name: str
    status: str = "pending"
    created_at: datetime = datetime.utcnow()

    class Settings:
        name = "users"

class Athlete(Document):
    specialist_id: str
    name: str
    age: int
    sport: str = "tennis"
    dominant_hand: str = "right"
    medical_notes: Optional[str] = None
    created_at: datetime = datetime.utcnow()

    class Settings:
        name = "athletes"

class Session(Document):
    athlete_id: str
    specialist_id: str
    gesture_type: str              # "service" | "coup_droit" | "revers"
    video_url: str
    status: str = "processing"     # "processing" | "completed" | "error"
    fps: int = 30
    total_frames: Optional[int] = None
    created_at: datetime = datetime.utcnow()

    class Settings:
        name = "sessions"

class Frame(Document):
    session_id: str
    frame_number: int
    timestamp_ms: int
    keypoints: Dict                # {"left_knee": {"x":..,"y":..,"z":..}}
    angles: Dict                   # {"knee_flexion": 92.3, ...}

    class Settings:
        name = "frames"

class Metrics(Document):
    session_id: str
    global_score: float
    averages: Dict
    variances: Dict
    alerts: List[Dict] = []
    computed_at: datetime = datetime.utcnow()

    class Settings:
        name = "metrics"