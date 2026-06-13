from beanie import Document
from pydantic import BaseModel, EmailStr , Field
from typing import Optional, List, Dict, Any
from datetime import datetime , timezone

def _now() -> datetime:
    """Retourne l'heure UTC actuelle — remplace datetime.utcnow() déprécié."""
    return datetime.now(timezone.utc)

class User(Document):
    email: EmailStr
    password_hash: str
    role: str = "specialist"       # "admin" | "specialist"
    full_name: str
    status: str = "pending"
    club_id: Optional[str] = None  # ← None = spécialiste indépendant
    created_at: datetime =Field(default_factory=_now)

    class Settings:
        name = "users"

class Athlete(Document):
    specialist_id: str
    name: str
    age: int
    sex: str = "male"
    sport: str = "tennis"
    dominant_hand: str = "right"
    medical_notes: Optional[str] = None
    created_at: datetime = Field(default_factory=_now)

    class Settings:
        name = "athletes"

class Session(Document):
    athlete_id: str
    specialist_id: str
    gesture_type: str  # "service" | "coup_droit" | "revers"
    video_url: str
    status: str = "created"  # "created" | "ready" | "processing" | "completed" | "error"
    fps: int = 30
    total_frames: Optional[int] = None
    phase_annotations: Optional[Dict[str, int]] = None  # {"trophy_position": 145, ...}
    created_at: datetime = Field(default_factory=_now)

    class Settings:
        name = "sessions"


class KeypointData(BaseModel):
    x: float
    y: float
    z: float
    visibility: float


class FrameAngles(BaseModel):
    # Angles communs aux 3 gestes
    knee_flexion_right: Optional[float] = None
    knee_flexion_left: Optional[float] = None
    trunk_inclination: Optional[float] = None
    trunk_rotation: Optional[float] = None
    hip_right: Optional[float] = None
    hip_left: Optional[float] = None

    # Service spécifique
    shoulder_rotation_right: Optional[float] = None
    shoulder_elevation_right: Optional[float] = None
    elbow_right: Optional[float] = None

    # Forehand spécifique
    pelvis_rotation: Optional[float] = None
    shoulder_separation: Optional[float] = None

    # Backhand spécifique
    wrist_extension_right: Optional[float] = None
    wrist_extension_left: Optional[float] = None
    elbow_left: Optional[float] = None


class Frame(Document):
    session_id: str
    frame_number: int
    timestamp_ms: int
    phase: Optional[str] = None  # "trophy_position" | "ball_impact" | "rlp" | None
    pipeline_mode: str  # "ex2_3d_mavg" | "ex3_3d_savgol"

    # 17 keypoints utiles seulement (pas les 33)
    keypoints: Dict[str, KeypointData]
    angles: FrameAngles

    class Settings:
        name = "frames"


class JointMetrics(BaseModel):
    min: float
    max: float
    mean: float
    std: float


class ClinicalAlert(BaseModel):
    joint: str
    value: float
    threshold: float
    reference: str  # "Gorce2024" | "Knudson2001" | "Elliott2008"
    severity: str  # "warning" | "critical"


class Metrics(Document):
    session_id: str
    gesture_type: str  # "serve" | "forehand" | "backhand"
    pipeline_mode: str  # "ex2" | "ex3"
    total_frames: int
    phases_detected: Dict[str, int]  # {"trophy_position": 145, ...}

    joint_metrics: Dict[str, JointMetrics]  # stats par articulation sur toute la session
    normative_comparison: Dict[str, Any]  # {"knee_flexion_right": +3.2, ...} écart % vs Gorce
    alerts: List[ClinicalAlert]  # vide pour l'instant, on valide les angles d'abord

    computed_at: datetime = Field(default_factory=_now)

    class Settings:
        name = "metrics"

class Club(Document):
    name: str
    city: Optional[str] = None
    created_at: datetime = Field(default_factory=_now)

    class Settings:
        name = "clubs"