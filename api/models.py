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
    error_message: Optional[str] = None

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
    phase: str
    note: str


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


class MatchSession(Document):
    """
    Équivalent de Session mais pour un match complet.
    Une MatchSession = une vidéo de match entier (plusieurs sets).
    Workflow identique : created → ready → processing → completed | error
    """
    athlete_id: str
    specialist_id: str

    video_url: str
    status: str = "created"  # "created" | "ready" | "processing" | "completed" | "error"

    # Infos vidéo remplies après upload
    fps: int = 30
    total_frames: Optional[int] = None
    duration_seconds: Optional[float] = None

    # Contexte du match (renseigné par le spécialiste)
    surface: str = "hard"  # "hard" | "clay" | "grass"
    match_format: str = "best_of_3"  # "best_of_3" | "best_of_5" | "pro_set"
    opponent_name: Optional[str] = None
    location: Optional[str] = None
    notes: Optional[str] = None

    created_at: datetime = Field(default_factory=_now)
    error_message: Optional[str] = None

    class Settings:
        name = "match_sessions"


class GestureEvent(BaseModel):
    """
    Un événement de frappe détecté dans la vidéo.
    Produit par le pipeline analyze_match.py pour chaque coup joué.
    """
    frame_number: int
    timestamp_ms: int
    gesture_type: str  # "forehand" | "backhand" | "serve" | "volley" | "unknown"
    confidence: float  # 0.0 – 1.0 (confiance du classificateur)
    set_number: int  # 1 | 2 | 3 | ...
    point_number: int  # numéro du point dans le set
    rally_stroke: int  # numéro de la frappe dans l'échange (1 = service, 2 = retour, ...)

    # Position du joueur au moment de la frappe (pixel, normalisé 0–1)
    # Rempli si YOLO joueur disponible, None sinon
    player_x: Optional[float] = None  # 0 = gauche, 1 = droite du court
    player_y: Optional[float] = None  # 0 = fond, 1 = filet


class RallyStats(BaseModel):
    """Statistiques d'un échange (rally) individuel."""
    point_number: int
    set_number: int
    winner: str  # "player" | "opponent" | "unknown"
    duration_frames: int
    duration_seconds: float
    stroke_count: int  # nombre total de frappes dans l'échange
    gesture_sequence: List[str]  # ex: ["serve", "forehand", "backhand", "forehand"]
    outcome: str  # "winner" | "unforced_error" | "forced_error" | "unknown"


class SetStats(BaseModel):
    """
    Statistiques agrégées pour un set entier.
    Calculées par le pipeline à partir des GestureEvent et RallyStats.
    """
    set_number: int
    points_won: int
    points_lost: int

    # Distribution des gestes (nombre de fois joué dans ce set)
    gesture_counts: Dict[str, int]  # {"forehand": 42, "backhand": 28, "serve": 12, ...}
    gesture_pct: Dict[str, float]  # {"forehand": 55.3, "backhand": 36.8, ...}

    # Stats d'échanges
    total_rallies: int
    avg_rally_length: float  # durée moyenne en secondes
    avg_strokes_per_rally: float

    # Points selon longueur d'échange
    short_rally_wins: int  # 1–3 frappes
    medium_rally_wins: int  # 4–8 frappes
    long_rally_wins: int  # 9+ frappes

    # Geste le plus utilisé dans ce set
    dominant_gesture: str


class MatchMetrics(Document):
    """
    Résultats complets du pipeline analyze_match.py.
    Un seul document par MatchSession — créé à la fin du traitement.
    Équivalent de Metrics pour les sessions biomécaniques.
    """
    match_session_id: str
    athlete_id: str

    # Infos générales
    total_frames_analyzed: int
    total_points_detected: int
    total_rallies_detected: int
    sets_detected: int  # nombre de sets détectés automatiquement

    # Événements bruts (une entrée par frappe détectée)
    gesture_events: List[GestureEvent]

    # Échanges individuels
    rallies: List[RallyStats]

    # Stats par set (clé = "1", "2", "3", ...)
    sets: Dict[str, SetStats]

    # Stats globales sur tout le match
    overall_gesture_counts: Dict[str, int]
    overall_gesture_pct: Dict[str, float]
    dominant_gesture_match: str  # geste le plus utilisé sur tout le match

    avg_rally_length_seconds: float
    avg_strokes_per_rally: float

    # Points forts et faibles (calculés automatiquement)
    # Format : [{"aspect": "...", "detail": "...", "value": ...}, ...]
    strengths: List[Dict[str, Any]]
    weaknesses: List[Dict[str, Any]]

    # Pipeline info
    pipeline_version: str = "v1"
    computed_at: datetime = Field(default_factory=_now)

    class Settings:
        name = "match_metrics"
