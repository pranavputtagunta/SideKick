from pydantic import BaseModel
from typing import List, Optional

# Schemas for UserAttempt
class UserAttemptBase(BaseModel):
    user_id: int
    skill_id: int
    status: str
    score: Optional[float] = None
    user_video_url: Optional[str] = None
    feedback: Optional[str] = None

class UserAttemptCreate(UserAttemptBase):
    pass

class UserAttempt(UserAttemptBase):
    id: int

    class Config:
        from_attributes = True

# Schemas for Skill
class SkillBase(BaseModel):
    name: str
    expert_video_url: Optional[str] = None
    expert_landmarks_url: Optional[str] = None
    masters_tips: Optional[str] = None

class SkillCreate(SkillBase):
    belt_id: int

class Skill(SkillBase):
    id: int
    belt_id: int
    attempts: List["UserAttempt"] = []

    class Config:
        from_attributes = True

# Schemas for Belt
class BeltBase(BaseModel):
    name: str
    rank_order: int

class BeltCreate(BeltBase):
    pass

class Belt(BeltBase):
    id: int
    skills: List[Skill] = []

    class Config:
        from_attributes = True

# Schemas for User
class UserBase(BaseModel):
    email: str
    name: str

class UserCreate(UserBase):
    pass

class User(UserBase):
    id: int
    current_belt_id: Optional[int] = None
    attempts: List[UserAttempt] = []

    class Config:
        from_attributes = True
