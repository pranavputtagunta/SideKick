from sqlalchemy import Column, ForeignKey, Integer, String, Float, Text
from sqlalchemy.orm import relationship

from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True) # Will replace with Firebase UID later
    current_belt_id = Column(Integer, ForeignKey("belts.id"), nullable=True)

    current_belt = relationship("Belt", lazy="selectin")
    attempts = relationship("UserAttempt", back_populates="user", lazy="selectin")

class Belt(Base):
    __tablename__ = "belts"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    rank_order = Column(Integer, unique=True)

    skills = relationship("Skill", back_populates="belt", lazy="selectin")

class Skill(Base):
    __tablename__ = "skills"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    belt_id = Column(Integer, ForeignKey("belts.id"))
    expert_video_url = Column(String, nullable=True)
    expert_landmarks_url = Column(String, nullable=True)
    masters_tips = Column(Text, nullable=True)

    belt = relationship("Belt", back_populates="skills", lazy="selectin")
    attempts = relationship("UserAttempt", back_populates="skill", lazy="selectin")

class UserAttempt(Base):
    __tablename__ = "user_attempts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    skill_id = Column(Integer, ForeignKey("skills.id"))
    score = Column(Float, nullable=True)
    status = Column(String) # e.g., 'pending', 'complete', 'failed'
    user_video_url = Column(String, nullable=True)
    feedback = Column(Text, nullable=True)

    user = relationship("User", back_populates="attempts", lazy="selectin")
    skill = relationship("Skill", back_populates="attempts", lazy="selectin")
