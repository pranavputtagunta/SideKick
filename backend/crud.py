from sqlalchemy.orm import Session
import models, schemas

def get_user(db: Session, user_id: int): 
    # This queries the User table, filters by the user's ID, and returns the first result.
    return db.query(models.User).filter(models.User.id == user_id).first()

def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

def get_users(db: Session, skip: int = 0, limit: int = 100):
    # This queries the User table, skips a certain number of records, limits the results, and returns all matches. 
    return db.query(models.User).offset(skip).limit(limit).all()

def create_user(db: Session, user: schemas.UserCreate):
    # Create an SQLAlchemy model instance from Pydantic schema data.
    db_user = models.User(email=user.email, name=user.name)
    # Add that instance to database session.
    db.add(db_user)
    # Commit the changes to the database. 
    db.commit()
    # Refresh instance so it contains new data from database, like generated ID. 
    db.refresh(db_user)
    return db_user

# Get skill by id
def get_skill(db: Session, skill_id: int): 
    return db.query(models.Skill).filter(models.Skill.id == skill_id).first()

def get_skills_by_belt(db: Session, belt_id: int, skip: int = 0, limit: int = 100):
    return db.query(models.Skill).filter(models.Skill.belt_id == belt_id).offset(skip).limit(limit).all()

def create_skill_for_belt(db: Session, skill: schemas.SkillCreate, belt_id: int):
    db_skill = models.Skill(**skill.model_dump(), belt_id=belt_id)
    db.add(db_skill)
    db.commit()
    db.refresh(db_skill)
    return db_skill

# Analysis crud operations
def create_user_attempt(db: Session, skill_id: int, user_id: int): 
    db_attempt = models.UserAttempt(
        skill_id = skill_id,
        user_id = user_id,
        status = 'pending',
    )
    db.add(db_attempt)
    db.commit()
    db.refresh(db_attempt)
    return db_attempt