from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session

import models, schemas, crud
from database import SessionLocal, engine

# This command tells SQLAlchemy to create all the tables defined in models.py
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# Dependency to get a DB session for each request
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_current_user(db: Session = Depends(get_db)):
    # simulate being logged in as user with id 1
    user = crud.get_user(db, user_id=1)
    if user is None: 
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.get("/")
def read_root():
    return {"Status": "Running"}

# Example endpoint to create a belt
@app.post("/belts/", response_model=schemas.Belt)
def create_belt(belt: schemas.BeltCreate, db: Session = Depends(get_db)):
    db_belt = models.Belt(name=belt.name, rank_order=belt.rank_order)
    db.add(db_belt)
    db.commit()
    db.refresh(db_belt)
    return db_belt

# Example endpoint to read belts
@app.get("/belts/", response_model=list[schemas.Belt])
def read_belts(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    belts = db.query(models.Belt).offset(skip).limit(limit).all()
    return belts

# Example endpoint to create a skill for a belt
@app.post("/belts/{belt_id}/skills/", response_model=schemas.SkillResponse)
def create_skill_for_belt(belt_id: int, skill: schemas.SkillCreate, db: Session = Depends(get_db)):
    return crud.create_skill_for_belt(db=db, skill=skill, belt_id=belt_id)

# Example endpoint to read skills by belt
@app.get("/belts/{belt_id}/skills", response_model=list[schemas.Skill])
def read_skills_by_belt(belt_id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return crud.get_skills_by_belt(db, belt_id=belt_id, skip=skip, limit=limit)

# USER ENDPOINTS

@app.post("/users/", response_model= schemas.UserResponse)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    # Check if a user with same email already exists
    db_user = crud.get_user_by_email(db, email= user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")         
    return crud.create_user(db=db, user=user)

@app.get("/users/", response_model=list[schemas.User])
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    # Calls the crud function to get users
    users = crud.get_users(db, skip=skip, limit=limit)
    return users

@app.get("/users/{user_id}", response_model=schemas.User)
def read_user(user_id: int, db: Session = Depends(get_db)):
    # Gets a single user by ID with crud function
    db_user = crud.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

# Login endpoints
@app.get("/users/me/", response_model=schemas.User)
async def read_current_user(current_user: schemas.User = Depends(get_current_user)):
    return current_user