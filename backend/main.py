from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from celery.result import AsyncResult

import os
import shutil

import models, schemas, crud
from database import SessionLocal, engine
from celery_worker import *

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

# Celery task endpoint
@app.post("/tasks/add/")
def run_add_task(a: int, b: int):
    task = add_together.delay(a, b) # .delay() sends call as message to queue
    return {"task_id": task.id} # return task's id, not output

@app.get("/tasks/result/{task_id}")
def get_task_result(task_id: str):
    task_result = AsyncResult(task_id, app= celery_app) # get result by task id

    return {
        "task_id": task_id,
        "status": task_result.status, # PENDING, STARTED, SUCCESS, FAILURE
        "result": task_result.result # Contains error if failed, result if successful
    }

# Analysis endpoints
@app.post("/analysis/start/{skill_id}/")
async def start_analysis(
    skill_id: int,
    db: Session = Depends(get_db), 
    current_user: schemas.User = Depends(get_current_user), 
    video_file: UploadFile = File(...)
): 
    # 1. Create UserAttempt in DB
    attempt = crud.create_user_attempt(db, skill_id=skill_id, user_id=current_user.id)

    # 2. Save video to temp location
    temp_dir = "temp_videos"
    temp_csv_dir = "temp_csvs" 
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(temp_csv_dir, exist_ok=True)
    video_path = os.path.join(temp_dir, f"attempt_{attempt.id}_{video_file.filename}")
    temp_csv_path = os.path.join(temp_csv_dir, f"attempt_{attempt.id}.csv")

    with open(video_path, "wb") as buf: 
        shutil.copyfileobj(video_file.file, buf) 

    # 3. Call the Celery task to analyze video
    task = analyze_video.delay(user_attempt_id = attempt.id, video_path= video_path, csv_path=temp_csv_path)

    # 4. Return the task ID to the client
    return {
        "task_id": task.id,
        "user_attempt_id": attempt.id,
    }

