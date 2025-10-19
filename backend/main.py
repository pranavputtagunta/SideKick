from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session

import models, schemas
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
