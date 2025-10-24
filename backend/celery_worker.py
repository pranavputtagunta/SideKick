from celery import Celery

from database import SessionLocal
import crud
from analysis import pose_extractor, csv_comparer

celery_app = Celery(
    "tasks", # name of current module
    broker="redis://localhost:6379/0", # URL of message broker
    backend="redis://localhost:6379/0", # Where to store task results
)

# Optional config
celery_app.conf.update(
    task_track_started= True,
)

# test task
@celery_app.task(name="tasks.add_together")
def add_together(a: int, b: int) -> int: 
    print(f"adding {a} + {b}")
    return a + b

# Analysis task
@celery_app.task(name="tasks.analyze_video")
def analyze_video(user_attempt_id: int, video_path: str, csv_path: str) -> dict: 
    # Placeholder currently, implement analysis logic later
    print(f"--- Starting analysis for attempt {user_attempt_id} on video {video_path} ---")

    # Real analysis logic: 
    # 1. Update attempt status to "processing"

    db = SessionLocal()

    try: 
        # 2. Get user attempt from DB
        attempt = db.query(crud.models.UserAttempt).filter(crud.models.UserAttempt.id == user_attempt_id).first()
        if not attempt:
            print(f"Error: UserAttempt with ID {user_attempt_id} not found.")
            return
    
        # 3. Update attempt status to processing
        attempt.status = "PROCESSING"
        db.commit()
        print(f"Processing attempt {user_attempt_id}...")

        # 4. process video to extract landmarks
        print("Extracting user landmarks...")
        pose_extractor.analyze_video(video_path, csv_path)
        print(f"User landmarks saved to {csv_path}")

        # 5. get expert's landmarks for the skill
        skill = crud.get_skill(db, attempt.skill_id)
        if not skill or not skill.expert_landmarks_url: 
            raise ValueError(f"Expert landmarks not found for skill ID {attempt.skill_id}")
        
        expert_landmarks_path = skill.expert_landmarks_url # use local path for now
        print(f"Comparing with expert landmarks from {expert_landmarks_path}...")

        # 6. compare user landmarks with expert's
        comparison = csv_comparer.compare_csv(expert_landmarks_path, csv_path)
        score = comparison["accuracy_score"]
        print(f"Comparison complete. Score: {score}")

        # 7. Update database with results
        attempt.status = "COMPLETE"
        attempt.score = score

        # Add LLM generated feedback later
        attempt.feedback = "Great job! Keep your back straight and maintain balance." # placeholder feedback
        db.commit()


        print(f"--- Completed analysis for attempt {user_attempt_id} ---")

    except Exception as e:
        # Mark attempt as failed if anything goes wrong
        print(f"Error during analysis of attempt {user_attempt_id}: {e}")
        attempt.status = "FAILED"
        attempt.feedback = f"Analysis failed: {e}"
        db.commit()

    finally: 
        db.close()

    # For now, return dummy result
    result = {
        "status": "complete"
    }

    return result