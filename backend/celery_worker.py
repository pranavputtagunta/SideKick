from celery import Celery

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