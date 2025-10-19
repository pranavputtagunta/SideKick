# 30-Day Taekwondo App Development Plan (Local First)

This plan provides a more detailed, beginner-friendly roadmap for developing the Sidekick app. We will focus on building and testing everything on your local machine first before moving to cloud deployment. Each day is planned for approximately 2-3 hours of work.

---

### **Phase 1: Local Backend Development (Weeks 1-2)**

**Objective:** Build the entire backend on your local machine, using Docker for services like the database and message queue to ensure a consistent and easy-to-manage environment.

#### **Week 1: API and Database Setup**

*   **Day 1-2: Setting up the FastAPI Backend**
    *   **Goal:** Initialize your backend project and get a simple API endpoint running.
    *   **Concept:** FastAPI is a modern Python web framework for building APIs. It's fast to code and fast to run. We'll use `uvicorn` as the server to run our FastAPI application.
    *   **Action Steps:**
        1.  Open a terminal in the `backend/` directory.
        2.  Create a Python virtual environment: `python -m venv venv`
        3.  Activate it: `.\venv\Scripts\Activate.ps1` (for PowerShell).
        4.  Install necessary packages: `pip install fastapi "uvicorn[standard]" sqlalchemy psycopg2-binary`
        5.  Create a `main.py` file inside `backend/` and add a basic "Hello World" endpoint to test the setup.
        6.  Run the server: `uvicorn main:app --reload`. You should be able to see your API running at `http://127.0.0.1:8000` in your browser.

*   **Day 3-4: Setting up a Local PostgreSQL Database with Docker**
    *   **Goal:** Get a PostgreSQL database running locally.
    *   **Concept:** Docker allows us to run applications in isolated containers. This is perfect for running a database locally without having to install it directly on your machine. `docker-compose` lets us define and run multi-container Docker applications.
    *   **Action Steps:**
        1.  Install Docker Desktop.
        2.  In the project's root directory, create a `docker-compose.yml` file.
        3.  Add a service for PostgreSQL, specifying the official `postgres` image, setting environment variables for the user, password, and database name, and mapping the port (e.g., `5432:5432`).
        4.  Run `docker-compose up -d` in the terminal to start the database container in the background.
        5.  (Optional) Use a database tool like DBeaver or pgAdmin to connect to your local database (`localhost:5432`) to verify it's working.

*   **Day 5-6: Database Modeling and ORM Setup**
    *   **Goal:** Define your application's data structure in Python and connect your FastAPI app to the database.
    *   **Concept:** An Object-Relational Mapper (ORM) like SQLAlchemy lets you interact with your database using Python classes instead of writing raw SQL. This makes your code more readable and maintainable. Pydantic models are used for data validation and serialization (defining the shape of your API data).
    *   **Action Steps:**
        1.  Create a `database.py` file in `backend/` to handle the database connection logic (SQLAlchemy engine and session).
        2.  Create a `models.py` file to define your database tables as Python classes (e.g., `User`, `Skill`, `UserAttempt`).
        3.  Create a `schemas.py` file to define the Pydantic models that will be used for API request and response validation.
        4.  Update `main.py` to create the database tables on startup.

*   **Day 7: Creating Initial API Endpoints**
    *   **Goal:** Build the first real API endpoints that interact with your database.
    *   **Concept:** API endpoints are the URLs that your frontend will call to get or send data.
    *   **Action Steps:**
        1.  In `main.py`, create an endpoint `GET /belts/{belt_id}/skills` that fetches all skills for a given belt from the database.
        2.  Use the SQLAlchemy session to query the database and the Pydantic schemas to format the response.
        3.  Test your new endpoint using FastAPI's automatic interactive documentation at `http://127.0.0.1:8000/docs`.

#### **Week 2: Core Logic and Asynchronous Tasks**

*   **Day 8-9: User Authentication (Mocked Locally)**
    *   **Goal:** Implement a way to handle users and secure endpoints.
    *   **Concept:** In production, you'll use Firebase to get a secure token. For local development, we'll create a "mock" authentication system that simulates a logged-in user, allowing you to build and test protected endpoints without a frontend.
    *   **Action Steps:**
        1.  Create a dependency function in FastAPI that returns a hardcoded user ID (e.g., `user_id: 1`).
        2.  "Protect" your endpoints by adding this dependency to them.
        3.  Implement the `POST /users/` endpoint to add a new user to your database.

*   **Day 10-11: Setting up Redis and Celery with Docker**
    *   **Goal:** Set up a system for running time-consuming tasks in the background.
    *   **Concept:** The video analysis is too slow to run in a normal API request. We'll use Celery (a task queue) and Redis (a message broker) to manage these long-running jobs. The API will add a "job" to the queue, and a separate Celery "worker" process will pick it up and do the heavy lifting.
    *   **Action Steps:**
        1.  Add Redis and a Celery worker as new services to your `docker-compose.yml` file.
        2.  Create a `celery_worker.py` file to configure Celery and define your tasks.
        3.  Create a simple test task (e.g., one that just prints a message) to verify the setup is working.
        4.  Run `docker-compose up -d --build` to start your new services.

*   **Day 12-13: Implementing the Analysis Pipeline**
    *   **Goal:** Connect your existing analysis scripts to the Celery task.
    *   **Concept:** This is where you integrate your Python expertise. The Celery task will act as an orchestrator.
    *   **Action Steps:**
        1.  Create an endpoint `POST /analysis/start` that receives a (for now, local) video file path and dispatches a job to Celery. It should immediately return a `job_id`.
        2.  In your Celery task, use your existing Python scripts (`pose_extractor.py`, `csv_comparer.py`) to perform the analysis.
        3.  For now, use the sample CSV files you already have. The task will calculate the score and save it to the `UserAttempts` table in your database.

*   **Day 14: Integrating Generative Feedback**
    *   **Goal:** Add the LLM call to your analysis pipeline to generate feedback.
    *   **Action Steps:**
        1.  Add a step in your Celery task that calls the Vertex AI Gemini API after the score has been calculated.
        2.  Use a well-crafted prompt that includes the score and analysis details to get personalized feedback.
        3.  Store this feedback in the `UserAttempts` table along with the score.

---

### **Phase 2: Local Frontend Development (Weeks 3-4)**

**Objective:** Build the mobile app interface and connect it to your local backend.

#### **Week 3: Screens & Navigation**

*   **Day 15-16: React Native Setup & Onboarding**
    *   **Goal:** Initialize the frontend project and build the sign-up flow.
    *   **Concept:** We'll use Expo, a framework that makes developing React Native apps much easier. We will use React Navigation for moving between screens.
    *   **Action Steps:**
        1.  Initialize a new Expo project: `npx create-expo-app frontend`.
        2.  Set up React Navigation with a basic stack navigator.
        3.  Build the `OnboardingScreen` and `QuestionsScreen`.
        4.  For now, a simple "Sign Up" button will call your local backend's `POST /users/` endpoint to create a user. We'll integrate real Firebase Auth later.

*   **Day 17-18: Dashboard and State Management**
    *   **Goal:** Build the main dashboard and manage application-wide data.
    *   **Concept:** State management libraries (like Zustand or Redux Toolkit) help you manage data that needs to be accessed by multiple screens, like the logged-in user's information.
    *   **Action Steps:**
        1.  Build the `DashboardScreen`.
        2.  When the screen loads, make an API call to your local backend (`http://<YOUR_LOCAL_IP>:8000/belts/.../skills`) to fetch the list of moves.
        3.  Set up Zustand to store user profile information globally.

*   **Day 19-21: Demo, Trial, and Recording Flow**
    *   **Goal:** Build the core user experience of watching, recording, and uploading.
    *   **Action Steps:**
        1.  Create the `DemoScreen` to play the expert video.
        2.  Create the `TrialScreen` using `expo-camera` for the recording interface.
        3.  Implement the upload logic. This is tricky locally. You'll need to make the `POST /analysis/start` endpoint accept a file upload directly for now.
        4.  After the upload is complete and the analysis is triggered, navigate to the `ResultsScreen`.

#### **Week 4: Finalizing the Loop and Preparing for Cloud**

*   **Day 22-23: Results Screen**
    *   **Goal:** Display the analysis results to the user.
    *   **Concept:** The frontend will need to "poll" the backend to check if the analysis is done.
    *   **Action Steps:**
        1.  Create an endpoint `GET /analysis/results/{job_id}` on your backend.
        2.  On the `ResultsScreen`, use a `useEffect` hook to call this endpoint every few seconds.
        3.  Once the results are ready, display the score, feedback, and videos.

*   **Day 24-25: UI Polish and Refinement**
    *   **Goal:** Make the app look and feel like your Figma designs.
    *   **Action Steps:**
        1.  Review all screens and clean up the styling.
        2.  Ensure the navigation flow is smooth and intuitive.
        3.  Fix any bugs in the local user journey.

*   **Day 26-28: Preparing for Cloud Deployment**
    *   **Goal:** Get ready to move your local setup to the cloud.
    *   **Concept:** We will now transition from local file paths and mocked auth to real cloud services.
    *   **Action Steps:**
        1.  **Backend:**
            *   Implement the real Firebase Authentication dependency to verify ID tokens.
            *   Implement the signed URL logic for Google Cloud Storage uploads.
            *   Update your Celery task to download videos from Cloud Storage instead of reading from a local path.
        2.  **Frontend:**
            *   Integrate the Firebase SDK for authentication.
            *   Update the upload logic to use the signed URL to upload directly to Google Cloud Storage.

*   **Day 29-30: Cloud Deployment & Testing**
    *   **Goal:** Deploy all services to Google Cloud Platform.
    *   **Action Steps:**
        1.  Follow the original plan's deployment steps: Deploy the database to **Cloud SQL**, the queue to **Memorystore**, and the FastAPI/Celery containers to **Cloud Run**.
        2.  Build your frontend using **EAS Build** and test the live application.
        3.  Use this time to troubleshoot any issues that arise from the cloud environment.

This detailed, local-first plan will give you a solid foundation and help you understand each piece of the stack before you have to worry about the complexities of the cloud. Good luck!
