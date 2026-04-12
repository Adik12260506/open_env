# server/app.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from openenv.core.env_server.http_server import create_app
from server.env import EmailEnv
from models import EmailAction, EmailObservation


def create_email_env():
    return EmailEnv()


# create_app from openenv-core auto-registers tasks and exposes /tasks endpoint
app = create_app(
    create_email_env,
    EmailAction,
    EmailObservation,
    env_name="email_env",
    max_concurrent_envs=1,
)

# Serve web UI on top of the openenv app
_WEB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")

@app.get("/web", include_in_schema=False)
@app.get("/web/", include_in_schema=False)
def web_index():
    return FileResponse(os.path.join(_WEB, "index.html"))

@app.get("/", include_in_schema=False)
def root():
    return FileResponse(os.path.join(_WEB, "index.html"))


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()