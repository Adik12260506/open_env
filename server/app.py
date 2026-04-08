# server/app.py
from openenv.core.env_server.http_server import create_app
from .env import EmailEnv
from email_env.models import EmailAction, EmailObservation


def create_email_env():
    return EmailEnv()


app = create_app(
    create_email_env,
    EmailAction,
    EmailObservation,
    env_name="email_env",
    max_concurrent_envs=1,
)


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()