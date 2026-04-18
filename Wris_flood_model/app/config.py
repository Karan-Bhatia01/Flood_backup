from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    log_level: str = "INFO"
    MODEL_PATH:   str = "models/flood_model.pkl"
    GROQ_API_KEY: str = ""
    GROQ_MODEL:   str = "llama3-8b-8192"
    APP_ENV:      str = "development"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
