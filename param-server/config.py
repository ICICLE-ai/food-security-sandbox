from pydantic_settings import BaseSettings
class Settings(BaseSettings):
    tapis_base_url: str

settings = Settings()