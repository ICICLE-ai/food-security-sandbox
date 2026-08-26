from pydantic_settings import BaseSettings

class AppSettings(BaseSettings):
    host: str
    port: int
    debug: bool
    tapis_base_url: str
    param_server_url: str
    mongodb_uri: str
    class Config:
        env_file: str = ".env"
        extra = "ignore"

app_settings = AppSettings()