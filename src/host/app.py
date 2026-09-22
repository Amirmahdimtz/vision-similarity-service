from fastapi import FastAPI

from src.application.web import WebService
from src.infrastructure.di.bootstrap import bootstrap_di


bootstrap_di()
app: FastAPI = WebService().create_app()
