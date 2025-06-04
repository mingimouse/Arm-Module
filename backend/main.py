from fastapi import FastAPI
from arm.backend.endpoints import arm_endpoint

app = FastAPI()
app.include_router(arm_endpoint.router, prefix="/arm")
