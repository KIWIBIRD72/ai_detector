from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import clustering, dashboard, detector

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(clustering.router)
app.include_router(dashboard.router)
app.include_router(detector.router)