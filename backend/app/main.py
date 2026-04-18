from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth import router as auth_router
from app.api.chat import router as chat_router
from app.api.services import router as services_router
from app.api.location import router as location_router

app = FastAPI(
    title="AI Public Administration API",
    version="1.0.0",
    swagger_ui_parameters={"persistAuthorization": True},
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Update with specific frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(services_router)
app.include_router(location_router)

@app.get("/")
async def read_root():
    return {"message": "Backend is running!"}