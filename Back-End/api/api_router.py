from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from api.endpoints.v1 import generate_answer

generate_answer_router = APIRouter()

generate_answer_router.include_router(
    generate_answer.router, 
    prefix="/api/v1", 
    tags=["generate_answer"]
)
