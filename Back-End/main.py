from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from api.api_router import generate_answer_router
import asyncio
import uvicorn

app = FastAPI(
    title="Doctor's Appointment Agentic Flow",
    version="6.0.0",
    description="Allow users to find and book availbility",
    openapi_url="/openapi.json",
    docs_url="/",
)


@app.get("/health")
async def health_check():
    return PlainTextResponse("OK")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(generate_answer_router)

print("Backend API is running")


asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0")