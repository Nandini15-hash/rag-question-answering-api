from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from app.api import routes_documents, routes_query
from app.middleware.rate_limit import limiter

app = FastAPI(
    title="RAG Question Answering API",
    description=(
        "Upload PDF/TXT documents, they're chunked + embedded in the background, "
        "then ask questions answered via retrieval-augmented generation."
    ),
    version="1.0.0",
)

app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": f"Rate limit exceeded: {exc.detail}"})


app.include_router(routes_documents.router)
app.include_router(routes_query.router)


@app.get("/health", tags=["meta"])
async def health():
    return {"status": "ok"}
