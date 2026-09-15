import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from app.config import settings
from app.ingestion.jobs import get_job_store, get_worker_pool
from app.ingestion.loaders import UnsupportedFileType
from app.middleware.rate_limit import limiter
from app.models import DocumentSummary, JobStatus, JobStatusResponse, UploadResponse

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_SUFFIXES = {".pdf", ".txt"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB


@router.post("/upload", response_model=UploadResponse, status_code=202)
@limiter.limit(settings.rate_limit)
async def upload_document(request: Request, file: UploadFile):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_SUFFIXES)}")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(400, "Uploaded file is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File exceeds max size of {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")

    document_id = str(uuid.uuid4())
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest_path = upload_dir / f"{document_id}{suffix}"
    dest_path.write_bytes(contents)

    job_store = get_job_store()
    job_id = job_store.create(document_id=document_id, filename=file.filename, filepath=str(dest_path))
    get_worker_pool().submit(job_id)

    return UploadResponse(job_id=job_id, document_id=document_id, filename=file.filename, status=JobStatus.QUEUED)


@router.get("/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    job = get_job_store().get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found.")
    return JobStatusResponse(
        job_id=job["job_id"],
        document_id=job["document_id"],
        filename=job["filename"],
        status=job["status"],
        chunk_count=job["chunk_count"],
        error=job["error"],
        created_at=job["created_at"],
        updated_at=job["updated_at"],
    )


@router.get("", response_model=list[DocumentSummary])
async def list_documents():
    jobs = get_job_store().list_documents()
    return [
        DocumentSummary(
            document_id=j["document_id"],
            filename=j["filename"],
            status=j["status"],
            chunk_count=j["chunk_count"],
            created_at=j["created_at"],
        )
        for j in jobs
    ]
