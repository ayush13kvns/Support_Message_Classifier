import hashlib
import io
import hmac
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
import joblib
import sklearn
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool
from app.text import normalize

load_dotenv()
logger = logging.getLogger("classifier")
logging.basicConfig(level=logging.INFO,format="%(message)s")

class Message(BaseModel):
    text: str = Field(min_length=1,max_length=10000)
    @field_validator("text")
    @classmethod
    def nonblank(cls,value):
        if not value.strip(): raise ValueError("Text must not be blank")
        return value

class Batch(BaseModel):
    messages: list[Message] = Field(min_length=1,max_length=100)

class Prediction(BaseModel):
    category: str
    confidence: float
    probabilities: dict[str,float]
    requires_review: bool
    review_reasons: list[str]
    model_version: str
    demo_model: bool


def create_app():
    @asynccontextmanager
    async def lifespan(app):
        key=os.getenv("API_KEY","")
        environment=os.getenv("APP_ENV","development")
        if environment not in {"development","production"}: raise RuntimeError("Invalid APP_ENV")
        if environment=="production" and len(key)<32: raise RuntimeError("Production requires API_KEY of at least 32 characters")
        threshold=float(os.getenv("CONFIDENCE_THRESHOLD","0.65"))
        if not 0 <= threshold <= 1: raise RuntimeError("Invalid confidence threshold")
        folder=Path(os.getenv("MODEL_DIR","models"))
        metadata=json.loads((folder/"metadata.json").read_text())
        artifact=(folder/"model.joblib").read_bytes()
        if hashlib.sha256(artifact).hexdigest()!=metadata["sha256"]: raise RuntimeError("Model checksum mismatch")
        if metadata["sklearn_version"]!=sklearn.__version__: raise RuntimeError("Model/library version mismatch")
        if environment=="production" and metadata["demo_data"]: raise RuntimeError("Train a real-data model before production startup")
        # Only trusted, administrator-controlled artifacts are safe to deserialize.
        model=joblib.load(io.BytesIO(artifact))
        if sorted(model.classes_)!=metadata["labels"]: raise RuntimeError("Model label mismatch")
        app.state.model=model; app.state.metadata=metadata
        app.state.threshold=threshold; app.state.key=key
        yield
    app=FastAPI(title="Support Message Classifier",version="1.0.0",lifespan=lifespan)

    @app.middleware("http")
    async def observe(request: Request, call_next):
        started=time.perf_counter(); request_id=uuid.uuid4().hex
        response=await call_next(request)
        response.headers["X-Request-ID"]=request_id
        response.headers["X-Content-Type-Options"]="nosniff"
        logger.info(json.dumps({"request_id":request_id,"method":request.method,"status":response.status_code,"duration_ms":round((time.perf_counter()-started)*1000,2)}))
        return response

    # Do not echo invalid messages or input bodies in validation errors.
    from fastapi.exceptions import RequestValidationError
    @app.exception_handler(RequestValidationError)
    async def validation_error(request,exc):
        return JSONResponse(status_code=422,content={"detail":"Invalid request. Check text length, nonblank input, and batch size."})

    async def authorize(x_api_key: str | None = Header(default=None)):
        key=app.state.key
        if key and (x_api_key is None or not hmac.compare_digest(x_api_key.encode(),key.encode())):
            raise HTTPException(status_code=401,detail="Invalid API key")

    def predict(messages):
        texts=[normalize(m.text) for m in messages]
        vectors=app.state.model.named_steps["tfidf"].transform(texts)
        probabilities=app.state.model.named_steps["classifier"].predict_proba(vectors)
        results=[]
        for i, probs in enumerate(probabilities):
            top=int(probs.argmax()); confidence=float(probs[top]); reasons=[]
            if confidence < app.state.threshold: reasons.append("low_confidence")
            if vectors[i].nnz==0: reasons.append("no_known_vocabulary")
            if app.state.metadata["demo_data"]: reasons.append("demo_model")
            results.append(Prediction(category=str(app.state.model.classes_[top]),confidence=confidence,
                probabilities={str(k):float(v) for k,v in zip(app.state.model.classes_,probs)},
                requires_review=bool(reasons),review_reasons=reasons,model_version=app.state.metadata["version"],demo_model=app.state.metadata["demo_data"]))
        return results

    @app.get("/health/live")
    def live(): return {"status":"ok"}
    @app.get("/health/ready")
    def ready(): return {"status":"ready","model_version":app.state.metadata["version"],"demo_model":app.state.metadata["demo_data"]}
    @app.post("/v1/classify",response_model=Prediction,dependencies=[Depends(authorize)])
    async def classify(message:Message): return (await run_in_threadpool(predict,[message]))[0]
    @app.post("/v1/classify/batch",response_model=list[Prediction],dependencies=[Depends(authorize)])
    async def batch(payload:Batch): return await run_in_threadpool(predict,payload.messages)
    return app

app=create_app()
