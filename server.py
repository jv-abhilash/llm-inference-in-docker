# server.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import json
import re
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- FastAPI App ----------------
app = FastAPI(title="LLM Router Server")

# ---------------- Env ----------------
MODEL_ID: str = os.getenv("ROUTER_MODEL", "Qwen/Qwen2.5-Math-7B-Instruct")
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN") or None
# IMPORTANT: default to empty so we don't force a non-existent local path
LOCAL_MODEL_DIR: str = os.getenv("LOCAL_MODEL_DIR", "")

# ---------------- Globals ----------------
_tok = None
_model = None

# ---------------- Model Load ----------------
def _load_model():
    """Load tokenizer + model (4-bit on CUDA if available)."""
    global _tok, _model
    if _tok is not None and _model is not None:
        return

    # Decide source: local directory if it exists, otherwise hub repo ID
    use_local = bool(LOCAL_MODEL_DIR) and os.path.isdir(LOCAL_MODEL_DIR)
    source = LOCAL_MODEL_DIR if use_local else MODEL_ID

    device_has_cuda = torch.cuda.is_available()
    device_map = "auto" if device_has_cuda else "cpu"
    # Use dtype= (torch_dtype is deprecated)
    dtype = torch.float16 if device_has_cuda else torch.float32

    quant_config = None
    if device_has_cuda:
        try:
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        except Exception as e:
            logger.warning(f"bitsandbytes unavailable, falling back to full precision: {e}")
            quant_config = None
            # On modern NVIDIA, bf16 is fine too; stick to fp16 to be conservative.

    logger.info(
        f"Loading {MODEL_ID} from "
        f"{'local dir: ' + LOCAL_MODEL_DIR if use_local else 'Hugging Face Hub'} "
        f"on {'CUDA' if device_has_cuda else 'CPU'}"
    )

    tok_kwargs = dict(trust_remote_code=True)
    mdl_kwargs = dict(
        trust_remote_code=True,
        device_map=device_map,
        dtype=dtype,                       # <- updated to dtype=
        quantization_config=quant_config,  # None on CPU or if bnb not present
    )
    if use_local:
        tok_kwargs["local_files_only"] = True
        mdl_kwargs["local_files_only"] = True
    else:
        if HF_TOKEN:
            tok_kwargs["token"] = HF_TOKEN
            mdl_kwargs["token"] = HF_TOKEN

    _tok = AutoTokenizer.from_pretrained(source, **tok_kwargs)
    _model = AutoModelForCausalLM.from_pretrained(source, **mdl_kwargs)

    logger.info("Model ready.")

# ---------------- Generation ----------------
def _generate(prompt: str, max_new_tokens: int = 100, temperature: float = 0.1, do_sample: bool = False) -> str:
    """Generate text using the loaded model."""
    inputs = _tok(prompt, return_tensors="pt")
    # Move to the model's device
    inputs = {k: v.to(_model.device) for k, v in inputs.items()}

    eos_id = _tok.eos_token_id
    pad_id = eos_id if _tok.pad_token_id is None else _tok.pad_token_id

    with torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )
    return _tok.decode(out[0], skip_special_tokens=True)

def _cleanup():
    """Free up GPU memory when shutting down."""
    global _tok, _model
    if _model is not None:
        try:
            logger.info("Offloading model from GPU/CPU...")
            _model = _model.cpu()
        except Exception:
            pass
        del _model
    _model = None
    _tok = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model offloaded successfully.")

# ---------------- Lifecycle ----------------
@app.on_event("startup")
def on_startup():
    _load_model()

@app.on_event("shutdown")
def on_shutdown():
    logger.info("Shutting down...")
    _cleanup()

# ---------------- Schemas ----------------
class InputQ(BaseModel):
    qid: str
    stem: str
    options: List[str] = []

class RouteRequest(BaseModel):
    inputs: List[InputQ]

# ---------------- Endpoints ----------------
@app.get("/healthz")
def healthz():
    device = "n/a"
    if _model is not None and hasattr(_model, "device"):
        device = str(_model.device)
    return {
        "ok": True,
        "model": MODEL_ID,
        "loaded": _model is not None,
        "device": device,
    }

@app.post("/route")
def route(req: RouteRequest):
    """
    Classify questions into topic/subtopic/difficulty.
    NOTE: Keeps your previous light parsing; can be upgraded to strict JSON if desired.
    """
    routes = []
    try:
        if len(req.inputs) > 0:
            q0 = req.inputs[0]
            prompt = (
                "Analyze this math question and provide:\n"
                "Topic: [topic]\n"
                "Subtopic: [subtopic]\n"
                "Difficulty: [E for easy, M for medium, H for hard]\n\n"
                f"Question: {q0.stem}\n"
            )
            response = _generate(prompt)
            logger.info(f"Model response: {response}")

            topic = "algebra"
            subtopic = None
            difficulty = "M"

            lower = response.lower()
            if "topic:" in lower:
                try:
                    topic = lower.split("topic:")[1].split("\n")[0].strip()
                except Exception:
                    pass
            if "subtopic:" in lower:
                try:
                    subtopic = lower.split("subtopic:")[1].split("\n")[0].strip()
                except Exception:
                    pass
            if "difficulty:" in lower:
                try:
                    diff = lower.split("difficulty:")[1].split("\n")[0].strip()
                    if diff and diff[0] in ("e", "m", "h"):
                        difficulty = diff[0].upper()
                except Exception:
                    pass

            for q in req.inputs:
                routes.append({
                    "qid": q.qid,
                    "topic": topic,
                    "difficulty": difficulty,
                    "subtopic": subtopic,
                    "needs_tools": [],
                    "confidence": 0.7,
                    "notes": None
                })
    except Exception as e:
        logger.error(f"Error processing route: {e}")
        for q in req.inputs:
            routes.append({
                "qid": q.qid,
                "topic": "algebra",
                "difficulty": "M",
                "subtopic": None,
                "needs_tools": [],
                "confidence": 0.5,
                "notes": f"Error: {str(e)}"
            })
    return routes

@app.post("/ask")
def ask(question: Dict[str, Any]):
    try:
        content = question.get("question", "What is the capital of France?")
        response = _generate(content)
        return {
            "answer": response,
            "model": MODEL_ID,
            "mode": "transformers"
        }
    except Exception as e:
        logger.error(f"Error in /ask: {e}")
        return {"error": str(e)}
