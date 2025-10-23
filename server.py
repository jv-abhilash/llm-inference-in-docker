from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os, torch, httpx, json, re
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Router Server")

MODEL_ID = os.getenv("ROUTER_MODEL", "Qwen/Qwen2.5-Math-7B-Instruct")
HF_HOME = os.getenv("HF_HOME", "/models/hf-cache")
HF_TOKEN = os.getenv("HF_TOKEN", None)
USE_VLLM = os.getenv("USE_VLLM", "false").lower() == "true"
VLLM_PORT = int(os.getenv("VLLM_PORT", "8000"))
VLLM_HOST = os.getenv("VLLM_HOST", "localhost")

_tok = None
_model = None

@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("LLM Router Server Starting")
    logger.info("=" * 60)
    logger.info(f"Model: {MODEL_ID}")
    logger.info(f"Mode: {'vLLM' if USE_VLLM else 'Transformers (4-bit quantized)'}")

    if not USE_VLLM:
        logger.info("Model will be loaded on first request (lazy loading)")
        logger.info("This saves memory and allows server to start quickly")
    else:
        logger.info(f"vLLM mode enabled - expecting vLLM server at {VLLM_HOST}:{VLLM_PORT}")
        logger.info("Note: You must start vLLM separately!")

    logger.info("=" * 60)

def _load_direct():
    """Load model directly with transformers and 4-bit quantization"""
    global _tok, _model

    if _tok is not None and _model is not None:
        logger.info("Model already loaded, skipping...")
        return

    logger.info("=" * 60)
    logger.info(f"Loading model {MODEL_ID} with 4-bit quantization...")
    logger.info("This may take 2-5 minutes on first load...")
    logger.info("=" * 60)

    # Check available GPU memory
    if torch.cuda.is_available():
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"Total GPU memory: {gpu_mem:.2f} GB")
    else:
        logger.warning("No GPU detected, using CPU only (will be slow!)")

    # Setup 4-bit quantization config (only if CUDA available)
    quantization_config = None
    device_map = "cpu"
    torch_dtype = torch.float32

    if torch.cuda.is_available():
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            device_map = "cuda:0"
            torch_dtype = torch.float16
            logger.info("✓ 4-bit quantization config created for CUDA")
        except Exception as e:
            logger.warning(f"bitsandbytes unavailable, falling back to full precision on GPU: {e}")
            device_map = "cuda:0"
            torch_dtype = torch.float16

    try:
        logger.info("Step 1/2: Loading tokenizer...")
        _tok = AutoTokenizer.from_pretrained(
            MODEL_ID,
            token=HF_TOKEN,
            trust_remote_code=True
        )
        logger.info("✓ Tokenizer loaded successfully")

        logger.info("Step 2/2: Loading model with 4-bit quantization...")
        logger.info("(This is the slow part - please wait...)")

        # Load model with proper configuration
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            token=HF_TOKEN,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True
        )

        logger.info("=" * 60)
        logger.info(f"✓ Model {MODEL_ID} loaded successfully!")

        # Verify model is on GPU
        model_device = str(_model.device) if hasattr(_model, 'device') else "unknown"
        if hasattr(_model, 'hf_device_map'):
            logger.info(f"Device map: {_model.hf_device_map}")
        logger.info(f"Model device: {model_device}")

        # Get memory usage after loading
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            logger.info(f"GPU memory allocated: {allocated:.2f} GB")
            logger.info(f"GPU memory reserved: {reserved:.2f} GB")

            if allocated < 0.5:
                logger.warning("⚠️ WARNING: GPU memory usage < 0.5 GB - model may not be on GPU!")
                logger.warning("⚠️ Expected 3-4 GB for Qwen2.5-Math-7B with 4-bit quantization")
            else:
                logger.info(f"✓ Model successfully loaded to GPU ({allocated:.2f} GB)")

        logger.info("=" * 60)

    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"✗ Error loading model: {str(e)}")
        logger.error("=" * 60)
        import traceback
        logger.error(traceback.format_exc())
        raise

async def _generate_with_vllm(prompt):
    """Generate text using vLLM API"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"http://{VLLM_HOST}:{VLLM_PORT}/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200,
                "temperature": 0.0
            },
            timeout=30.0
        )

        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            logger.error(f"vLLM API error: {response.text}")
            raise Exception(f"vLLM API error: {response.status_code}")

def _generate_direct(prompt, max_new_tokens=200):
    """Generate text directly using the loaded model"""
    if _tok is None or _model is None:
        raise RuntimeError("Model not loaded! Call _load_direct() first")

    inputs = _tok(prompt, return_tensors="pt").to(_model.device)

    # Get pad token id
    pad_token_id = _tok.eos_token_id if _tok.pad_token_id is None else _tok.pad_token_id

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,  # Low but not zero to allow some variation
            do_sample=True,   # Enable sampling
            top_p=0.9,        # Nucleus sampling
            pad_token_id=pad_token_id,
            eos_token_id=_tok.eos_token_id
        )
    
    # Decode and return only the NEW tokens (not the input)
    full_output = _tok.decode(outputs[0], skip_special_tokens=True)
    # Remove the input prompt from the output
    if full_output.startswith(prompt):
        return full_output[len(prompt):].strip()
    return full_output

def _cleanup():
    """Clean GPU memory release"""
    global _tok, _model
    if _model is not None:
        logger.info("Offloading model from GPU...")
        _model = _model.cpu()
        del _model
        torch.cuda.empty_cache()
        _model = None
        _tok = None
        logger.info("Model offloaded successfully")

class InputQ(BaseModel):
    qid: str
    stem: str
    options: List[str] = []

class RouteRequest(BaseModel):
    inputs: List[InputQ]

# ---------------- LLM-First Classification ----------------
def _classify_with_llm(stem: str, options: List[str]) -> Optional[Dict[str, Any]]:
    """
    Use LLM to classify the question. Returns None if fails.
    Uses a simple instruction-following format.
    """
    # CRITICAL FIX: Ensure model is loaded before using it
    if _model is None:
        logger.info("Model not loaded yet, loading now for classification...")
        _load_direct()

    # Simple, direct prompt - no chat format
    prompt = f"""Task: Classify this math question into ONE category and return a JSON object.

Question: {stem}

Categories:
1. algebra - matrices, vectors, complex numbers, equations, inequalities, sequences, series, binomial theorem
2. calculus - derivatives, integrals, limits, continuity, differential equations
3. discrete - probability, permutations, combinations, statistics, sets, graph theory
4. geometry - circles, triangles, coordinate geometry, conic sections, lines, polygons

Instructions: Output ONLY a valid JSON object with these exact fields. No other text.
- topic: one of [algebra, calculus, discrete, geometry]
- subtopic: specific area within the topic
- difficulty: one of [E, M, H] for easy, medium, hard
- confidence: number between 0.0 and 1.0

Output (JSON only):
{{"""

    try:
        # Generate response - the {{ will help the model start generating JSON
        response = _generate_direct(prompt, max_new_tokens=150)
        
        logger.info(f"Raw LLM output: {response[:300]}")

        # The response should start with the JSON completion
        # Add back the opening brace and try to find complete JSON
        json_candidate = "{" + response
        
        # Find the first complete JSON object
        brace_count = 0
        end_idx = -1
        for i, char in enumerate(json_candidate):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        
        if end_idx > 0:
            json_str = json_candidate[:end_idx]
            logger.info(f"Extracted JSON: {json_str}")
            
            try:
                result = json.loads(json_str)
                
                # Handle keys with spaces (normalize all keys by stripping whitespace)
                normalized_result = {}
                for key, value in result.items():
                    clean_key = key.strip().replace(' ', '')  # Remove all spaces
                    normalized_result[clean_key] = value
                
                # Validate and return
                if 'topic' in normalized_result:
                    topic = normalized_result.get('topic', 'algebra').lower().strip()
                    subtopic = normalized_result.get('subtopic', 'general').lower().strip()
                    difficulty = str(normalized_result.get('difficulty', 'M')).upper().strip()
                    if len(difficulty) > 1:
                        difficulty = difficulty[0]  # Take first char if multi-char
                    confidence = float(normalized_result.get('confidence', 0.7))
                    
                    logger.info(f"✓ Parsed: {topic}/{subtopic} [{difficulty}] ({confidence:.2f})")
                    
                    return {
                        'topic': topic,
                        'subtopic': subtopic,
                        'difficulty': difficulty,
                        'confidence': min(max(confidence, 0.0), 1.0)  # Clamp between 0-1
                    }
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e} | String was: {json_str}")
                return None
        
        logger.warning("No valid JSON found in LLM response")
        return None

    except Exception as e:
        logger.error(f"LLM classification error: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

def _classify_with_keywords(stem: str, options: List[str]) -> Dict[str, Any]:
    """
    Keyword-based fallback classification.
    Returns topic, subtopic, difficulty, and confidence.
    """
    stem_lower = stem.lower()

    # CALCULUS - highest priority keywords
    if any(kw in stem_lower for kw in ['∫', 'integral', 'integrate']):
        return {'topic': 'calculus', 'subtopic': 'integration', 'difficulty': 'M', 'confidence': 0.95}
    if any(kw in stem_lower for kw in ['derivative', 'dy/dx', 'd/dx', 'differentiat']):
        return {'topic': 'calculus', 'subtopic': 'differentiation', 'difficulty': 'M', 'confidence': 0.95}
    if any(kw in stem_lower for kw in ['lim', 'limit']):
        return {'topic': 'calculus', 'subtopic': 'limits', 'difficulty': 'M', 'confidence': 0.90}

    # DISCRETE - combinatorics and probability
    if any(kw in stem_lower for kw in ['permutation', 'arrangement', 'arrange']):
        return {'topic': 'discrete', 'subtopic': 'permutations', 'difficulty': 'M', 'confidence': 0.95}
    if any(kw in stem_lower for kw in ['combination', 'choose', 'selection']):
        return {'topic': 'discrete', 'subtopic': 'combinations', 'difficulty': 'M', 'confidence': 0.95}
    if any(kw in stem_lower for kw in ['probability', 'dice', 'coin', 'card']):
        return {'topic': 'discrete', 'subtopic': 'probability', 'difficulty': 'M', 'confidence': 0.90}
    if any(kw in stem_lower for kw in ['mean', 'median', 'variance', 'standard deviation']):
        return {'topic': 'discrete', 'subtopic': 'statistics', 'difficulty': 'M', 'confidence': 0.90}
    if any(kw in stem_lower for kw in ['set', 'union', 'intersection']):
        return {'topic': 'discrete', 'subtopic': 'sets', 'difficulty': 'M', 'confidence': 0.85}

    # GEOMETRY - shapes and coordinate geometry
    if any(kw in stem_lower for kw in ['circle', 'radius', 'diameter']):
        return {'topic': 'geometry', 'subtopic': 'circles', 'difficulty': 'M', 'confidence': 0.95}
    if any(kw in stem_lower for kw in ['parabola', 'ellipse', 'hyperbola', 'latus rectum']):
        return {'topic': 'geometry', 'subtopic': 'conic sections', 'difficulty': 'M', 'confidence': 0.95}
    if any(kw in stem_lower for kw in ['triangle', 'polygon', 'octagon', 'diagonal']):
        return {'topic': 'geometry', 'subtopic': 'polygons', 'difficulty': 'M', 'confidence': 0.90}
    if any(kw in stem_lower for kw in ['line', 'slope', 'perpendicular', 'intercept']):
        return {'topic': 'geometry', 'subtopic': 'straight lines', 'difficulty': 'M', 'confidence': 0.85}

    # ALGEBRA - default category with specific keywords
    if any(kw in stem_lower for kw in ['matrix', 'matrices', 'determinant', 'adjoint']):
        return {'topic': 'algebra', 'subtopic': 'matrices', 'difficulty': 'M', 'confidence': 0.95}
    if any(kw in stem_lower for kw in ['vector', 'dot product', 'cross product']):
        return {'topic': 'algebra', 'subtopic': 'vectors', 'difficulty': 'M', 'confidence': 0.90}
    if any(kw in stem_lower for kw in ['complex', 'imaginary', 'modulus']):
        return {'topic': 'algebra', 'subtopic': 'complex numbers', 'difficulty': 'M', 'confidence': 0.90}
    if any(kw in stem_lower for kw in ['g.p.', 'a.p.', 'geometric progression', 'arithmetic progression']):
        return {'topic': 'algebra', 'subtopic': 'sequences and series', 'difficulty': 'M', 'confidence': 0.90}
    if any(kw in stem_lower for kw in ['inequality', 'inequalities']):
        return {'topic': 'algebra', 'subtopic': 'inequalities', 'difficulty': 'M', 'confidence': 0.85}
    if any(kw in stem_lower for kw in ['binomial', 'expansion']):
        return {'topic': 'algebra', 'subtopic': 'binomial theorem', 'difficulty': 'M', 'confidence': 0.90}

    # Default fallback
    return {'topic': 'algebra', 'subtopic': 'general', 'difficulty': 'M', 'confidence': 0.60}

def _classify_question(stem: str, options: List[str]) -> Dict[str, Any]:
    """
    Main classification function: LLM FIRST, then keyword fallback.
    """
    # Step 1: Try LLM first (if not using vLLM)
    if not USE_VLLM:
        llm_result = _classify_with_llm(stem, options)
        if llm_result:
            logger.info(f"✓ LLM: {llm_result['topic']} - {llm_result['subtopic']} (conf: {llm_result['confidence']:.2f})")
            return llm_result
        else:
            logger.info("LLM classification failed, falling back to keywords")

    # Step 2: Fallback to keywords
    keyword_result = _classify_with_keywords(stem, options)
    logger.info(f"✓ Keyword: {keyword_result['topic']} - {keyword_result['subtopic']} (conf: {keyword_result['confidence']:.2f})")
    return keyword_result

@app.get("/healthz")
async def healthz():
    if USE_VLLM:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"http://{VLLM_HOST}:{VLLM_PORT}/health", timeout=2.0)
                vllm_ok = response.status_code == 200
        except Exception:
            vllm_ok = False

        return {
            "ok": True,
            "model": MODEL_ID,
            "loaded": vllm_ok,
            "mode": "vllm",
            "device": "cuda:0" if vllm_ok else "unknown"
        }
    else:
        device = "unknown"
        if _model is not None:
            device = str(_model.device) if hasattr(_model, 'device') else "cuda:0"

        return {
            "ok": True,
            "model": MODEL_ID,
            "loaded": _model is not None,
            "mode": "transformers",
            "device": device
        }

@app.post("/route")
async def route(req: RouteRequest):
    """
    Classify questions into topic/subtopic/difficulty.
    LLM-first approach with keyword fallback.
    """
    routes = []

    try:
        for q in req.inputs:
            try:
                classification = _classify_question(q.stem, q.options)

                routes.append({
                    "qid": q.qid,
                    "topic": classification['topic'],
                    "difficulty": classification['difficulty'],
                    "subtopic": classification['subtopic'],
                    "needs_tools": [],
                    "confidence": classification['confidence'],
                    "notes": None
                })

                logger.info(f"→ {q.qid}: {classification['topic']}/{classification['subtopic']} [{classification['difficulty']}] ({classification['confidence']:.2f})")

            except Exception as e:
                logger.error(f"Error classifying {q.qid}: {e}")
                routes.append({
                    "qid": q.qid,
                    "topic": "algebra",
                    "difficulty": "M",
                    "subtopic": "error",
                    "needs_tools": [],
                    "confidence": 0.5,
                    "notes": f"Error: {str(e)}"
                })

    except Exception as e:
        logger.error(f"Error processing route request: {e}")
        for q in req.inputs:
            routes.append({
                "qid": q.qid,
                "topic": "algebra",
                "difficulty": "M",
                "subtopic": "error",
                "needs_tools": [],
                "confidence": 0.5,
                "notes": f"Error: {str(e)}"
            })

    return routes

@app.post("/ask")
async def ask(question: Dict[str, Any]):
    try:
        content = question.get("question", "What is the capital of France?")

        if USE_VLLM:
            try:
                response = await _generate_with_vllm(content)
            except Exception as e:
                return {"error": f"vLLM error: {str(e)}"}
        else:
            # Lazy load model
            if _model is None:
                _load_direct()

            response = _generate_direct(content, max_new_tokens=200)

        return {
            "answer": response,
            "model": MODEL_ID,
            "mode": "vllm" if USE_VLLM else "transformers"
        }

    except Exception as e:
        logger.error(f"Error in /ask: {str(e)}")
        return {"error": str(e)}

@app.on_event("shutdown")
def shutdown_event():
    logger.info("Shutting down MCP router service")
    if not USE_VLLM:
        _cleanup()