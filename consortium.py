import requests
import json
import ast
import os
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, as_completed

router = APIRouter()

UPSTREAM_URL = "https://uiuc.chat/api/chat-api/chat"
TIMEOUT = 30


DEFAULT_MODELS = ["gemma3:27b","Qwen/Qwen2.5-VL-72B-Instruct","llama4:16x17b"]  # gpt-5-mini, Qwen/Qwen2.5-VL-72B-Instruct, llama4:16x17b
DEFAULT_API_KEY = "uc_3fc20373173944c09d0ee8a0b62af79c"
DEFAULT_COURSE_NAME = "cropwizard-1.5"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_RETRIEVAL_ONLY = False


class ConsortiumRequest(BaseModel):
    user_question: str
    temperature: float = DEFAULT_TEMPERATURE
    course_name: str = DEFAULT_COURSE_NAME
    retrieval_only: bool = DEFAULT_RETRIEVAL_ONLY


class ModelResponse(BaseModel):
    model: str
    response: Any
    is_best: bool = False
    score: float = 0.0 


class ConsortiumResponse(BaseModel):
    results: Dict[str, ModelResponse]


def call_llm(model: str, messages: List[Dict[str, str]], timeout: int = 40) -> str:
    """Fixed: No stream, short timeout, return string."""
    payload = {
        "model": model,
        "messages": messages,
        "api_key": DEFAULT_API_KEY,
        "temperature": DEFAULT_TEMPERATURE,
        "course_name": DEFAULT_COURSE_NAME,
        "stream": False 
    }
    
    try:
        resp = requests.post(UPSTREAM_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("response", resp.text)[:2000]  # Truncate long responses
    except Exception as e:
        return f"Error {model}: {str(e)}"

def summarize_diagnosis(raw_json: str) -> str:
    """Fast 5-line summary."""
    messages = [
        {"role": "system", "content": """Summarize crop diagnosis in EXACTLY 5 bullets:
• Location: lat/long
• Weather: temp/humidity  
• Disease: common (scientific)
• Severity/Confidence: stage/medium
• Impact: weather effect"""},
        {"role": "user", "content": raw_json[:4000]} 
    ]
    return call_llm("llama4:16x17b", messages, timeout=30)

def build_standard_messages(user_question: str, diagnosis_summary: str = None) -> List[Dict[str, str]]:
    """
    Build a rich, expert-level prompt for all LLMs, combining:
    - A concise diagnosis summary
    - A clear, expert-framed farmer question
    """

    if not diagnosis_summary:
        diagnosis_summary = (
            "**DIAGNOSIS SUMMARY**\n"
            "- No structured diagnosis data was provided.\n"
            "- Ask clarifying questions about crop, field conditions, and symptoms before recommending actions."
        )

    # Infer crop type to set context tone
    crop_type = "row crops"
    for crop in ["cotton", "corn", "soybean", "wheat"]:
        if crop in diagnosis_summary.lower():
            crop_type = crop
            break

    system_prompt = f"""You are a senior plant pathologist and agronomy consultant for Illinois {crop_type} farmers.
You specialize in integrating weather, field observations, and diagnosis outputs into clear, practical recommendations."""

    user_content = f"""Here is the current DIAGNOSIS SUMMARY for this field:

{diagnosis_summary}

Here is the FARMER QUESTION:

{user_question}

As an expert consultant, respond with EXACTLY these sections in order:

1. **Farmer-friendly explanation**
   - 2–3 sentences, plain language.
   - Reference key facts from the diagnosis (crop, disease, severity, weather, confidence).

2. **Immediate actions (next 7 days)**
   - 3 bullet points, highest priority first.
   - Include field operations to do or avoid (e.g., scouting focus, sanitation, traffic when foliage is wet, irrigation changes).

3. **Treatment plan**
   - 2–3 bullets.
   - Name the type of product (e.g., copper-based bactericide, fungicide class), target timing (before/after rain, temperature/humidity conditions), and any cautions.
   - If chemical intervention is not recommended, say so and explain why.

4. **Prevention for next season**
   - 3–4 bullets.
   - Include seed choice (resistant/tolerant varieties), rotation, residue management, and canopy/irrigation practices.

5. **Risk forecast (next 7 days)**
   - 2–3 bullets.
   - Use the described weather (temperature, humidity, leaf wetness, wind, precipitation) to rate risk as low / moderate / high.
   - Explain what conditions would make risk increase, and what the farmer should watch for when scouting.

Constraints:
- Be concise but specific (numbers or ranges when useful, like temperature ranges or days to re-scout).
- Base your advice strictly on the diagnosis summary and weather context; do not invent lab results.
- Assume the farm is in Illinois and follow region-appropriate, broadly safe best practices.
- Answer as a human expert consultant, not as an AI model."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def select_best_response(
    results: Dict[str, ModelResponse],
    diagnosis_summary: str
) -> ModelResponse:
    """Enhanced: tiebreakers + expose score."""
    disease_terms = []
    crop_terms = []
    text_summary = diagnosis_summary.lower() if diagnosis_summary else ""

    # Extract keywords
    for crop in ["cotton", "corn", "soy", "wheat"]:
        if crop in text_summary:
            crop_terms.append(crop)
    for disease in ["angular leaf spot", "bacterial blight"]:
        if disease in text_summary:
            disease_terms.append(disease)

    required_sections = [
        "explanation", "immediate actions", "treatment",
        "prevention", "risk forecast"
    ]

    best_model = None
    best_score = -1
    best_response = None

    for model_name, mr in results.items():
        raw = str(mr.response)
        text = raw.lower()

        # Skip failures
        if any(err in text for err in ["httpsconnectionpool", "bad gateway"]):
            continue

        score = 0

        # Sections: 2 pts each
        section_hits = sum(1 for sec in required_sections if sec in text)
        score += section_hits * 2

        # Keywords: disease 3 pts, crop 2 pts
        score += sum(3 for term in disease_terms if term in text)
        score += sum(2 for term in crop_terms if term in text)

        # Length: +2 pts
        length = len(text)
        if 500 <= length <= 4000:
            score += 2

        # Tiebreakers (+1 pt each)
        if "illinois" in text or "row crop" in text:
            score += 1
        if "**" in raw or "###" in raw:  # Markdown structure
            score += 1
        if model_name == "gpt-oss:120b":  # Model preference
            score += 0.5

        # Track best + expose score
        if score > best_score:
            best_score = score
            best_model = model_name
            best_response = mr
            best_response.score = best_score  # Add attr for debug

    return best_response if best_response else list(results.values())[0]


@router.post("/llm_consortium", response_model=ConsortiumResponse)
def llm_consortium(payload: ConsortiumRequest) -> ConsortiumResponse:
    debug = {}

    # Input handling [unchanged]
    json_path = "sample.json"
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            raw_data = json.load(f)
        debug["source"] = "sample.json"
        debug["session_id"] = raw_data.get("session_id", "unknown")
        debug["image_url"] = raw_data.get("image_url", "none")
        field_conditions = raw_data.get("field_conditions", {})
        crop_diagnosis = raw_data.get("crop_diagnosis", {})
        flat_data = {**field_conditions, **crop_diagnosis}
        raw_input = json.dumps(flat_data)
    else:
        raw_input = payload.user_question.strip()
        debug["source"] = "request_body"

    is_raw_json = len(raw_input) > 200 and any(kw in raw_input.lower() for kw in
        ['latitude', 'diagnosis', 'disease_name', 'weather_context'])

    if is_raw_json:
        debug["raw_detected"] = "yes"
        diagnosis_summary = summarize_diagnosis(raw_input)
        print(diagnosis_summary)
        debug["summary"] = diagnosis_summary[:200]
        clean_question = "As an expert consultant, what should the farmer do NOW based on this diagnosis?"
    else:
        diagnosis_summary = None
        clean_question = payload.user_question
        debug["raw_detected"] = "no"

    messages = build_standard_messages(clean_question, diagnosis_summary)
    debug["prompt_preview"] = messages[0]["content"][:300]

    # Parallel calls
    results = {}
    with ThreadPoolExecutor(max_workers=len(DEFAULT_MODELS)) as executor:
        futures = {executor.submit(call_llm, model, messages): model
                  for model in DEFAULT_MODELS}
        for future in as_completed(futures):
            model = futures[future]
            results[model] = ModelResponse(model=model, response=future.result())

    best = select_best_response(results, diagnosis_summary)

    # Final summary from BEST model (SINGLE try)
    if best and "error" not in str(best.response).lower():
        final_messages = [
            {"role": "system", "content": """Crop expert: Summarize YOUR detailed advice above in 2 farmer-friendly paras (150 words):
            Para 1: Diagnosis + weather explanation. 
            Para 2: 3 NOW actions + prevention. Plain language, bullets OK."""},
            {"role": "user", "content": f"DIAGNOSIS: {diagnosis_summary}\n\nYOUR ADVICE: {best.response}"}
        ]
        try:
            final_summary = call_llm(best.model, final_messages, timeout=40)
        except Exception:
            final_summary = "Summary unavailable—use detailed advice."
        
        # Update results[best] directly
        results[best.model].response = {
            "detailed": best.response,
            "final_summary": final_summary
        }
        results[best.model].is_best = True
        results[best.model].score = getattr(best, 'score', 0)
        debug["final_summary_model"] = best.model
        debug["final_summary_length"] = len(final_summary)
    else:
        debug["final_summary_skipped"] = "Best errored"

    debug["best_model"] = best.model if best else "none"
    debug["best_score"] = getattr(best, 'score', 'N/A') if best else "N/A"

    # SINGLE best result (your request)
    if best and "final_summary" in results[best.model].response:
        return ConsortiumResponse(
            results={best.model: results[best.model]}, 
            debug=debug
        )
    
    # Fallback: all results
    highlighted_results = {m: mr.dict() for m, mr in results.items()}
    return ConsortiumResponse(results=highlighted_results, debug=debug)
