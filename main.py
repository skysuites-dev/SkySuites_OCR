from fastapi import APIRouter, UploadFile, File, Query, FastAPI
from fastapi.responses import JSONResponse
from tempfile import SpooledTemporaryFile
from datetime import datetime
import os
import json
import yaml
import textwrap
from vertexai.preview.generative_models import GenerativeModel
from vertexai import init
from google.cloud import vision
import io
from functools import lru_cache

# ✅ Environment setup
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"skysuitesbookingapp-7f085-cdd2ab2069b4.json"

# ✅ Initialize app
app = FastAPI(title="OCR Scanner API")
app.include_router(APIRouter())

# ✅ Vertex AI setup
init(project="skysuitesbookingapp-7f085", location="us-east1")
MODEL_NAME = "gemini-2.5-flash-lite"
model = GenerativeModel(MODEL_NAME)

# ✅ OCR Processing
def extract_text(image_path: str) -> str:
    client = vision.ImageAnnotatorClient()
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    if response.error.message:
        raise Exception(f'{response.error.message}')
    texts = response.text_annotations
    return texts[0].description if texts else ""

# ✅ Airline Policy Cache
@lru_cache(maxsize=50)
def load_policy_text(airline: str, policy_dir: str = r"policy_docs") -> str:
    path = os.path.join(policy_dir, f"{airline.lower().strip()}.yaml")
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# ✅ Prompt Builder (unchanged)
def build_prompt(doc_type: str, ocr_text: str, policy_text: str) -> str:
    doc_type = doc_type.lower()
    policy_section = f"\n--- AIRLINE POLICY RULES ---\n{textwrap.dedent(policy_text)}\n--- END POLICY ---\n" if policy_text else ""

    if doc_type == "cnic":
        return f"""
You are a document data extraction AI designed for airline passenger autofill. Extract only English values from a CNIC (national ID) OCR.

Follow airline policy rules if provided.

Rules:
- Extract `full_name` by combining surname/last_name too if present separately, and split into:
    - `first_name`: Must exactly match the Given Name Field if present
    - `middle_name` if Middle Name present
    - `last_name` from Surname or use father's name if Surname missing
- Use 'M', 'F', or 'O' for `gender`
- Salutation:
  - 'Mr' for male
  - 'Ms' if unmarried female
- Use `husband_name` only if 'wife of' is present
- Otherwise use `father_name`
- CNIC number format: As given in OCR Text
- DOB in YYYY-MM-DD format
- Extract nationality if present in English
- Check and Include CNIC expiry

--- OCR TEXT ---
{textwrap.dedent(ocr_text)}
{policy_section}
Return clean JSON (leave unknown fields as empty strings):

{{
  "full_name": "",
  "first_name": "",
  "middle_name": "",
  "last_name": "",
  "identity_number": "",
  "dob": "",
  "salutation": "",
  "expiry_date": ""
}}
"""
    elif doc_type == "passport":
        return f"""
You are a document data extraction AI designed for airline booking autofill. Extract only English values from a passport OCR.

Rules:
- Extract `full_name` by combining surname/last_name too if present separately and split into:
    - `first_name`: Must match the full Given Name exactly as it appears (even if it has multiple words)
    - `middle_name` if Middle Name present
    - `last_name` from Surname or use father's name if Surname missing
- Use 'M', 'F', or 'O' for `gender`
- Salutation:
  - 'Mr' for male
  - 'Ms' if unmarried female
- Use `husband_name` only if 'wife of' is present
- Otherwise use `father_name`
- Passport number: Alphanumeric (usually 8–10 characters)
- Passport expiry: YYYY-MM-DD
- Passport country: Extract as full country or ISO 3-letter code and convert it into alpha 2-code
- DOB in YYYY-MM-DD format
- Extract nationality if mentioned (not just issuing country)
- Check and Include Passport expiry

--- OCR TEXT ---
{textwrap.dedent(ocr_text)}
{policy_section}
Return clean JSON (leave unknown fields as empty strings):

{{
  "full_name": "",
  "first_name": "",
  "middle_name": "",
  "last_name": "",
  "passport_number": "",
  "passport_country": "",
  "passport_expiry": "",
  "dob": "",
  "salutation": "",
}}
"""
    else:
        return "Unsupported document type."

# ✅ Field Extractor
def extract_fields_from_text(ocr_text: str, doc_type: str, airline: str) -> dict:
    policy_text = load_policy_text(airline)
    prompt = build_prompt(doc_type, ocr_text, policy_text)

    try:
        response = model.generate_content(prompt)
        raw_text = response.text

        # 🔍 Optimized JSON extraction
        try:
            json_str = raw_text[raw_text.index('{'): raw_text.rindex('}') + 1]
            structured_data = json.loads(json_str)
        except (ValueError, json.JSONDecodeError):
            return {"error": "Failed to extract structured JSON from Gemini output."}

        return {
            "structured_data": structured_data,
            "policy_used": policy_text,
            "gemini_output_raw": raw_text,
            "prompt": prompt
        }

    except Exception as e:
        return {"error": f"Gemini Vertex AI error: {str(e)}"}

# ✅ Expiry Checker
def is_expired(date_str: str) -> bool:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d") < datetime.today()
    except Exception:
        return False

# ✅ Age/Passenger Type Check
def calculate_age(dob_str: str) -> int:
    try:
        dob = datetime.strptime(dob_str, "%Y-%m-%d")
        today = datetime.today()
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    except Exception:
        return None

def validate_passenger_type(dob: str, passenger_type: str):
    age = calculate_age(dob)
    pt = passenger_type.lower()

    if age is None:
        return JSONResponse(status_code=422, content={"status": 422, "message": "Please rescan your document again!"})

    if pt == "infant" and not (0 <= age < 2):
        return JSONResponse(status_code=400, content={"status": 400, "message": "Invalid passenger type: Not an infant."})
    elif pt == "child" and not (2 <= age < 12):
        return JSONResponse(status_code=400, content={"status": 400, "message": "Invalid passenger type: Not a child."})
    elif pt == "adult" and not (age >= 12):
        return JSONResponse(status_code=400, content={"status": 400, "message": "Invalid passenger type: Not an adult."})
    return None

# ✅ Main Scan Route
@app.post("/scan")
async def scan_document(
    file: UploadFile = File(...),
    doc_type: str = Query(...),
    passenger_type: str = Query(...),
    airline: str = Query(...)
):
    try:
        with SpooledTemporaryFile(suffix=".jpg") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp.seek(0)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as disk_tmp:
                disk_tmp.write(tmp.read())
                disk_tmp_path = disk_tmp.name

        # OCR
        ocr_text = extract_text(disk_tmp_path)
        print("🔍 OCR TEXT:", ocr_text)

        # Gemini
        extracted_data = extract_fields_from_text(ocr_text, doc_type, airline)
        if "error" in extracted_data:
            return JSONResponse(status_code=500, content={"status": 500, "message": extracted_data["error"]})

        structured = extracted_data.get("structured_data", {})
        dob = structured.get("dob")
        if not dob:
            return JSONResponse(status_code=422, content={"status": 422, "message": "DOB missing. Please scan again!"})

        validation_error = validate_passenger_type(dob, passenger_type)
        if validation_error:
            return validation_error

        expiry_keys = ["doe", "passport_expiry", "expiry_date", "cnic_expiry"]
        expiry = next((structured.get(k) for k in expiry_keys if structured.get(k)), "")

        if expiry and is_expired(expiry):
            return JSONResponse(status_code=400, content={"status": 400, "message": f"{doc_type.upper()} is expired!"})

        return {"status": "success", "corrected_json": structured}

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": 500, "message": f"Unexpected error: {str(e)}"})

