from fastapi import APIRouter, UploadFile, File, Query,FastAPI
from fastapi.responses import JSONResponse
import tempfile
from datetime import datetime
import os
import json
import yaml
import textwrap
from vertexai.preview.generative_models import GenerativeModel
from vertexai import init
from google.cloud import vision
import io

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"skysuitesbookingapp-7f085-cdd2ab2069b4.json"

def preprocess_image(image_path: str) -> str:
    return image_path

def extract_text(image_path: str) -> str:
    client = vision.ImageAnnotatorClient()

    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    if response.error.message:
        raise Exception(f'{response.error.message}')

    return texts[0].description if texts else ""

def is_expired(date_str: str) -> bool:
    try:
        expiry_date = datetime.strptime(date_str, "%Y-%m-%d")
        return expiry_date < datetime.today()
    except Exception as e:
        print(f"❗ Failed to parse expiry date '{date_str}':", e)
        return False


def extract_cnic_fields(text: str) -> dict:
    return extract_fields_from_text(text, doc_type="cnic")

def parse_passport_data(text: str) -> dict:
    return extract_fields_from_text(text, doc_type="passport")


app = FastAPI(title="OCR Scanner API")
app.include_router(APIRouter())


# ✅ Initialize Vertex AI
init(project="skysuitesbookingapp-7f085", location="us-east1")

# ✅ Load Gemini model
MODEL_NAME = "gemini-2.5-flash-lite"  # or your preferred model
model = GenerativeModel(MODEL_NAME)

def load_policy_text(airline: str, policy_dir: str = r"C:\Users\PMLS\Music\ocr_project\policy_docs") -> str:
    path = os.path.join(policy_dir, f"{airline.lower().strip()}.yaml")
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def extract_fields_from_text(ocr_text: str, doc_type: str, airline: str) -> dict:
    policy_text = load_policy_text(airline)
    prompt = build_prompt(doc_type, ocr_text, policy_text)

    try:
        response = model.generate_content(prompt)

        raw_text = response.text

        json_start = raw_text.find('{')
        json_end = raw_text.rfind('}') + 1
        json_str = raw_text[json_start:json_end]
        structured_data = json.loads(json_str)

        return {
            "structured_data": structured_data,
            "policy_used": policy_text,
            "gemini_output_raw": raw_text,
            "prompt": prompt
        }

    except Exception as e:
        return {"error": f"Gemini Vertex AI error: {str(e)}"}

def build_prompt(doc_type: str, ocr_text: str, policy_text: str) -> str:
    doc_type = doc_type.lower()
    policy_section = f"\n--- AIRLINE POLICY RULES ---\n{textwrap.dedent(policy_text)}\n--- END POLICY ---\n" if policy_text else ""

    if doc_type == "cnic":
        return f"""
You are a document data extraction AI designed for airline passenger autofill. Extract only English values from a CNIC (national ID) OCR.

Follow airline policy rules if provided.

Rules:
- Extract `full_name`, and split into `first_name`, `middle_name`, `last_name`
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
- Extract `full_name` and split into `first_name`, `middle_name`, `last_name`
- Use 'M', 'F', or 'O' for `gender`
- Salutation:
  - 'Mr' for male
  - 'Ms' if unmarried female
- Use `husband_name` only if 'wife of' is present
- Otherwise use `father_name`
- Passport number: Alphanumeric (usually 8–10 characters)
- Passport expiry: YYYY-MM-DD
- Passport country: Extract as full country or ISO 3-letter code
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
  "expiry_date": ""
}}
"""
    else:
        return "Unsupported document type."


# Utility: Age calculation
def calculate_age(dob_str: str) -> int:
    try:
        dob = datetime.strptime(dob_str, "%Y-%m-%d")
        today = datetime.today()
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    except Exception:
        return None

# Passenger type validator
def validate_passenger_type(dob: str, passenger_type: str):
    age = calculate_age(dob)
    pt = passenger_type.lower()

    if age is None:
        return JSONResponse(
            status_code=422,
            content={"status": 422, "message": "Please rescan your document again!"}
        )

    if pt == "infant" and not (0 <= age < 2):
        return JSONResponse(
            status_code=400,
            content={"status": 400, "message": "Invalid passenger type: Not an infant."}
        )
    elif pt == "child" and not (2 <= age < 12):
        return JSONResponse(
            status_code=400,
            content={"status": 400, "message": "Invalid passenger type: Not a child."}
        )
    elif pt == "adult" and not (age >= 12):
        return JSONResponse(
            status_code=400,
            content={"status": 400, "message": "Invalid passenger type: Not an adult."}
        )
    return None  # Valid case

@app.post("/scan")
async def scan_document(
    file: UploadFile = File(...),
    doc_type: str = Query(...),
    passenger_type: str = Query(...),
    airline: str = Query(...)
):
    contents = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    # Step 1: OCR
    ocr_text = extract_text(tmp_path)
    print("🔍 OCR TEXT:")
    print(ocr_text)

    # Step 2: Gemini with airline policy
    extracted_data = extract_fields_from_text(ocr_text, doc_type, airline)
    print("🧠 Gemini Output:")
    print(extracted_data)

    if "error" in extracted_data:
        return JSONResponse(
            status_code=500,
            content={"status": 500, "message": extracted_data["error"]}
        )

    # Step 3: Passenger type check (based on DOB)
    structured = extracted_data.get("structured_data", {})
    dob = structured.get("dob")

    if not dob:
        # print("❌ DOB missing in structured_data:", structured)
        return JSONResponse(
            status_code=422,
            content={"status": 422, "message": "DOB missing from document. Please scan again!"}
        )

    # Validate passenger type
    validation_error = validate_passenger_type(dob, passenger_type)
    if validation_error:
        return validation_error
        
    expiry_keys = ["doe", "passport_expiry", "expiry_date", "cnic_expiry"]
    expiry = next((structured.get(k) for k in expiry_keys if structured.get(k)), "")

    print("📅 Expiry Date Found:", expiry)

    if expiry and is_expired(expiry):
        return JSONResponse(
            status_code=400,
            content={"status": 400, "message": f"{doc_type.upper()} is expired!"}
        )

    # Step 4: Return to frontend
    return {
        "status": "success",
        "corrected_json": structured
    }


