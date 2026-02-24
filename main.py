
import os
from groq import Groq
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("❌ GROQ_API_KEY not found. Please set it in the .env file.")

# Configure Groq client
client = Groq(api_key=api_key)
MODEL_NAME = "llama-3.1-8b-instant"

# Initialize FastAPI app
app = FastAPI(
    title="Reverse Geocoding API",
    description="An API to get a human-readable address from geographic coordinates using Groq LLM.",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---

class CoordinatesRequest(BaseModel):
    latitude: float = Field(..., example=12.9716, description="The latitude of the location.")
    longitude: float = Field(..., example=77.5946, description="The longitude of the location.")

class AddressResponse(BaseModel):
    success: bool
    address: str | None = None
    error: str | None = None

# --- Routes ---

@app.get("/")
def home():
    return {"message": f"🌍 Reverse Geocoding API is running with {MODEL_NAME} via Groq!"}


@app.post("/get-address", response_model=AddressResponse)
def get_address_from_coordinates(req: CoordinatesRequest):
    prompt = f"""
You are an expert reverse geocoding service.

Your ONLY task:
Provide a clean, single-line, human-readable address 
for the given geographical coordinates.

Rules:
- Return only the address.
- No explanation.
- No formatting.
- Single line only.

Latitude: {req.latitude}
Longitude: {req.longitude}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You return only clean address text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        address = response.choices[0].message.content.strip()

        return {
            "success": True,
            "address": address,
            "error": None
        }

    except Exception as e:
        print(f"❌ Error from Groq API during geocoding: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred with the Groq API: {str(e)}"
        )