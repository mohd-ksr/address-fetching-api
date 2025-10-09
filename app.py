import os
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found. Please set it in the .env file.")

# Configure the Gemini API with the key
genai.configure(api_key=api_key)

# Use the powerful 'gemini-1.5-pro-latest' model as requested
MODEL_NAME = "gemini-2.0-flash"
model = genai.GenerativeModel(MODEL_NAME)

# Initialize FastAPI app with a specific title and description
app = FastAPI(
    title="Reverse Geocoding API",
    description="An API to get a human-readable address from geographic coordinates using Gemini.",
    version="1.0.0"
)

# Enable CORS to allow frontend applications to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this to your actual frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for Request/Response ---

class CoordinatesRequest(BaseModel):
    latitude: float = Field(..., example=12.9716, description="The latitude of the location.")
    longitude: float = Field(..., example=77.5946, description="The longitude of the location.")

class AddressResponse(BaseModel):
    success: bool
    address: str | None = None
    error: str | None = None

# --- API Routes (Endpoints) ---

@app.get("/")
def home():
    """Root endpoint providing a welcome message."""
    return {"message": f"🌍 Reverse Geocoding API is running with {MODEL_NAME}!"}


@app.post("/get-address", response_model=AddressResponse)
def get_address_from_coordinates(req: CoordinatesRequest):
    """Gets a human-readable address from geographic coordinates."""
    prompt = f"""
    You are an expert reverse geocoding service.
    Your only task is to provide a clean, single-line, human-readable address 
    for the given geographical coordinates. Do not add any extra explanation or formatting.
    Just return the plain text address.

    Latitude: {req.latitude}
    Longitude: {req.longitude}
    """
    try:
        response = model.generate_content(prompt)
        address = response.text.strip()
        # On success, return a JSON object that matches the AddressResponse model
        return {"success": True, "address": address, "error": None}
    except Exception as e:
        print(f"❌ Error from Gemini API during geocoding: {e}")
        # If there is an error, raise an HTTPException which FastAPI handles correctly
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred with the Gemini API: {str(e)}"
        )