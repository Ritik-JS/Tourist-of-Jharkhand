from fastapi import FastAPI, APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel, Field, EmailStr, field_validator, model_validator
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import os
import bcrypt
from jose import jwt, JWTError
import json
import uuid
from pathlib import Path
from services.gemini_service import GeminiService
from models.blockchain_models import BlockchainStatus
from services.blockchain_service import blockchain_service
import aiomysql

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 3001)),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', 'Prince1504'),
    'db': os.getenv('DB_NAME', 'jharkhand_tourism'),
    'autocommit': True
}

# JWT configuration
JWT_SECRET = os.getenv('JWT_SECRET', 'default_secret')
JWT_ALGORITHM = os.getenv('JWT_ALGORITHM', 'HS256')
JWT_EXPIRE_MINUTES = int(os.getenv('JWT_EXPIRE_MINUTES', 1440))

# Create the main app
app = FastAPI(title="Jharkhand Tourism API", version="1.0.0")
@app.get("/api/blockchain/status", response_model=BlockchainStatus)
async def blockchain_status():
    info = blockchain_service.get_network_info()  # Returns dict with keys: connected, network, chain_id, etc.

    # Ensure 'contracts' key exists for Pydantic validation
    contracts_dict = info.get("contracts") or {}

    return BlockchainStatus(
        network=info.get("network", "unknown"),
        connected=info.get("connected", False),
        block_number=info.get("latest_block"),
        gas_price=str(info.get("gas_price_gwei")) if info.get("gas_price_gwei") is not None else None,
        contract_addresses=contracts_dict
    )

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Security
security = HTTPBearer()

# Initialize Gemini service
gemini_service = GeminiService()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection pool
db_pool = None

async def init_db():
    global db_pool
    db_pool = await aiomysql.create_pool(**DB_CONFIG)

async def get_db():
    if not db_pool:
        await init_db()
    return db_pool

# Pydantic models
class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str
    phone: str
    role: str = "tourist"

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class User(BaseModel):
    id: str
    name: str
    email: str
    role: str
    phone: str
    created_at: datetime

class Destination(BaseModel):
    id: str
    name: str
    location: str
    description: str
    image_url: str
    rating: float
    price: float
    category: str
    highlights: List[str]
    created_at: datetime

class Provider(BaseModel):
    id: str
    user_id: str
    name: str
    category: str
    service_name: str
    description: str
    price: float
    rating: float
    location: str
    contact: str
    image_url: str
    is_active: bool
    destination_id: Optional[str] = None
    created_at: datetime

class Review(BaseModel):
    id: str
    user_id: str
    destination_id: Optional[str] = None
    provider_id: Optional[str] = None
    rating: int
    comment: str
    created_at: datetime
    


# Utility functions
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
                user_data = await cur.fetchone()
                if user_data is None:
                    raise HTTPException(status_code=401, detail="User not found")
                return user_data
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# API Routes
@api_router.get("/")
async def root():
    return {"message": "Jharkhand Tourism API is running!"}

@api_router.post("/auth/register")
async def register_user(user_data: UserCreate):
    try:
        # Block admin role registration for security
        if user_data.role == "admin":
            raise HTTPException(status_code=403, detail="Admin registration is not allowed through public registration")
        
        # Ensure only valid roles are allowed
        if user_data.role not in ["tourist", "provider"]:
            raise HTTPException(status_code=400, detail="Invalid role. Only 'tourist' and 'provider' roles are allowed")
        
        pool = await get_db()
        user_id = str(uuid.uuid4())
        hashed_password = hash_password(user_data.password)
        
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                # Check if user already exists
                await cur.execute("SELECT id FROM users WHERE email = %s", (user_data.email,))
                if await cur.fetchone():
                    raise HTTPException(status_code=400, detail="Email already registered")
                
                # Insert new user
                await cur.execute("""
                    INSERT INTO users (id, name, email, password, role, phone)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (user_id, user_data.name, user_data.email, hashed_password, user_data.role, user_data.phone))
                
                # Create access token
                access_token = create_access_token(data={"sub": user_id})
                
                return {
                    "access_token": access_token,
                    "token_type": "bearer",
                    "user": {
                        "id": user_id,
                        "name": user_data.name,
                        "email": user_data.email,
                        "role": user_data.role,
                        "phone": user_data.phone
                    }
                }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/auth/login")
async def login_user(user_credentials: UserLogin):
    try:
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute("SELECT * FROM users WHERE email = %s", (user_credentials.email,))
                user_data = await cur.fetchone()
                
                if not user_data or not verify_password(user_credentials.password, user_data['password']):
                    raise HTTPException(status_code=401, detail="Invalid credentials")
                
                # Create access token
                access_token = create_access_token(data={"sub": user_data['id']})
                
                return {
                    "access_token": access_token,
                    "token_type": "bearer",
                    "user": {
                        "id": user_data['id'],
                        "name": user_data['name'],
                        "email": user_data['email'],
                        "role": user_data['role'],
                        "phone": user_data['phone']
                    }
                }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/auth/me")
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    return {
        "id": current_user['id'],
        "name": current_user['name'],
        "email": current_user['email'],
        "role": current_user['role'],
        "phone": current_user['phone']
    }

@api_router.get("/regions")
async def get_regions():
    """Get all regions in Jharkhand with user-friendly names"""
    try:
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute("SELECT * FROM regions ORDER BY name")
                regions = await cur.fetchall()
                
                # Add user-friendly region mapping
                region_mapping = {
                    'kolhan': 'east',
                    'north_chotanagpur': 'north', 
                    'south_chotanagpur': 'south',
                    'santhal_pargana': 'central',  # Keep as central for now, but user wanted east
                    'palamu': 'west'
                }
                
                # Parse JSON highlights and add user-friendly names
                for region in regions:
                    if region['highlights']:
                        region['highlights'] = json.loads(region['highlights'])
                    else:
                        region['highlights'] = []
                    
                    # Add user-friendly region code
                    region['region_code'] = region_mapping.get(region['id'], region['id'])
                    
                    # Override specific regions based on user request (central -> east)
                    if region['id'] == 'santhal_pargana':
                        region['region_code'] = 'east'  # User specifically requested east instead of central
                        region['user_friendly_name'] = 'East Jharkhand'
                
                return regions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/destinations")
async def get_destinations(category: Optional[str] = None, region: Optional[str] = None, limit: int = 50):
    """Get destinations with optional category and region filtering"""
    try:
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                # Build query with filters
                query = "SELECT * FROM destinations WHERE 1=1"
                params = []
                
                if category:
                    query += " AND category = %s"
                    params.append(category)
                
                if region:
                    # Create region mapping for better filtering - Updated per user request
                    region_mapping = {
                        # User-friendly names to database values (east instead of central as requested)
                        'east': ['Santhal Pargana Division'],  # User requested east instead of central
                        'west': ['Palamu Division'], 
                        'north': ['North Chhotanagpur Division'],
                        'south': ['South Chhotanagpur Division'],
                        'central': ['Kolhan Division'],  # Move Kolhan to central
                        # Handle variations and full names
                        'kolhan': ['Kolhan Division'],
                        'north_chotanagpur': ['North Chhotanagpur Division'],
                        'south_chotanagpur': ['South Chhotanagpur Division'],
                        'santhal_pargana': ['Santhal Pargana Division'],
                        'palamu': ['Palamu Division'],
                        # Direct matches
                        'Kolhan Division': ['Kolhan Division'],
                        'North Chhotanagpur Division': ['North Chhotanagpur Division'],
                        'South Chhotanagpur Division': ['South Chhotanagpur Division'],
                        'Santhal Pargana Division': ['Santhal Pargana Division'],
                        'Palamu Division': ['Palamu Division']
                    }
                    
                    region_lower = region.lower()
                    matched_regions = None
                    
                    # Try to find a matching region
                    for key, values in region_mapping.items():
                        if key.lower() == region_lower or region.lower() in key.lower():
                            matched_regions = values
                            break
                    
                    if matched_regions:
                        # Use the mapped region values
                        placeholders = ', '.join(['%s'] * len(matched_regions))
                        query += f" AND region IN ({placeholders})"
                        params.extend(matched_regions)
                    else:
                        # Fallback: try direct match or LIKE match
                        query += " AND (region = %s OR region LIKE %s)"
                        params.append(region)
                        params.append(f"%{region}%")
                
                query += " ORDER BY name LIMIT %s"
                params.append(limit)
                
                await cur.execute(query, params)
                destinations = await cur.fetchall()
                
                # Parse JSON highlights
                for dest in destinations:
                    if dest['highlights']:
                        dest['highlights'] = json.loads(dest['highlights'])
                    else:
                        dest['highlights'] = []
                
                return destinations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/destinations/{destination_id}")
async def get_destination_detail(destination_id: str):
    try:
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute("SELECT * FROM destinations WHERE id = %s", (destination_id,))
                destination = await cur.fetchone()
                
                if not destination:
                    raise HTTPException(status_code=404, detail="Destination not found")
                
                # Parse JSON highlights
                if destination['highlights']:
                    destination['highlights'] = json.loads(destination['highlights'])
                else:
                    destination['highlights'] = []
                
                return destination
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/destinations/list/dropdown")
async def get_destinations_for_dropdown():
    """Get simplified list of destinations for dropdown selection"""
    try:
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute("""
                    SELECT id, name, location 
                    FROM destinations 
                    ORDER BY name ASC
                """)
                destinations = await cur.fetchall()
                return destinations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/providers")
async def get_providers(category: Optional[str] = None, location: Optional[str] = None, destination_id: Optional[str] = None, limit: int = 50):
    try:
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                if destination_id:
                    # Get providers for specific destination using destination_id relationship
                    query = """
                        SELECT p.*, 
                               d.name as destination_name,
                               d.location as destination_location,
                               AVG(r.rating) as avg_rating,
                               COUNT(r.id) as review_count
                        FROM providers p
                        LEFT JOIN destinations d ON p.destination_id = d.id
                        LEFT JOIN reviews r ON p.id = r.provider_id
                        WHERE p.is_active = 1 AND p.destination_id = %s
                    """
                    params = [destination_id]
                    
                    if category:
                        query += " AND p.category = %s"
                        params.append(category)
                        
                    query += " GROUP BY p.id ORDER BY avg_rating DESC, p.rating DESC LIMIT %s"
                    params.append(limit)
                else:
                    # Get all providers with optional filters
                    query = """
                        SELECT p.*, 
                               d.name as destination_name,
                               d.location as destination_location,
                               AVG(r.rating) as avg_rating,
                               COUNT(r.id) as review_count
                        FROM providers p
                        LEFT JOIN destinations d ON p.destination_id = d.id
                        LEFT JOIN reviews r ON p.id = r.provider_id
                        WHERE p.is_active = 1
                    """
                    params = []
                    
                    if category:
                        query += " AND p.category = %s"
                        params.append(category)
                    
                    if location:
                        query += " AND (p.location LIKE %s OR d.location LIKE %s)"
                        params.extend([f"%{location}%", f"%{location}%"])
                    
                    query += " GROUP BY p.id ORDER BY avg_rating DESC, p.rating DESC LIMIT %s"
                    params.append(limit)
                
                await cur.execute(query, params)
                providers = await cur.fetchall()
                
                return providers
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/reviews")
async def get_reviews(destination_id: Optional[str] = None, provider_id: Optional[str] = None, limit: int = 20):
    try:
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                query = """
                    SELECT r.*, u.name as user_name 
                    FROM reviews r 
                    JOIN users u ON r.user_id = u.id
                """
                params = []
                
                conditions = []
                if destination_id:
                    conditions.append("r.destination_id = %s")
                    params.append(destination_id)
                
                if provider_id:
                    conditions.append("r.provider_id = %s")
                    params.append(provider_id)
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                
                query += " ORDER BY r.created_at DESC LIMIT %s"
                params.append(limit)
                
                await cur.execute(query, params)
                reviews = await cur.fetchall()
                
                return reviews
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/planner")
async def generate_itinerary(
    request_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Generate AI-powered travel itinerary using Deepseek"""
    try:
        # Extract user preferences from request
        preferences = {
            "destinations": request_data.get("destinations", ["Ranchi"]),
            "budget": request_data.get("budget", 15000),
            "days": request_data.get("days", 3),
            "interests": request_data.get("interests", ["Sightseeing"]),
            "travel_style": request_data.get("travel_style", "balanced"),
            "group_size": request_data.get("group_size", 2)
        }
        
        # Generate itinerary using Gemini
        itinerary = await gemini_service.generate_itinerary(preferences)
        
        # Optionally save to database
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    INSERT INTO itineraries (id, user_id, destination, days, budget, content, preferences, generated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    itinerary["id"],
                    current_user["id"],
                    itinerary["destination"],
                    itinerary["days"],
                    itinerary["budget"],
                    itinerary["content"],
                    json.dumps(itinerary["preferences"]),
                    itinerary["generated_at"]
                ))
        
        return itinerary
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating itinerary: {str(e)}")

@api_router.post("/chatbot")
async def chatbot_message(
    request_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Handle chatbot conversation using Deepseek"""
    try:
        user_message = request_data.get("message", "")
        session_id = request_data.get("session_id", str(uuid.uuid4()))
        
        if not user_message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        # Get conversation history (optional)
        pool = await get_db()
        conversation_history = []
        
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                # Get recent conversation history
                await cur.execute("""
                    SELECT message, response, created_at 
                    FROM chat_logs 
                    WHERE user_id = %s AND session_id = %s 
                    ORDER BY created_at DESC 
                    LIMIT 5
                """, (current_user["id"], session_id))
                
                history = await cur.fetchall()
                for chat in reversed(history):
                    conversation_history.extend([
                        {"role": "user", "content": chat["message"]},
                        {"role": "assistant", "content": chat["response"]}
                    ])
        
        # Generate response using Gemini
        response = await gemini_service.chat_response(user_message, conversation_history)
        
        # Save conversation to database
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    INSERT INTO chat_logs (id, user_id, session_id, message, response, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    str(uuid.uuid4()),
                    current_user["id"],
                    session_id,
                    user_message,
                    response["message"],
                    datetime.utcnow()
                ))
        
        return {
            "response": response["message"],
            "session_id": session_id,
            "timestamp": response["timestamp"],
            "model": response["model"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat message: {str(e)}")

@api_router.get("/chatbot/history/{session_id}")
async def get_chat_history(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get chat history for a session"""
    try:
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute("""
                    SELECT message, response, created_at 
                    FROM chat_logs 
                    WHERE user_id = %s AND session_id = %s 
                    ORDER BY created_at ASC
                """, (current_user["id"], session_id))
                
                history = await cur.fetchall()
                
                # Format for frontend
                formatted_history = []
                for chat in history:
                    formatted_history.extend([
                        {
                            "id": f"user_{chat['created_at'].timestamp()}",
                            "text": chat["message"],
                            "sender": "user",
                            "timestamp": chat["created_at"].isoformat()
                        },
                        {
                            "id": f"bot_{chat['created_at'].timestamp()}",
                            "text": chat["response"],
                            "sender": "bot",
                            "timestamp": chat["created_at"].isoformat()
                        }
                    ])
                
                return formatted_history
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Booking Management API
class BookingCreate(BaseModel):
    provider_id: str = Field(..., min_length=1, description="Provider ID is required")
    destination_id: str = Field(..., min_length=1, description="Destination ID is required")
    booking_date: str = Field(..., description="Booking date in YYYY-MM-DD format")
    check_in: str = Field(..., description="Check-in date in YYYY-MM-DD format") 
    check_out: str = Field(..., description="Check-out date in YYYY-MM-DD format")
    guests: int = Field(default=1, ge=1, le=20, description="Number of guests (1-20)")
    rooms: int = Field(default=1, ge=1, le=10, description="Number of rooms (1-10)")
    special_requests: Optional[str] = Field(None, max_length=500, description="Special requests")
    city_origin: Optional[str] = Field(None, max_length=100, description="City of origin")
    calculated_price: Optional[float] = Field(None, ge=0, description="Calculated price from frontend")
    addons: Optional[str] = Field(None, description="JSON string of selected addons")
    # Package information for tourism packages
    package_type: Optional[str] = Field(None, max_length=50, description="Type of tourism package (heritage, adventure, spiritual, premium)")
    package_name: Optional[str] = Field(None, max_length=100, description="Name of the tourism package")
    # Personal information from booking form
    booking_full_name: str = Field(..., min_length=1, description="Full name from booking form")
    booking_email: EmailStr = Field(..., description="Email from booking form")
    booking_phone: str = Field(..., min_length=10, description="Phone number from booking form")
    # Reference number for customer/provider communication
    reference_number: Optional[str] = Field(None, max_length=20, description="Frontend generated reference number (e.g., JH123456)")
    # 🔗 PHASE 6.1: Blockchain Integration
    blockchain_verification: Optional[bool] = Field(False, description="Whether user requested blockchain verification")
    
    @field_validator('booking_date', 'check_in', 'check_out')
    @classmethod
    def validate_date_format(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD format')
    
    @model_validator(mode='after')
    def validate_dates(self):
        try:
            booking_date = datetime.strptime(self.booking_date, '%Y-%m-%d')
            check_in = datetime.strptime(self.check_in, '%Y-%m-%d')  
            check_out = datetime.strptime(self.check_out, '%Y-%m-%d')
            
            if check_in >= check_out:
                raise ValueError('Check-out date must be after check-in date')
            
            if booking_date < datetime.now().replace(hour=0, minute=0, second=0, microsecond=0):
                raise ValueError('Booking date cannot be in the past')
                
        except ValueError as e:
            if 'does not match format' in str(e):
                raise ValueError('Invalid date format. Use YYYY-MM-DD format')
            raise e
            
        return self

@api_router.post("/bookings")
async def create_booking(
    booking_data: BookingCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a new booking with blockchain integration"""
    try:
        pool = await get_db()
        booking_id = str(uuid.uuid4())
        
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                # Get provider and destination details
                await cur.execute("SELECT name, price FROM providers WHERE id = %s", (booking_data.provider_id,))
                provider = await cur.fetchone()
                if not provider:
                    raise HTTPException(status_code=404, detail="Provider not found")
                
                await cur.execute("SELECT name FROM destinations WHERE id = %s", (booking_data.destination_id,))
                destination = await cur.fetchone()
                if not destination:
                    raise HTTPException(status_code=404, detail="Destination not found")
                
                # Use calculated price from frontend if provided, otherwise calculate from provider/destination
                if booking_data.calculated_price and booking_data.calculated_price > 0:
                    total_price = booking_data.calculated_price
                else:
                    total_price = (provider['price'] + destination['price']) * booking_data.guests
                
                # 🔗 PHASE 6.1: Blockchain Integration - Check if user wants blockchain verification
                blockchain_verified = False
                blockchain_hash = None
                certificate_eligible = False
                
                # Check if blockchain verification was requested (from frontend)
                blockchain_verification_requested = getattr(booking_data, 'blockchain_verification', False)
                
                # Create booking with personal information and package details
                await cur.execute("""
                    INSERT INTO bookings (id, user_id, provider_id, destination_id, user_name, 
                                        provider_name, destination_name, booking_date, check_in, 
                                        check_out, guests, rooms, total_price, special_requests, status,
                                        addons, package_type, package_name, booking_full_name, booking_email, 
                                        booking_phone, city_origin, reference_number, blockchain_verified, 
                                        blockchain_hash, certificate_eligible)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    booking_id, current_user['id'], booking_data.provider_id, booking_data.destination_id,
                    current_user['name'], provider['name'], destination['name'], booking_data.booking_date,
                    booking_data.check_in, booking_data.check_out, booking_data.guests, booking_data.rooms,
                    total_price, booking_data.special_requests, 'pending',
                    booking_data.addons, booking_data.package_type, booking_data.package_name, 
                    booking_data.booking_full_name, booking_data.booking_email, booking_data.booking_phone, 
                    booking_data.city_origin, booking_data.reference_number, blockchain_verified, 
                    blockchain_hash, certificate_eligible
                ))
                
                # 🔗 PHASE 6.1: Auto-Award Initial Loyalty Points for Booking
                loyalty_points_awarded = 0
                if blockchain_verification_requested:
                    try:
                        # Award base loyalty points for booking (10% of price in points)
                        loyalty_points_awarded = int(total_price * 0.1)
                        
                        # Award points in database first
                        await cur.execute("""
                            INSERT INTO loyalty_points (id, user_id, points_balance, total_earned, total_redeemed)
                            VALUES (UUID(), %s, %s, %s, 0)
                            ON DUPLICATE KEY UPDATE 
                            points_balance = points_balance + VALUES(points_balance),
                            total_earned = total_earned + VALUES(total_earned)
                        """, (current_user['id'], loyalty_points_awarded, loyalty_points_awarded))
                        
                        # Log loyalty transaction
                        await cur.execute("""
                            INSERT INTO loyalty_transactions (id, user_id, transaction_type, points_amount, 
                                                           description, booking_id, status)
                            VALUES (UUID(), %s, 'earned', %s, %s, %s, 'completed')
                        """, (current_user['id'], loyalty_points_awarded, 
                             f"Booking reward for {destination['name']}", booking_id))
                        
                        # Mark booking as certificate eligible for completed tours
                        await cur.execute("""
                            UPDATE bookings SET certificate_eligible = TRUE WHERE id = %s
                        """, (booking_id,))
                        
                    except Exception as loyalty_error:
                        print(f"Failed to award loyalty points: {loyalty_error}")
                
                response = {
                    "id": booking_id,
                    "status": "pending",
                    "total_price": total_price,
                    "package_type": booking_data.package_type,
                    "package_name": booking_data.package_name,
                    "addons": booking_data.addons,
                    "booking_full_name": booking_data.booking_full_name,
                    "booking_email": booking_data.booking_email,
                    "reference_number": booking_data.reference_number,
                    "blockchain_verified": blockchain_verified,
                    "certificate_eligible": certificate_eligible,
                    "loyalty_points_awarded": loyalty_points_awarded,
                    "message": "Booking created successfully"
                }
                
                if blockchain_verification_requested:
                    response["blockchain_message"] = f"Blockchain features enabled! Earned {loyalty_points_awarded} loyalty points."
                
                return response
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/bookings")
async def get_user_bookings(current_user: dict = Depends(get_current_user)):
    """Get all bookings for current user"""
    try:
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute("""
                    SELECT * FROM bookings WHERE user_id = %s ORDER BY created_at DESC
                """, (current_user['id'],))
                bookings = await cur.fetchall()
                return bookings
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/provider/bookings")
async def get_provider_bookings(current_user: dict = Depends(get_current_user)):
    """Get all bookings for current provider"""
    try:
        if current_user['role'] != 'provider':
            raise HTTPException(status_code=403, detail="Access denied")
        
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute("""
                    SELECT b.* FROM bookings b 
                    JOIN providers p ON b.provider_id = p.id 
                    WHERE p.user_id = %s ORDER BY b.created_at DESC
                """, (current_user['id'],))
                bookings = await cur.fetchall()
                return bookings
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/provider/bookings/search")
async def search_bookings_by_reference(
    reference_number: str,
    current_user: dict = Depends(get_current_user)
):
    """Search bookings by reference number for providers"""
    try:
        if current_user['role'] != 'provider':
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not reference_number or len(reference_number.strip()) == 0:
            raise HTTPException(status_code=400, detail="Reference number is required")
        
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                # Search for bookings by reference number for this provider
                # First check if reference_number column exists, if not search by booking ID
                await cur.execute("SHOW COLUMNS FROM bookings LIKE 'reference_number'")
                reference_column_exists = await cur.fetchone()
                
                if reference_column_exists:
                    # Search by reference_number column
                    await cur.execute("""
                        SELECT b.* FROM bookings b 
                        JOIN providers p ON b.provider_id = p.id 
                        WHERE p.user_id = %s AND b.reference_number = %s
                        ORDER BY b.created_at DESC
                    """, (current_user['id'], reference_number.strip()))
                else:
                    # Fallback: search by booking ID (which might be used as reference)
                    await cur.execute("""
                        SELECT b.* FROM bookings b 
                        JOIN providers p ON b.provider_id = p.id 
                        WHERE p.user_id = %s AND (b.id = %s OR b.id LIKE %s)
                        ORDER BY b.created_at DESC
                    """, (current_user['id'], reference_number.strip(), f"%{reference_number.strip()}%"))
                
                bookings = await cur.fetchall()
                
                if not bookings:
                    return {
                        "message": "No bookings found with the provided reference number",
                        "reference_number": reference_number.strip(),
                        "bookings": []
                    }
                
                return {
                    "message": f"Found {len(bookings)} booking(s) with reference number {reference_number.strip()}",
                    "reference_number": reference_number.strip(),
                    "bookings": bookings
                }
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching bookings: {str(e)}")

@api_router.put("/bookings/{booking_id}/status")
async def update_booking_status(
    booking_id: str,
    status_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Update booking status with blockchain integration"""
    try:
        new_status = status_data.get('status')
        if new_status not in ['confirmed', 'cancelled', 'completed']:
            raise HTTPException(status_code=400, detail="Invalid status")
        
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                # Check if user has permission to update this booking
                if current_user['role'] == 'provider':
                    await cur.execute("""
                        SELECT b.*, d.name as destination_name FROM bookings b 
                        JOIN providers p ON b.provider_id = p.id 
                        JOIN destinations d ON b.destination_id = d.id
                        WHERE b.id = %s AND p.user_id = %s
                    """, (booking_id, current_user['id']))
                elif current_user['role'] == 'admin':
                    await cur.execute("""
                        SELECT b.*, d.name as destination_name FROM bookings b
                        JOIN destinations d ON b.destination_id = d.id
                        WHERE b.id = %s
                    """, (booking_id,))
                else:
                    await cur.execute("""
                        SELECT b.*, d.name as destination_name FROM bookings b
                        JOIN destinations d ON b.destination_id = d.id
                        WHERE b.id = %s AND b.user_id = %s
                    """, (booking_id, current_user['id']))
                
                booking = await cur.fetchone()
                if not booking:
                    raise HTTPException(status_code=404, detail="Booking not found or access denied")
                
                # Update booking status
                await cur.execute("UPDATE bookings SET status = %s WHERE id = %s", (new_status, booking_id))
                
                response = {"message": "Booking status updated successfully"}
                
                # 🔗 PHASE 6.1: Auto-issue certificate when tour is completed
                if new_status == 'completed' and booking.get('certificate_eligible', False):
                    try:
                        # Check if certificate already issued
                        await cur.execute("SELECT id FROM certificates WHERE booking_id = %s", (booking_id,))
                        existing_cert = await cur.fetchone()
                        
                        if not existing_cert:
                            # Create certificate record
                            cert_id = str(uuid.uuid4())
                            await cur.execute("""
                                INSERT INTO certificates (
                                    id, user_id, booking_id, certificate_type, contract_address,
                                    certificate_title, certificate_description, destination_name,
                                    completion_date, is_minted
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """, (
                                cert_id, booking['user_id'], booking_id, 'tour_completion',
                                os.getenv('CONTRACT_ADDRESS_CERTIFICATES', 'pending'),
                                f"Tourism Certificate - {booking['destination_name']}",
                                f"Successfully completed tour package: {booking.get('package_name', 'Tourism Experience')}",
                                booking['destination_name'],
                                datetime.now().date(),
                                False  # Will be minted when user requests it
                            ))
                            
                            # Mark booking as certificate issued
                            await cur.execute("UPDATE bookings SET certificate_issued = TRUE WHERE id = %s", (booking_id,))
                            
                            # 🔗 PHASE 6.1: Award bonus loyalty points for completion
                            completion_bonus = 50  # Fixed bonus points for completing a tour
                            
                            await cur.execute("""
                                INSERT INTO loyalty_points (id, user_id, points_balance, total_earned, total_redeemed)
                                VALUES (UUID(), %s, %s, %s, 0)
                                ON DUPLICATE KEY UPDATE 
                                points_balance = points_balance + VALUES(points_balance),
                                total_earned = total_earned + VALUES(total_earned)
                            """, (booking['user_id'], completion_bonus, completion_bonus))
                            
                            # Log completion bonus transaction
                            await cur.execute("""
                                INSERT INTO loyalty_transactions (id, user_id, transaction_type, points_amount, 
                                                               description, booking_id, status)
                                VALUES (UUID(), %s, 'earned', %s, %s, %s, 'completed')
                            """, (booking['user_id'], completion_bonus, 
                                 f"Tour completion bonus - {booking['destination_name']}", booking_id))
                            
                            response["certificate_issued"] = True
                            response["certificate_id"] = cert_id
                            response["bonus_points_awarded"] = completion_bonus
                            response["blockchain_message"] = f"Certificate ready for minting! Earned {completion_bonus} bonus points."
                    
                    except Exception as cert_error:
                        print(f"Failed to issue certificate: {cert_error}")
                        response["certificate_error"] = "Failed to issue certificate automatically"
                
                return response
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Wishlist API
class WishlistItemCreate(BaseModel):
    destination_id: str

@api_router.get("/wishlist")
async def get_user_wishlist(current_user: dict = Depends(get_current_user)):
    """Get all wishlist items for current user"""
    try:
        if current_user['role'] != 'tourist':
            raise HTTPException(status_code=403, detail="Only tourists can access wishlist")
        
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute("""
                    SELECT w.id, w.user_id, w.destination_id, w.created_at,
                           d.name, d.location, d.description, d.image_url, 
                           d.rating, d.price, d.category, d.highlights
                    FROM wishlist w
                    JOIN destinations d ON w.destination_id = d.id
                    WHERE w.user_id = %s
                    ORDER BY w.created_at DESC
                """, (current_user['id'],))
                wishlist_items = await cur.fetchall()
                
                # Format the response
                formatted_items = []
                for item in wishlist_items:
                    # Parse highlights JSON if it exists
                    highlights = []
                    if item['highlights']:
                        try:
                            highlights = json.loads(item['highlights'])
                        except:
                            highlights = []
                    
                    formatted_items.append({
                        'id': item['id'],
                        'user_id': item['user_id'],
                        'destination_id': item['destination_id'],
                        'created_at': item['created_at'],
                        'destination': {
                            'id': item['destination_id'],
                            'name': item['name'],
                            'location': item['location'],
                            'description': item['description'],
                            'image_url': item['image_url'],
                            'rating': float(item['rating']) if item['rating'] else 0,
                            'price': float(item['price']),
                            'category': item['category'],
                            'highlights': highlights
                        }
                    })
                
                return {
                    'items': formatted_items,
                    'total_count': len(formatted_items)
                }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/wishlist")
async def add_to_wishlist(
    wishlist_item: WishlistItemCreate,
    current_user: dict = Depends(get_current_user)
):
    """Add destination to user's wishlist"""
    try:
        if current_user['role'] != 'tourist':
            raise HTTPException(status_code=403, detail="Only tourists can manage wishlist")
        
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                # Check if destination exists
                await cur.execute("SELECT id FROM destinations WHERE id = %s", (wishlist_item.destination_id,))
                if not await cur.fetchone():
                    raise HTTPException(status_code=404, detail="Destination not found")
                
                # Check if already in wishlist
                await cur.execute("""
                    SELECT id FROM wishlist WHERE user_id = %s AND destination_id = %s
                """, (current_user['id'], wishlist_item.destination_id))
                
                if await cur.fetchone():
                    raise HTTPException(status_code=400, detail="Destination already in wishlist")
                
                # Add to wishlist
                wishlist_id = str(uuid.uuid4())
                await cur.execute("""
                    INSERT INTO wishlist (id, user_id, destination_id)
                    VALUES (%s, %s, %s)
                """, (wishlist_id, current_user['id'], wishlist_item.destination_id))
                
                return {"message": "Destination added to wishlist successfully", "id": wishlist_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.delete("/wishlist/{destination_id}")
async def remove_from_wishlist(
    destination_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Remove destination from user's wishlist"""
    try:
        if current_user['role'] != 'tourist':
            raise HTTPException(status_code=403, detail="Only tourists can manage wishlist")
        
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                # Check if item exists in wishlist
                await cur.execute("""
                    SELECT id FROM wishlist WHERE user_id = %s AND destination_id = %s
                """, (current_user['id'], destination_id))
                
                if not await cur.fetchone():
                    raise HTTPException(status_code=404, detail="Destination not found in wishlist")
                
                # Remove from wishlist
                await cur.execute("""
                    DELETE FROM wishlist WHERE user_id = %s AND destination_id = %s
                """, (current_user['id'], destination_id))
                
                return {"message": "Destination removed from wishlist successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/wishlist/check/{destination_id}")
async def check_wishlist_status(
    destination_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Check if destination is in user's wishlist"""
    try:
        if current_user['role'] != 'tourist':
            return {"is_wishlisted": False}
        
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT id FROM wishlist WHERE user_id = %s AND destination_id = %s
                """, (current_user['id'], destination_id))
                
                is_wishlisted = await cur.fetchone() is not None
                return {"is_wishlisted": is_wishlisted}
    except Exception as e:
        return {"is_wishlisted": False}

# Provider Management API
class ProviderCreate(BaseModel):
    name: str
    category: str
    service_name: str
    description: str
    price: float
    destination_id: str  # Changed from location to destination_id
    contact: str
    image_url: Optional[str] = None

@api_router.post("/providers")
async def create_provider(
    provider_data: ProviderCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a new provider service"""
    try:
        if current_user['role'] != 'provider':
            raise HTTPException(status_code=403, detail="Only providers can create services")
        
        pool = await get_db()
        provider_id = str(uuid.uuid4())
        
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                # First get the destination location for backward compatibility
                await cur.execute("SELECT location FROM destinations WHERE id = %s", (provider_data.destination_id,))
                dest_result = await cur.fetchone()
                
                if not dest_result:
                    raise HTTPException(status_code=400, detail="Invalid destination_id")
                
                destination_location = dest_result[0]
                
                await cur.execute("""
                    INSERT INTO providers (id, user_id, name, category, service_name, description, 
                                         price, location, contact, image_url, is_active, destination_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    provider_id, current_user['id'], provider_data.name, provider_data.category,
                    provider_data.service_name, provider_data.description, provider_data.price,
                    destination_location, provider_data.contact, provider_data.image_url, True, provider_data.destination_id
                ))
                
                return {
                    "id": provider_id,
                    "message": "Provider service created successfully"
                }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Provider-Destination Management API
@api_router.post("/providers/{provider_id}/destinations")
async def add_provider_to_destination(
    provider_id: str,
    destination_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Link provider to a destination"""
    try:
        if current_user['role'] != 'provider':
            raise HTTPException(status_code=403, detail="Only providers can manage their services")
        
        destination_id = destination_data.get('destination_id')
        if not destination_id:
            raise HTTPException(status_code=400, detail="destination_id is required")
        
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                # Check if provider belongs to current user
                await cur.execute("SELECT user_id FROM providers WHERE id = %s", (provider_id,))
                provider = await cur.fetchone()
                if not provider or provider[0] != current_user['id']:
                    raise HTTPException(status_code=404, detail="Provider not found or access denied")
                
                # Check if destination exists
                await cur.execute("SELECT id FROM destinations WHERE id = %s", (destination_id,))
                if not await cur.fetchone():
                    raise HTTPException(status_code=404, detail="Destination not found")
                
                # Check if relationship already exists
                await cur.execute("""
                    SELECT id FROM provider_destinations 
                    WHERE provider_id = %s AND destination_id = %s
                """, (provider_id, destination_id))
                
                if await cur.fetchone():
                    raise HTTPException(status_code=400, detail="Provider already linked to this destination")
                
                # Create the relationship
                pd_id = str(uuid.uuid4())
                await cur.execute("""
                    INSERT INTO provider_destinations (id, provider_id, destination_id)
                    VALUES (%s, %s, %s)
                """, (pd_id, provider_id, destination_id))
                
                return {"message": "Provider linked to destination successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.delete("/providers/{provider_id}/destinations/{destination_id}")
async def remove_provider_from_destination(
    provider_id: str,
    destination_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Remove provider from destination"""
    try:
        if current_user['role'] != 'provider':
            raise HTTPException(status_code=403, detail="Only providers can manage their services")
        
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                # Check if provider belongs to current user
                await cur.execute("SELECT user_id FROM providers WHERE id = %s", (provider_id,))
                provider = await cur.fetchone()
                if not provider or provider[0] != current_user['id']:
                    raise HTTPException(status_code=404, detail="Provider not found or access denied")
                
                # Remove the relationship
                await cur.execute("""
                    DELETE FROM provider_destinations 
                    WHERE provider_id = %s AND destination_id = %s
                """, (provider_id, destination_id))
                
                return {"message": "Provider removed from destination successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/user/providers")
async def get_user_providers(current_user: dict = Depends(get_current_user)):
    """Get all providers for current user"""
    try:
        if current_user['role'] != 'provider':
            raise HTTPException(status_code=403, detail="Access denied")
        
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute("SELECT * FROM providers WHERE user_id = %s", (current_user['id'],))
                providers = await cur.fetchall()
                return providers
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/providers/{provider_id}")
async def get_provider_by_id(
    provider_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get provider by ID - for editing purposes"""
    try:
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                # Check ownership
                await cur.execute("SELECT user_id FROM providers WHERE id = %s", (provider_id,))
                provider_check = await cur.fetchone()
                if not provider_check or provider_check['user_id'] != current_user['id']:
                    raise HTTPException(status_code=404, detail="Provider not found or access denied")
                
                # Get provider details with destination info
                query = """
                    SELECT p.*, 
                           d.name as destination_name,
                           d.location as destination_location
                    FROM providers p
                    LEFT JOIN destinations d ON p.destination_id = d.id
                    WHERE p.id = %s
                """
                await cur.execute(query, (provider_id,))
                provider = await cur.fetchone()
                
                if not provider:
                    raise HTTPException(status_code=404, detail="Provider not found")
                
                return provider
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.put("/providers/{provider_id}")
async def update_provider(
    provider_id: str,
    provider_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Update provider service"""
    try:
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                # Check ownership
                await cur.execute("SELECT user_id FROM providers WHERE id = %s", (provider_id,))
                provider = await cur.fetchone()
                if not provider or provider[0] != current_user['id']:
                    raise HTTPException(status_code=404, detail="Provider not found or access denied")
                
                # Update provider  
                update_fields = []
                update_values = []
                for field, value in provider_data.items():
                    if field == 'destination_id':
                        # Get destination location for backward compatibility
                        await cur.execute("SELECT location FROM destinations WHERE id = %s", (value,))
                        dest_result = await cur.fetchone()
                        if dest_result:
                            update_fields.append("destination_id = %s")
                            update_values.append(value)
                            update_fields.append("location = %s")
                            update_values.append(dest_result[0])
                    elif field in ['name', 'category', 'service_name', 'description', 'price', 'contact', 'image_url', 'is_active']:
                        update_fields.append(f"{field} = %s")
                        update_values.append(value)
                
                if update_fields:
                    update_values.append(provider_id)
                    query = f"UPDATE providers SET {', '.join(update_fields)} WHERE id = %s"
                    await cur.execute(query, update_values)
                
                return {"message": "Provider updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Reviews API
class ReviewCreate(BaseModel):
    destination_id: Optional[str] = None
    provider_id: Optional[str] = None
    booking_id: Optional[str] = None  # 🔗 PHASE 6.2: Link review to verified booking
    rating: int
    comment: str
    blockchain_verification: Optional[bool] = Field(False, description="Request blockchain verification for authentic review")

@api_router.post("/reviews")
async def create_review(
    review_data: ReviewCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a new review with optional blockchain verification"""
    try:
        if not review_data.destination_id and not review_data.provider_id:
            raise HTTPException(status_code=400, detail="Either destination_id or provider_id is required")
        
        if review_data.rating < 1 or review_data.rating > 5:
            raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
        
        pool = await get_db()
        review_id = str(uuid.uuid4())
        
        # 🔗 PHASE 6.2: Verify user eligibility for blockchain-verified reviews
        verified_booking = None
        if review_data.blockchain_verification and review_data.booking_id:
            async with pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cur:
                    # Check if user has a completed booking for this destination/provider
                    await cur.execute("""
                        SELECT * FROM bookings 
                        WHERE id = %s AND user_id = %s AND status = 'completed'
                        AND (destination_id = %s OR provider_id = %s)
                    """, (review_data.booking_id, current_user['id'], 
                         review_data.destination_id, review_data.provider_id))
                    verified_booking = await cur.fetchone()
                    
                    if not verified_booking:
                        raise HTTPException(
                            status_code=403, 
                            detail="Blockchain-verified reviews require a completed booking for this destination/provider"
                        )
        
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    INSERT INTO reviews (id, user_id, destination_id, provider_id, rating, comment)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    review_id, current_user['id'], review_data.destination_id,
                    review_data.provider_id, review_data.rating, review_data.comment
                ))
                
                # 🔗 PHASE 6.2: Create blockchain review record if requested
                blockchain_created = False
                loyalty_bonus_awarded = 0
                
                if review_data.blockchain_verification and verified_booking:
                    try:
                        # Create blockchain review record
                        blockchain_review_id = str(uuid.uuid4())
                        # Get user wallet address for blockchain operations
                        await cur.execute("SELECT wallet_address FROM user_wallets WHERE user_id = %s", (current_user['id'],))
                        wallet_data = await cur.fetchone()
                        user_wallet = wallet_data['wallet_address'] if wallet_data else None
                        
                        await cur.execute("""
                            INSERT INTO blockchain_reviews (
                                id, review_id, user_id, user_wallet, booking_id, destination_id,
                                review_hash, contract_address, verification_status, is_authentic
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            blockchain_review_id, review_id, current_user['id'], user_wallet,
                            review_data.booking_id, review_data.destination_id or verified_booking['destination_id'],
                            f"review_hash_{review_id}", 
                            os.getenv('CONTRACT_ADDRESS_REVIEWS', 'pending'),
                            'pending', True
                        ))
                        
                        # 🔗 PHASE 6.2: Award bonus loyalty points for verified reviews
                        review_bonus = 25  # Bonus points for verified review
                        loyalty_bonus_awarded = review_bonus
                        
                        await cur.execute("""
                            INSERT INTO loyalty_points (id, user_id, points_balance, total_earned, total_redeemed)
                            VALUES (UUID(), %s, %s, %s, 0)
                            ON DUPLICATE KEY UPDATE 
                            points_balance = points_balance + VALUES(points_balance),
                            total_earned = total_earned + VALUES(total_earned)
                        """, (current_user['id'], review_bonus, review_bonus))
                        
                        # Log verified review bonus transaction
                        await cur.execute("""
                            INSERT INTO loyalty_transactions (id, user_id, transaction_type, points_amount, 
                                                           description, review_id, status)
                            VALUES (UUID(), %s, 'earned', %s, %s, %s, 'completed')
                        """, (current_user['id'], review_bonus, 
                             f"Verified review bonus - {verified_booking.get('destination_name', 'Unknown')}", 
                             review_id))
                        
                        blockchain_created = True
                        
                    except Exception as blockchain_error:
                        print(f"Failed to create blockchain review: {blockchain_error}")
                
                # Update average rating
                if review_data.destination_id:
                    await cur.execute("""
                        UPDATE destinations SET rating = (
                            SELECT AVG(rating) FROM reviews WHERE destination_id = %s
                        ) WHERE id = %s
                    """, (review_data.destination_id, review_data.destination_id))
                
                if review_data.provider_id:
                    await cur.execute("""
                        UPDATE providers SET rating = (
                            SELECT AVG(rating) FROM reviews WHERE provider_id = %s
                        ) WHERE id = %s
                    """, (review_data.provider_id, review_data.provider_id))
                
                response = {
                    "id": review_id,
                    "message": "Review created successfully",
                    "blockchain_verified": blockchain_created,
                    "loyalty_bonus_awarded": loyalty_bonus_awarded
                }
                
                if blockchain_created:
                    response["blockchain_message"] = f"Review verified on blockchain! Earned {loyalty_bonus_awarded} bonus points."
                
                return response
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Admin API
@api_router.get("/admin/stats")
async def get_admin_stats(current_user: dict = Depends(get_current_user)):
    """Get comprehensive admin dashboard statistics with time-series data"""
    try:
        if current_user['role'] != 'admin':
            raise HTTPException(status_code=403, detail="Admin access required")
        
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                # Basic statistics
                stats = {}
                
                await cur.execute("SELECT COUNT(*) as total FROM users")
                stats['total_users'] = (await cur.fetchone())['total']
                
                await cur.execute("SELECT COUNT(*) as total FROM destinations")
                stats['total_destinations'] = (await cur.fetchone())['total']
                
                await cur.execute("SELECT COUNT(*) as total FROM providers")
                stats['total_providers'] = (await cur.fetchone())['total']
                
                await cur.execute("SELECT COUNT(*) as total FROM providers WHERE is_active = 1")
                stats['active_providers'] = (await cur.fetchone())['total']
                
                await cur.execute("SELECT COUNT(*) as total FROM bookings")
                stats['total_bookings'] = (await cur.fetchone())['total']
                
                await cur.execute("SELECT SUM(total_price) as revenue FROM bookings WHERE status IN ('completed', 'paid')")
                revenue_result = await cur.fetchone()
                stats['total_revenue'] = float(revenue_result['revenue']) if revenue_result['revenue'] else 0
                
                # Monthly revenue for the last 6 months
                await cur.execute("""
                    SELECT 
                        DATE_FORMAT(created_at, '%Y-%m') as month,
                        SUM(total_price) as revenue,
                        COUNT(*) as bookings
                    FROM bookings 
                    WHERE created_at >= DATE_SUB(NOW(), INTERVAL 6 MONTH)
                        AND status IN ('completed', 'paid')
                    GROUP BY DATE_FORMAT(created_at, '%Y-%m')
                    ORDER BY month ASC
                """)
                monthly_revenue = await cur.fetchall()
                stats['monthly_revenue'] = monthly_revenue
                
                # User growth - last 6 months
                await cur.execute("""
                    SELECT 
                        DATE_FORMAT(created_at, '%Y-%m') as month,
                        COUNT(*) as new_users
                    FROM users 
                    WHERE created_at >= DATE_SUB(NOW(), INTERVAL 6 MONTH)
                    GROUP BY DATE_FORMAT(created_at, '%Y-%m')
                    ORDER BY month ASC
                """)
                user_growth = await cur.fetchall()
                stats['user_growth'] = user_growth
                
                # Booking growth - last 6 months
                await cur.execute("""
                    SELECT 
                        DATE_FORMAT(created_at, '%Y-%m') as month,
                        COUNT(*) as bookings
                    FROM bookings 
                    WHERE created_at >= DATE_SUB(NOW(), INTERVAL 6 MONTH)
                    GROUP BY DATE_FORMAT(created_at, '%Y-%m')
                    ORDER BY month ASC
                """)
                booking_growth = await cur.fetchall()
                stats['booking_growth'] = booking_growth
                
                # Recent bookings with details
                await cur.execute("""
                    SELECT 
                        b.id,
                        b.booking_date,
                        b.total_price,
                        b.status,
                        b.package_type,
                        b.guests,
                        u.name as customer_name,
                        u.email as customer_email,
                        d.name as destination_name,
                        p.name as provider_name,
                        b.created_at
                    FROM bookings b
                    JOIN users u ON b.user_id = u.id
                    LEFT JOIN destinations d ON b.destination_id = d.id
                    LEFT JOIN providers p ON b.provider_id = p.id
                    ORDER BY b.created_at DESC
                    LIMIT 10
                """)
                recent_bookings = await cur.fetchall()
                stats['recent_bookings'] = recent_bookings
                
                # Booking status distribution
                await cur.execute("""
                    SELECT status, COUNT(*) as count FROM bookings GROUP BY status
                """)
                booking_stats = await cur.fetchall()
                stats['booking_by_status'] = {stat['status']: stat['count'] for stat in booking_stats}
                
                # Revenue by destination
                await cur.execute("""
                    SELECT 
                        d.name as destination_name,
                        SUM(b.total_price) as revenue,
                        COUNT(b.id) as bookings
                    FROM bookings b
                    JOIN destinations d ON b.destination_id = d.id
                    WHERE b.status IN ('completed', 'paid')
                    GROUP BY d.id, d.name
                    ORDER BY revenue DESC
                    LIMIT 5
                """)
                revenue_by_destination = await cur.fetchall()
                stats['revenue_by_destination'] = revenue_by_destination
                
                # Average booking value
                await cur.execute("""
                    SELECT AVG(total_price) as avg_booking_value 
                    FROM bookings 
                    WHERE status IN ('completed', 'paid')
                """)
                avg_result = await cur.fetchone()
                stats['avg_booking_value'] = float(avg_result['avg_booking_value']) if avg_result['avg_booking_value'] else 0
                
                return stats
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Admin Destinations Management
class DestinationCreate(BaseModel):
    name: str
    location: str
    description: str
    image_url: str
    price: float
    category: str
    region: str
    highlights: List[str]

class DestinationUpdate(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = None
    image_url: Optional[str] = None
    price: Optional[float] = None
    category: Optional[str] = None
    region: Optional[str] = None
    highlights: Optional[List[str]] = None

@api_router.post("/admin/destinations")
async def create_destination(destination_data: DestinationCreate, current_user: dict = Depends(get_current_user)):
    """Create a new destination (Admin only)"""
    try:
        if current_user['role'] != 'admin':
            raise HTTPException(status_code=403, detail="Admin access required")
        
        destination_id = str(uuid.uuid4())
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute("""
                    INSERT INTO destinations (id, name, location, description, image_url, price, category, region, highlights, rating, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    destination_id,
                    destination_data.name,
                    destination_data.location,
                    destination_data.description,
                    destination_data.image_url,
                    destination_data.price,
                    destination_data.category,
                    destination_data.region,
                    json.dumps(destination_data.highlights),
                    0.0,  # Default rating
                    datetime.now()
                ))
                await conn.commit()
                
                return {
                    "id": destination_id,
                    "message": "Destination created successfully"
                }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.put("/admin/destinations/{destination_id}")
async def update_destination(destination_id: str, destination_data: DestinationUpdate, current_user: dict = Depends(get_current_user)):
    """Update a destination (Admin only)"""
    try:
        if current_user['role'] != 'admin':
            raise HTTPException(status_code=403, detail="Admin access required")
        
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                # Check if destination exists
                await cur.execute("SELECT id FROM destinations WHERE id = %s", (destination_id,))
                if not await cur.fetchone():
                    raise HTTPException(status_code=404, detail="Destination not found")
                
                # Build update query dynamically
                update_fields = []
                values = []
                
                if destination_data.name is not None:
                    update_fields.append("name = %s")
                    values.append(destination_data.name)
                if destination_data.location is not None:
                    update_fields.append("location = %s")
                    values.append(destination_data.location)
                if destination_data.description is not None:
                    update_fields.append("description = %s")
                    values.append(destination_data.description)
                if destination_data.image_url is not None:
                    update_fields.append("image_url = %s")
                    values.append(destination_data.image_url)
                if destination_data.price is not None:
                    update_fields.append("price = %s")
                    values.append(destination_data.price)
                if destination_data.category is not None:
                    update_fields.append("category = %s")
                    values.append(destination_data.category)
                if destination_data.region is not None:
                    update_fields.append("region = %s")
                    values.append(destination_data.region)
                if destination_data.highlights is not None:
                    update_fields.append("highlights = %s")
                    values.append(json.dumps(destination_data.highlights))
                
                if not update_fields:
                    raise HTTPException(status_code=400, detail="No fields to update")
                
                values.append(destination_id)
                query = f"UPDATE destinations SET {', '.join(update_fields)} WHERE id = %s"
                
                await cur.execute(query, values)
                await conn.commit()
                
                return {"message": "Destination updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.delete("/admin/destinations/{destination_id}")
async def delete_destination(destination_id: str, current_user: dict = Depends(get_current_user)):
    """Delete a destination (Admin only)"""
    try:
        if current_user['role'] != 'admin':
            raise HTTPException(status_code=403, detail="Admin access required")
        
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                # Check if destination exists
                await cur.execute("SELECT id FROM destinations WHERE id = %s", (destination_id,))
                if not await cur.fetchone():
                    raise HTTPException(status_code=404, detail="Destination not found")
                
                # Check if destination has active bookings
                await cur.execute("SELECT COUNT(*) as count FROM bookings WHERE destination_id = %s AND status IN ('pending', 'confirmed')", (destination_id,))
                result = await cur.fetchone()
                if result['count'] > 0:
                    raise HTTPException(status_code=400, detail="Cannot delete destination with active bookings")
                
                # Delete destination
                await cur.execute("DELETE FROM destinations WHERE id = %s", (destination_id,))
                await conn.commit()
                
                return {"message": "Destination deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Admin Provider/Service Management
@api_router.delete("/admin/providers/{provider_id}")
async def delete_provider(provider_id: str, current_user: dict = Depends(get_current_user)):
    """Delete a provider/service (Admin only)"""
    try:
        if current_user['role'] != 'admin':
            raise HTTPException(status_code=403, detail="Admin access required")
        
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                # Check if provider exists
                await cur.execute("SELECT id FROM providers WHERE id = %s", (provider_id,))
                if not await cur.fetchone():
                    raise HTTPException(status_code=404, detail="Provider not found")
                
                # Check if provider has active bookings
                await cur.execute("SELECT COUNT(*) as count FROM bookings WHERE provider_id = %s AND status IN ('pending', 'confirmed')", (provider_id,))
                result = await cur.fetchone()
                if result['count'] > 0:
                    raise HTTPException(status_code=400, detail="Cannot delete provider with active bookings")
                
                # Delete provider
                await cur.execute("DELETE FROM providers WHERE id = %s", (provider_id,))
                await conn.commit()
                
                return {"message": "Provider deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.put("/admin/providers/{provider_id}/status")
async def toggle_provider_status(provider_id: str, current_user: dict = Depends(get_current_user)):
    """Toggle provider active status (Admin only)"""
    try:
        if current_user['role'] != 'admin':
            raise HTTPException(status_code=403, detail="Admin access required")
        
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                # Check if provider exists and get current status
                await cur.execute("SELECT id, is_active FROM providers WHERE id = %s", (provider_id,))
                provider = await cur.fetchone()
                if not provider:
                    raise HTTPException(status_code=404, detail="Provider not found")
                
                # Toggle status
                new_status = not provider['is_active']
                await cur.execute("UPDATE providers SET is_active = %s WHERE id = %s", (new_status, provider_id))
                await conn.commit()
                
                return {
                    "message": f"Provider {'activated' if new_status else 'deactivated'} successfully",
                    "is_active": new_status
                }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/admin/users")
async def get_all_users(current_user: dict = Depends(get_current_user)):
    """Get all users for admin"""
    try:
        if current_user['role'] != 'admin':
            raise HTTPException(status_code=403, detail="Admin access required")
        
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute("SELECT id, name, email, role, phone, created_at FROM users ORDER BY created_at DESC")
                users = await cur.fetchall()
                return users
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.put("/admin/users/{user_id}/ban")
async def ban_user(user_id: str, current_user: dict = Depends(get_current_user)):
    """Ban a user (Admin only)"""
    try:
        if current_user['role'] != 'admin':
            raise HTTPException(status_code=403, detail="Admin access required")
        
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                # Check if user exists
                await cur.execute("SELECT id, name, role FROM users WHERE id = %s", (user_id,))
                user = await cur.fetchone()
                if not user:
                    raise HTTPException(status_code=404, detail="User not found")
                
                # Don't allow banning admin users
                if user['role'] == 'admin':
                    raise HTTPException(status_code=403, detail="Cannot ban admin users")
                
                # Update user status to banned (set is_active to 0)
                # First check if is_active column exists, if not add it
                await cur.execute("SHOW COLUMNS FROM users LIKE 'is_active'")
                column_exists = await cur.fetchone()
                
                if not column_exists:
                    await cur.execute("ALTER TABLE users ADD COLUMN is_active TINYINT(1) DEFAULT 1")
                    await conn.commit()
                
                await cur.execute("""
                    UPDATE users 
                    SET is_active = 0, updated_at = CURRENT_TIMESTAMP 
                    WHERE id = %s
                """, (user_id,))
                await conn.commit()
                
                return {"message": f"User {user['name']} has been banned"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.delete("/admin/users/{user_id}")
async def delete_user(user_id: str, current_user: dict = Depends(get_current_user)):
    """Delete a user (Admin only)"""
    try:
        if current_user['role'] != 'admin':
            raise HTTPException(status_code=403, detail="Admin access required")
        
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                # Check if user exists
                await cur.execute("SELECT id, name, role FROM users WHERE id = %s", (user_id,))
                user = await cur.fetchone()
                if not user:
                    raise HTTPException(status_code=404, detail="User not found")
                
                # Don't allow deleting admin users
                if user['role'] == 'admin':
                    raise HTTPException(status_code=403, detail="Cannot delete admin users")
                
                # Delete user (this will cascade to related records due to foreign keys)
                await cur.execute("DELETE FROM users WHERE id = %s", (user_id,))
                await conn.commit()
                
                return {"message": f"User {user['name']} has been deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/admin/bookings")
async def get_all_bookings(current_user: dict = Depends(get_current_user)):
    """Get all bookings for admin"""
    try:
        if current_user['role'] != 'admin':
            raise HTTPException(status_code=403, detail="Admin access required")
        
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute("SELECT * FROM bookings ORDER BY created_at DESC")
                bookings = await cur.fetchall()
                return bookings
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Payment API - Import payment models and service
from models.payment_models import (
    PaymentCreate, PaymentVerification, PaymentStatusUpdate, 
    UPIQRRequest, PaymentResponse, AdminPaymentApproval, PaymentStatus
)
from services.payment_service import PaymentService

# Initialize payment service
payment_service = PaymentService()

# Payment Management API (COMMENTED OUT TO FIX MODULE ISSUE)
@api_router.post("/payments/create")
async def create_payment(
    payment_data: PaymentCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a new payment request for a booking"""
    try:
        pool = await get_db()
        payment_id = str(uuid.uuid4())
        
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                # Verify booking exists and belongs to user
                await cur.execute("""
                    SELECT id, total_price, status, booking_full_name, booking_phone 
                    FROM bookings 
                    WHERE id = %s AND user_id = %s
                """, (payment_data.booking_id, current_user['id']))
                
                booking = await cur.fetchone()
                if not booking:
                    raise HTTPException(status_code=404, detail="Booking not found")
                
                # Check if payment already exists for this booking
                await cur.execute("SELECT id FROM payments WHERE booking_id = %s", (payment_data.booking_id,))
                existing_payment = await cur.fetchone()
                if existing_payment:
                    raise HTTPException(status_code=400, detail="Payment already exists for this booking")
                
                # Generate payment reference and UPI QR code
                transaction_ref = payment_service.generate_payment_reference()
                qr_data = payment_service.generate_upi_qr_code(
                    amount=payment_data.amount,
                    transaction_ref=transaction_ref,
                    customer_name=booking['booking_full_name'] or current_user['name']
                )
                
                # Create payment record
                qr_data_for_db = qr_data.copy()
                qr_data_for_db['expires_at'] = qr_data['expires_at'].isoformat() if qr_data['expires_at'] else None
                
                await cur.execute("""
                    INSERT INTO payments (
                        id, booking_id, amount, status, payment_method, 
                        transaction_reference, upi_id, qr_code_data, expires_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    payment_id, payment_data.booking_id, payment_data.amount,
                    PaymentStatus.PENDING, payment_data.payment_method,
                    transaction_ref, payment_data.upi_id or payment_service.upi_id,
                    json.dumps(qr_data_for_db), qr_data['expires_at']
                ))
                
                # Update booking status to payment_required
                await cur.execute("""
                    UPDATE bookings 
                    SET status = 'payment_required', payment_status = 'required', 
                        payment_amount = %s, payment_deadline = %s
                    WHERE id = %s
                """, (payment_data.amount, qr_data['expires_at'], payment_data.booking_id))
                
                return {
                    "id": payment_id,
                    "booking_id": payment_data.booking_id,
                    "amount": payment_data.amount,
                    "status": PaymentStatus.PENDING,
                    "payment_method": payment_data.payment_method,
                    "upi_qr_code": qr_data['qr_code_base64'],
                    "upi_payment_url": qr_data['upi_url'],
                    "transaction_reference": transaction_ref,
                    "created_at": datetime.utcnow().isoformat(),
                    "expires_at": qr_data['expires_at'].isoformat() if qr_data['expires_at'] else None
                }
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create payment: {str(e)}")

@api_router.post("/payments/generate-qr")
async def generate_payment_qr(
    qr_request: UPIQRRequest,
    current_user: dict = Depends(get_current_user)
):
    """Generate UPI QR code for payment"""
    try:
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                # Verify booking exists and belongs to user
                await cur.execute("""
                    SELECT id, total_price, status 
                    FROM bookings 
                    WHERE id = %s AND user_id = %s
                """, (qr_request.booking_id, current_user['id']))
                
                booking = await cur.fetchone()
                if not booking:
                    raise HTTPException(status_code=404, detail="Booking not found")
                
                # Generate payment reference and QR code
                transaction_ref = payment_service.generate_payment_reference()
                qr_data = payment_service.generate_upi_qr_code(
                    amount=qr_request.amount,
                    transaction_ref=transaction_ref,
                    customer_name=qr_request.customer_name
                )
                
                # Add payment instructions
                instructions = payment_service.get_payment_instructions()
                
                return {
                    **qr_data,
                    "instructions": instructions,
                    "booking_id": qr_request.booking_id,
                    "customer_name": qr_request.customer_name,
                    "customer_phone": qr_request.customer_phone
                }
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate QR code: {str(e)}")

@api_router.post("/payments/verify")
async def verify_payment(
    verification_data: PaymentVerification,
    current_user: dict = Depends(get_current_user)
):
    """Submit payment verification with transaction ID"""
    try:
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                # Get payment details
                await cur.execute("""
                    SELECT p.*, b.user_id, b.booking_full_name 
                    FROM payments p 
                    JOIN bookings b ON p.booking_id = b.id 
                    WHERE p.id = %s
                """, (verification_data.payment_id,))
                
                payment = await cur.fetchone()
                if not payment:
                    raise HTTPException(status_code=404, detail="Payment not found")
                
                # Verify user owns the booking
                if payment['user_id'] != current_user['id']:
                    raise HTTPException(status_code=403, detail="Unauthorized access")
                
                # Validate transaction ID format
                if not payment_service.validate_transaction_id(verification_data.transaction_id):
                    raise HTTPException(status_code=400, detail="Invalid transaction ID format")
                
                # Check if payment is expired
                if payment_service.is_payment_expired(payment['created_at']):
                    raise HTTPException(status_code=400, detail="Payment request has expired")
                
                # Update payment with verification details
                await cur.execute("""
                    UPDATE payments 
                    SET upi_transaction_id = %s, status = 'verification_required',
                        customer_note = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (
                    verification_data.transaction_id,
                    verification_data.customer_note,
                    verification_data.payment_id
                ))
                
                # Update booking status
                await cur.execute("""
                    UPDATE bookings 
                    SET status = 'payment_pending', payment_status = 'pending'
                    WHERE id = %s
                """, (payment['booking_id'],))
                
                # Log the verification attempt
                await cur.execute("""
                    INSERT INTO payment_logs (id, payment_id, action, old_status, new_status, user_id, user_role, details)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    str(uuid.uuid4()), verification_data.payment_id, 'customer_verification',
                    payment['status'], 'verification_required', current_user['id'], 'customer',
                    json.dumps({
                        "transaction_id": verification_data.transaction_id,
                        "amount": verification_data.amount,
                        "customer_note": verification_data.customer_note
                    })
                ))
                
                return {
                    "message": "Payment verification submitted successfully",
                    "payment_id": verification_data.payment_id,
                    "transaction_id": verification_data.transaction_id,
                    "status": "verification_required",
                    "next_steps": "Our team will verify your payment within 24 hours. You will receive a confirmation once approved."
                }
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to verify payment: {str(e)}")

@api_router.get("/payments/{payment_id}")
async def get_payment_details(
    payment_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get payment details"""
    try:
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute("""
                    SELECT p.*, b.user_id, b.booking_full_name, b.total_price as booking_amount
                    FROM payments p 
                    JOIN bookings b ON p.booking_id = b.id 
                    WHERE p.id = %s
                """, (payment_id,))
                
                payment = await cur.fetchone()
                if not payment:
                    raise HTTPException(status_code=404, detail="Payment not found")
                
                # Check access permissions
                if payment['user_id'] != current_user['id'] and current_user['role'] != 'admin':
                    raise HTTPException(status_code=403, detail="Unauthorized access")
                
                # Parse QR code data if exists
                if payment['qr_code_data']:
                    payment['qr_code_data'] = json.loads(payment['qr_code_data'])
                
                return payment
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/payments/booking/{booking_id}")
async def get_payment_by_booking(
    booking_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get payment details for a booking"""
    try:
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                # Verify booking belongs to user
                await cur.execute("""
                    SELECT user_id FROM bookings WHERE id = %s
                """, (booking_id,))
                
                booking = await cur.fetchone()
                if not booking:
                    raise HTTPException(status_code=404, detail="Booking not found")
                
                if booking['user_id'] != current_user['id'] and current_user['role'] != 'admin':
                    raise HTTPException(status_code=403, detail="Unauthorized access")
                
                # Get payment details
                await cur.execute("""
                    SELECT * FROM payments WHERE booking_id = %s ORDER BY created_at DESC
                """, (booking_id,))
                
                payments = await cur.fetchall()
                
                # Parse QR code data for each payment
                for payment in payments:
                    if payment['qr_code_data']:
                        payment['qr_code_data'] = json.loads(payment['qr_code_data'])
                
                return payments
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Admin Payment Management
@api_router.get("/admin/payments")
async def get_all_payments(
    status: Optional[str] = None,
    limit: int = 50,
    current_user: dict = Depends(get_current_user)
):
    """Get all payments for admin review"""
    try:
        if current_user['role'] != 'admin':
            raise HTTPException(status_code=403, detail="Admin access required")
        
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                query = """
                    SELECT p.*, b.booking_full_name, b.booking_email, b.booking_phone, 
                           b.package_name, u.name as user_name, u.email as user_email
                    FROM payments p 
                    JOIN bookings b ON p.booking_id = b.id 
                    JOIN users u ON b.user_id = u.id
                """
                params = []
                
                if status:
                    query += " WHERE p.status = %s"
                    params.append(status)
                
                query += " ORDER BY p.created_at DESC LIMIT %s"
                params.append(limit)
                
                await cur.execute(query, params)
                payments = await cur.fetchall()
                
                return payments
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/admin/payments/approve")
async def approve_payment(
    approval_data: AdminPaymentApproval,
    current_user: dict = Depends(get_current_user)
):
    """Approve or reject payment verification"""
    try:
        if current_user['role'] != 'admin':
            raise HTTPException(status_code=403, detail="Admin access required")
        
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                # Get payment details
                await cur.execute("""
                    SELECT p.*, b.booking_id 
                    FROM payments p 
                    JOIN bookings b ON p.booking_id = b.id 
                    WHERE p.id = %s
                """, (approval_data.payment_id,))
                
                payment = await cur.fetchone()
                if not payment:
                    raise HTTPException(status_code=404, detail="Payment not found")
                
                old_status = payment['status']
                
                if approval_data.action == "approve":
                    new_status = PaymentStatus.COMPLETED
                    booking_status = "paid"
                    booking_payment_status = "completed"
                    
                    # Update payment
                    await cur.execute("""
                        UPDATE payments 
                        SET status = %s, admin_note = %s, verified_amount = %s,
                            verified_by = %s, verified_at = CURRENT_TIMESTAMP,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                    """, (
                        new_status, approval_data.admin_note,
                        approval_data.verified_amount or payment['amount'],
                        current_user['id'], approval_data.payment_id
                    ))
                    
                    message = "Payment approved and booking confirmed"
                    
                elif approval_data.action == "reject":
                    new_status = PaymentStatus.FAILED
                    booking_status = "payment_required"
                    booking_payment_status = "failed"
                    
                    # Update payment
                    await cur.execute("""
                        UPDATE payments 
                        SET status = %s, admin_note = %s, verified_by = %s,
                            verified_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                    """, (
                        new_status, approval_data.admin_note,
                        current_user['id'], approval_data.payment_id
                    ))
                    
                    message = "Payment rejected. Customer will need to retry payment."
                
                # Update booking status
                await cur.execute("""
                    UPDATE bookings 
                    SET status = %s, payment_status = %s
                    WHERE id = %s
                """, (booking_status, booking_payment_status, payment['booking_id']))
                
                # Log admin action
                await cur.execute("""
                    INSERT INTO payment_logs (id, payment_id, action, old_status, new_status, user_id, user_role, details)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    str(uuid.uuid4()), approval_data.payment_id, f'admin_{approval_data.action}',
                    old_status, new_status, current_user['id'], 'admin',
                    json.dumps({
                        "action": approval_data.action,
                        "admin_note": approval_data.admin_note,
                        "verified_amount": approval_data.verified_amount
                    })
                ))
                
                return {
                    "message": message,
                    "payment_id": approval_data.payment_id,
                    "action": approval_data.action,
                    "new_status": new_status,
                    "admin_note": approval_data.admin_note
                }
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process payment approval: {str(e)}")

@api_router.get("/admin/payments/pending")
async def get_pending_payments(current_user: dict = Depends(get_current_user)):
    """Get payments pending admin approval"""
    try:
        if current_user['role'] != 'admin':
            raise HTTPException(status_code=403, detail="Admin access required")
        
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute("""
                    SELECT p.*, b.booking_full_name, b.booking_email, b.booking_phone,
                           b.package_name, b.total_price as booking_amount, u.name as user_name
                    FROM payments p 
                    JOIN bookings b ON p.booking_id = b.id 
                    JOIN users u ON b.user_id = u.id
                    WHERE p.status = 'verification_required'
                    ORDER BY p.created_at ASC
                """)
                
                pending_payments = await cur.fetchall()
                return pending_payments
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===========================================
# BLOCKCHAIN API ENDPOINTS
# ===========================================

# Import blockchain models
from models import (
    WalletConnect, WalletResponse, CertificateCreate, Certificate, 
    LoyaltyPointsBalance, LoyaltyTransaction, BlockchainBooking, 
    BlockchainReview, BlockchainStatus, GasCostEstimate
)

@api_router.get("/blockchain/status", response_model=BlockchainStatus)
async def get_blockchain_status():
    """Get blockchain network status and configuration"""
    try:
        network_info = blockchain_service.get_network_info()
        # Map the fields correctly to match BlockchainStatus model
        contracts_dict = network_info.get("contracts") or {}
        
        return BlockchainStatus(
            network=network_info.get("network", "unknown"),
            connected=network_info.get("connected", False),
            block_number=network_info.get("latest_block"),
            gas_price=str(network_info.get("gas_price_gwei")) if network_info.get("gas_price_gwei") is not None else None,
            contract_addresses=contracts_dict
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/blockchain/wallet/connect", response_model=WalletResponse)
async def connect_wallet(wallet_data: WalletConnect, current_user: dict = Depends(get_current_user)):
    """Connect user's Web3 wallet"""
    try:
        pool = await get_db()
        async with pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cur:
                    # Check if wallet is already connected to another user
                    await cur.execute(
                        "SELECT user_id FROM user_wallets WHERE wallet_address = %s",
                        (wallet_data.wallet_address,)
                    )
                    existing_wallet = await cur.fetchone()
                    
                    if existing_wallet and existing_wallet['user_id'] != current_user['id']:
                        raise HTTPException(
                            status_code=400, 
                            detail="This wallet is already connected to another account"
                        )
                    
                    # Update or insert wallet connection
                    wallet_id = str(uuid.uuid4())
                    await cur.execute("""
                        INSERT INTO user_wallets (id, user_id, wallet_address, is_verified, created_at)
                        VALUES (%s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE 
                            wallet_address = VALUES(wallet_address),
                            is_verified = VALUES(is_verified),
                            updated_at = CURRENT_TIMESTAMP
                    """, (wallet_id, current_user['id'], wallet_data.wallet_address, True, datetime.now()))
                    
                    # Update user's wallet_address and wallet_connected status
                    await cur.execute("""
                        UPDATE users 
                        SET wallet_address = %s, wallet_connected = %s 
                        WHERE id = %s
                    """, (wallet_data.wallet_address, True, current_user['id']))
                    
                    return WalletResponse(
                        user_id=current_user['id'],
                        wallet_address=wallet_data.wallet_address,
                        is_verified=True,
                        connected=True
                    )
                    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/blockchain/wallet/status", response_model=WalletResponse)
async def get_wallet_status(current_user: dict = Depends(get_current_user)):
    """Get user's wallet connection status"""
    try:
        pool = await get_db()
        async with pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cur:
                    await cur.execute(
                        "SELECT wallet_address, wallet_connected FROM users WHERE id = %s",
                        (current_user['id'],)
                    )
                    user_data = await cur.fetchone()
                    
                    if not user_data or not user_data['wallet_address']:
                        return WalletResponse(
                            user_id=current_user['id'],
                            wallet_address="",
                            is_verified=False,
                            connected=False
                        )
                    
                    return WalletResponse(
                        user_id=current_user['id'],
                        wallet_address=user_data['wallet_address'],
                        is_verified=user_data['wallet_connected'],
                        connected=user_data['wallet_connected']
                    )
                    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/blockchain/certificates/mint")
async def mint_certificate(cert_data: CertificateCreate, current_user: dict = Depends(get_current_user)):
    """Mint a certificate NFT for completed tour"""
    try:
        # Get user's wallet address
        pool = await get_db()
        async with pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cur:
                    await cur.execute(
                        "SELECT wallet_address FROM users WHERE id = %s",
                        (current_user['id'],)
                    )
                    user_data = await cur.fetchone()
                    
                    if not user_data or not user_data['wallet_address']:
                        raise HTTPException(
                            status_code=400,
                            detail="Please connect your wallet first"
                        )
                    
                    # Check if booking exists and is completed
                    await cur.execute(
                        "SELECT * FROM bookings WHERE id = %s AND user_id = %s AND status = 'completed'",
                        (cert_data.booking_id, current_user['id'])
                    )
                    booking = await cur.fetchone()
                    
                    if not booking:
                        raise HTTPException(
                            status_code=404,
                            detail="Completed booking not found"
                        )
                    
                    # Check if certificate already exists
                    await cur.execute(
                        "SELECT id FROM certificates WHERE booking_id = %s",
                        (cert_data.booking_id,)
                    )
                    existing_cert = await cur.fetchone()
                    
                    if existing_cert:
                        raise HTTPException(
                            status_code=400,
                            detail="Certificate already exists for this booking"
                        )
                    
                    # Mint certificate on blockchain
                    mint_result = await blockchain_service.mint_certificate(
                        user_wallet=user_data['wallet_address'],
                        booking_id=cert_data.booking_id,
                        destination_name=cert_data.destination_name,
                        certificate_type=cert_data.certificate_type
                    )
                    
                    if not mint_result['success']:
                        raise HTTPException(
                            status_code=500,
                            detail=f"Failed to mint certificate: {mint_result['error']}"
                        )
                    
                    # Save certificate to database
                    cert_id = str(uuid.uuid4())
                    await cur.execute("""
                        INSERT INTO certificates (
                            id, user_id, booking_id, certificate_type, nft_token_id,
                            contract_address, transaction_hash, blockchain_network,
                            metadata_url, certificate_title, certificate_description,
                            destination_name, completion_date, is_minted
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        cert_id, current_user['id'], cert_data.booking_id, cert_data.certificate_type,
                        mint_result['token_id'], blockchain_service.contracts['certificates'],
                        mint_result['transaction_hash'], blockchain_service.network,
                        mint_result['metadata_url'], f"Tourism Certificate - {cert_data.destination_name}",
                        f"Certificate of {cert_data.certificate_type} for visiting {cert_data.destination_name}",
                        cert_data.destination_name, datetime.now().date(), True
                    ))
                    
                    # Update booking certificate status
                    await cur.execute(
                        "UPDATE bookings SET certificate_issued = %s WHERE id = %s",
                        (True, cert_data.booking_id)
                    )
                    
                    return {
                        "success": True,
                        "certificate_id": cert_id,
                        "transaction_hash": mint_result['transaction_hash'],
                        "token_id": mint_result['token_id']
                    }
                    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/blockchain/certificates/my", response_model=List[Certificate])
async def get_my_certificates(current_user: dict = Depends(get_current_user)):
    """Get user's certificates"""
    try:
        async with get_db() as pool:
            async with pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cur:
                    await cur.execute(
                        "SELECT * FROM certificates WHERE user_id = %s ORDER BY issued_at DESC",
                        (current_user['id'],)
                    )
                    certificates = await cur.fetchall()
                    return certificates
                    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/blockchain/loyalty/balance", response_model=LoyaltyPointsBalance)
async def get_loyalty_balance(current_user: dict = Depends(get_current_user)):
    """Get user's loyalty points balance"""
    try:
        async with get_db() as pool:
            async with pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cur:
                    await cur.execute(
                        "SELECT * FROM loyalty_points WHERE user_id = %s",
                        (current_user['id'],)
                    )
                    loyalty_data = await cur.fetchone()
                    
                    if not loyalty_data:
                        # Create initial loyalty record
                        loyalty_id = str(uuid.uuid4())
                        user_wallet = ""
                        
                        # Get user wallet address
                        await cur.execute(
                            "SELECT wallet_address FROM users WHERE id = %s",
                            (current_user['id'],)
                        )
                        user_data = await cur.fetchone()
                        if user_data and user_data['wallet_address']:
                            user_wallet = user_data['wallet_address']
                        
                        await cur.execute("""
                            INSERT INTO loyalty_points (
                                id, user_id, wallet_address, points_balance, 
                                total_earned, total_redeemed, contract_address
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """, (
                            loyalty_id, current_user['id'], user_wallet, 0.0, 0.0, 0.0,
                            blockchain_service.contracts['loyalty']
                        ))
                        
                        return LoyaltyPointsBalance(
                            user_id=current_user['id'],
                            wallet_address=user_wallet,
                            points_balance=0.0,
                            total_earned=0.0,
                            total_redeemed=0.0
                        )
                    
                    return LoyaltyPointsBalance(**loyalty_data)
                    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/blockchain/loyalty/award")
async def award_loyalty_points(
    booking_id: str, 
    points: int, 
    current_user: dict = Depends(get_current_user)
):
    """Award loyalty points for a booking (internal use)"""
    try:
        # This endpoint would typically be called internally after booking completion
        async with get_db() as pool:
            async with pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cur:
                    # Get user wallet
                    await cur.execute(
                        "SELECT wallet_address FROM users WHERE id = %s",
                        (current_user['id'],)
                    )
                    user_data = await cur.fetchone()
                    
                    if not user_data or not user_data['wallet_address']:
                        raise HTTPException(
                            status_code=400,
                            detail="Please connect your wallet first"
                        )
                    
                    # Award points on blockchain
                    award_result = await blockchain_service.award_loyalty_points(
                        user_wallet=user_data['wallet_address'],
                        points=points,
                        booking_id=booking_id
                    )
                    
                    if not award_result['success']:
                        raise HTTPException(
                            status_code=500,
                            detail=f"Failed to award points: {award_result['error']}"
                        )
                    
                    # Update database
                    await cur.execute("""
                        UPDATE loyalty_points 
                        SET points_balance = points_balance + %s, 
                            total_earned = total_earned + %s,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE user_id = %s
                    """, (points, points, current_user['id']))
                    
                    # Log transaction
                    tx_id = str(uuid.uuid4())
                    await cur.execute("""
                        INSERT INTO loyalty_transactions (
                            id, user_id, transaction_type, points_amount, 
                            booking_id, transaction_hash, description
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        tx_id, current_user['id'], 'earned', points,
                        booking_id, award_result['transaction_hash'],
                        f"Points awarded for booking {booking_id}"
                    ))
                    
                    return {
                        "success": True,
                        "points_awarded": points,
                        "transaction_hash": award_result['transaction_hash']
                    }
                    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 🔗 PHASE 6.3: Loyalty Points Redemption System
@api_router.post("/loyalty/redeem")
async def redeem_loyalty_points(
    redeem_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Redeem loyalty points for booking discount"""
    try:
        points_to_redeem = redeem_data.get('points', 0)
        booking_id = redeem_data.get('booking_id')
        
        if not points_to_redeem or points_to_redeem <= 0:
            raise HTTPException(status_code=400, detail="Invalid points amount")
        
        if not booking_id:
            raise HTTPException(status_code=400, detail="Booking ID is required")
        
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                # Get user's loyalty points balance
                await cur.execute(
                    "SELECT * FROM loyalty_points WHERE user_id = %s",
                    (current_user['id'],)
                )
                loyalty_data = await cur.fetchone()
                
                if not loyalty_data:
                    raise HTTPException(status_code=404, detail="No loyalty points found")
                
                if loyalty_data['points_balance'] < points_to_redeem:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Insufficient points. Available: {loyalty_data['points_balance']}, Requested: {points_to_redeem}"
                    )
                
                # Calculate discount (100 points = ₹10)
                discount_amount = points_to_redeem / 100 * 10
                
                # Get booking details
                await cur.execute(
                    "SELECT * FROM bookings WHERE id = %s AND user_id = %s AND status = 'pending'",
                    (booking_id, current_user['id'])
                )
                booking = await cur.fetchone()
                
                if not booking:
                    raise HTTPException(status_code=404, detail="Pending booking not found")
                
                # Check if discount would exceed total price
                max_discount = booking['total_price'] * 0.5  # Max 50% discount
                final_discount = min(discount_amount, max_discount)
                final_points_used = int(final_discount * 100 / 10)  # Recalculate points needed for final discount
                
                # Update loyalty points balance
                await cur.execute("""
                    UPDATE loyalty_points 
                    SET points_balance = points_balance - %s, 
                        total_redeemed = total_redeemed + %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = %s
                """, (final_points_used, final_points_used, current_user['id']))
                
                # Update booking price
                new_total_price = booking['total_price'] - final_discount
                await cur.execute("""
                    UPDATE bookings 
                    SET total_price = %s
                    WHERE id = %s
                """, (new_total_price, booking_id))
                
                # Log redemption transaction
                tx_id = str(uuid.uuid4())
                await cur.execute("""
                    INSERT INTO loyalty_transactions (
                        id, user_id, transaction_type, points_amount, 
                        booking_id, description, status
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    tx_id, current_user['id'], 'redeemed', final_points_used,
                    booking_id, f"Points redeemed for booking discount - ₹{final_discount:.2f}", 'completed'
                ))
                
                # Get updated balance
                await cur.execute(
                    "SELECT points_balance FROM loyalty_points WHERE user_id = %s",
                    (current_user['id'],)
                )
                updated_balance = await cur.fetchone()
                
                return {
                    "success": True,
                    "points_redeemed": final_points_used,
                    "discount_applied": final_discount,
                    "new_total_price": new_total_price,
                    "remaining_points": updated_balance['points_balance'] if updated_balance else 0,
                    "message": f"Successfully redeemed {final_points_used} points for ₹{final_discount:.2f} discount"
                }
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/loyalty/transactions")
async def get_loyalty_transactions(current_user: dict = Depends(get_current_user)):
    """Get user's loyalty transaction history"""
    try:
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute("""
                    SELECT * FROM loyalty_transactions 
                    WHERE user_id = %s 
                    ORDER BY created_at DESC 
                    LIMIT 50
                """, (current_user['id'],))
                transactions = await cur.fetchall()
                
                return {"transactions": transactions}
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/blockchain/bookings/verify/{booking_id}")
async def verify_booking_blockchain(booking_id: str, current_user: dict = Depends(get_current_user)):
    """Verify booking on blockchain"""
    try:
        async with get_db() as pool:
            async with pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cur:
                    # Get booking details
                    await cur.execute(
                        "SELECT * FROM bookings WHERE id = %s AND user_id = %s",
                        (booking_id, current_user['id'])
                    )
                    booking = await cur.fetchone()
                    
                    if not booking:
                        raise HTTPException(status_code=404, detail="Booking not found")
                    
                    # Get user wallet
                    await cur.execute(
                        "SELECT wallet_address FROM users WHERE id = %s",
                        (current_user['id'],)
                    )
                    user_data = await cur.fetchone()
                    
                    if not user_data or not user_data['wallet_address']:
                        raise HTTPException(
                            status_code=400,
                            detail="Please connect your wallet first"
                        )
                    
                    # Verify booking on blockchain
                    verify_result = await blockchain_service.verify_booking_on_blockchain(
                        booking_id=booking_id,
                        booking_data=booking,
                        user_wallet=user_data['wallet_address']
                    )
                    
                    if not verify_result['success']:
                        raise HTTPException(
                            status_code=500,
                            detail=f"Failed to verify booking: {verify_result['error']}"
                        )
                    
                    # Save blockchain booking record
                    blockchain_booking_id = str(uuid.uuid4())
                    await cur.execute("""
                        INSERT INTO blockchain_bookings (
                            id, booking_id, user_wallet, booking_hash,
                            contract_address, transaction_hash, verification_status,
                            blockchain_network
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        blockchain_booking_id, booking_id, user_data['wallet_address'],
                        verify_result['booking_hash'], blockchain_service.contracts['booking'],
                        verify_result['transaction_hash'], 'verified', blockchain_service.network
                    ))
                    
                    # Update booking blockchain status
                    await cur.execute("""
                        UPDATE bookings 
                        SET blockchain_verified = %s, blockchain_hash = %s, 
                            smart_contract_address = %s, certificate_eligible = %s
                        WHERE id = %s
                    """, (
                        True, verify_result['booking_hash'], 
                        blockchain_service.contracts['booking'], True, booking_id
                    ))
                    
                    return {
                        "success": True,
                        "booking_hash": verify_result['booking_hash'],
                        "transaction_hash": verify_result['transaction_hash']
                    }
                    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/blockchain/gas/estimate/{operation}")
async def estimate_gas_cost(operation: str):
    """Estimate gas cost for blockchain operations"""
    try:
        if operation not in ['mint_certificate', 'award_points', 'redeem_points', 'verify_booking', 'verify_review']:
            raise HTTPException(status_code=400, detail="Invalid operation")
        
        estimate = blockchain_service.estimate_gas_cost(operation)
        return estimate
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Additional blockchain endpoints that frontend expects
@api_router.get("/blockchain/certificates")
async def get_certificates(current_user: dict = Depends(get_current_user)):
    """Get user's certificates (alias for /blockchain/certificates/my)"""
    try:
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    "SELECT * FROM certificates WHERE user_id = %s ORDER BY created_at DESC",
                    (current_user['id'],)
                )
                certificates = await cur.fetchall()
                return certificates
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/blockchain/loyalty/points")  
async def get_loyalty_points(current_user: dict = Depends(get_current_user)):
    """Get user's loyalty points (alias for /blockchain/loyalty/balance)"""
    try:
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    "SELECT * FROM loyalty_points WHERE user_id = %s",
                    (current_user['id'],)
                )
                loyalty_data = await cur.fetchone()
                
                if not loyalty_data:
                    # Create initial loyalty record
                    loyalty_id = str(uuid.uuid4())
                    user_wallet = ""
                    
                    # Get user wallet address
                    await cur.execute(
                        "SELECT wallet_address FROM users WHERE id = %s",
                        (current_user['id'],)
                    )
                    user_data = await cur.fetchone()
                    if user_data and user_data['wallet_address']:
                        user_wallet = user_data['wallet_address']
                    
                    await cur.execute("""
                        INSERT INTO loyalty_points (
                            id, user_id, wallet_address, points_balance, 
                            total_earned, total_redeemed, contract_address
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        loyalty_id, current_user['id'], user_wallet, 0.0, 0.0, 0.0,
                        os.environ.get('CONTRACT_ADDRESS_LOYALTY', '')
                    ))
                    
                    return {
                        "user_id": current_user['id'],
                        "wallet_address": user_wallet,
                        "points_balance": 0.0,
                        "total_earned": 0.0,
                        "total_redeemed": 0.0
                    }
                
                return loyalty_data
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/blockchain/bookings/status/{booking_id}")
async def get_booking_blockchain_status(booking_id: str, current_user: dict = Depends(get_current_user)):
    """Get blockchain verification status for a booking"""
    try:
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                # Check if booking exists and belongs to user
                await cur.execute(
                    "SELECT * FROM bookings WHERE id = %s AND user_id = %s",
                    (booking_id, current_user['id'])
                )
                booking = await cur.fetchone()
                
                if not booking:
                    raise HTTPException(status_code=404, detail="Booking not found")
                
                # Check blockchain verification status
                await cur.execute(
                    "SELECT * FROM blockchain_bookings WHERE booking_id = %s",
                    (booking_id,)
                )
                blockchain_booking = await cur.fetchone()
                
                if not blockchain_booking:
                    return {
                        "booking_id": booking_id,
                        "verification_status": "not_requested",
                        "blockchain_verified": False,
                        "booking_hash": None,
                        "transaction_hash": None,
                        "contract_address": None
                    }
                
                return {
                    "booking_id": booking_id,
                    "verification_status": blockchain_booking['verification_status'],
                    "blockchain_verified": True,
                    "booking_hash": blockchain_booking['booking_hash'],
                    "transaction_hash": blockchain_booking['transaction_hash'],
                    "contract_address": blockchain_booking['contract_address']
                }
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/blockchain/loyalty/redeem")
async def redeem_blockchain_points(
    redeem_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Redeem loyalty points for booking discount (blockchain version)"""
    try:
        points_to_redeem = redeem_data.get('points', 0)
        booking_id = redeem_data.get('booking_id')
        
        if not points_to_redeem or points_to_redeem <= 0:
            raise HTTPException(status_code=400, detail="Invalid points amount")
        
        if not booking_id:
            raise HTTPException(status_code=400, detail="Booking ID is required")
        
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                # Get user's loyalty points balance
                await cur.execute(
                    "SELECT * FROM loyalty_points WHERE user_id = %s",
                    (current_user['id'],)
                )
                loyalty_data = await cur.fetchone()
                
                if not loyalty_data:
                    raise HTTPException(status_code=404, detail="No loyalty points found")
                
                if loyalty_data['points_balance'] < points_to_redeem:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Insufficient points. Available: {loyalty_data['points_balance']}, Requested: {points_to_redeem}"
                    )
                
                # Calculate discount (100 points = ₹10)
                discount_amount = points_to_redeem / 100 * 10
                
                # Get booking details
                await cur.execute(
                    "SELECT * FROM bookings WHERE id = %s AND user_id = %s AND status = 'pending'",
                    (booking_id, current_user['id'])
                )
                booking = await cur.fetchone()
                
                if not booking:
                    raise HTTPException(status_code=404, detail="Pending booking not found")
                
                # Check if discount would exceed total price
                max_discount = booking['total_price'] * 0.5  # Max 50% discount
                final_discount = min(discount_amount, max_discount)
                final_points_used = int(final_discount * 100 / 10)  # Recalculate points needed for final discount
                
                # Update loyalty points balance
                await cur.execute("""
                    UPDATE loyalty_points 
                    SET points_balance = points_balance - %s, 
                        total_redeemed = total_redeemed + %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = %s
                """, (final_points_used, final_points_used, current_user['id']))
                
                # Update booking price
                new_total_price = booking['total_price'] - final_discount
                await cur.execute("""
                    UPDATE bookings 
                    SET total_price = %s
                    WHERE id = %s
                """, (new_total_price, booking_id))
                
                # Log redemption transaction
                tx_id = str(uuid.uuid4())
                await cur.execute("""
                    INSERT INTO loyalty_transactions (
                        id, user_id, transaction_type, points_amount, 
                        booking_id, description, status
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    tx_id, current_user['id'], 'redeemed', final_points_used,
                    booking_id, f"Points redeemed for booking discount - ₹{final_discount:.2f}", 'completed'
                ))
                
                # Get updated balance
                await cur.execute(
                    "SELECT points_balance FROM loyalty_points WHERE user_id = %s",
                    (current_user['id'],)
                )
                updated_balance = await cur.fetchone()
                
                return {
                    "success": True,
                    "points_redeemed": final_points_used,
                    "discount_applied": final_discount,
                    "new_total_price": new_total_price,
                    "remaining_points": updated_balance['points_balance'] if updated_balance else 0,
                    "message": f"Successfully redeemed {final_points_used} points for ₹{final_discount:.2f} discount"
                }
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/blockchain/reviews/verify")
async def verify_review_blockchain(
    review_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Verify review on blockchain"""
    try:
        review_id = review_data.get('review_id')
        booking_id = review_data.get('booking_id')
        
        if not review_id or not booking_id:
            raise HTTPException(status_code=400, detail="Review ID and Booking ID are required")
        
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                # Check if review exists and belongs to user
                await cur.execute(
                    "SELECT * FROM reviews WHERE id = %s AND user_id = %s",
                    (review_id, current_user['id'])
                )
                review = await cur.fetchone()
                
                if not review:
                    raise HTTPException(status_code=404, detail="Review not found")
                
                # Check if booking is verified on blockchain
                await cur.execute(
                    "SELECT * FROM blockchain_bookings WHERE booking_id = %s AND verification_status = 'verified'",
                    (booking_id,)
                )
                blockchain_booking = await cur.fetchone()
                
                if not blockchain_booking:
                    raise HTTPException(status_code=400, detail="Booking must be verified on blockchain first")
                
                # Get user wallet
                await cur.execute(
                    "SELECT wallet_address FROM users WHERE id = %s",
                    (current_user['id'],)
                )
                user_data = await cur.fetchone()
                
                if not user_data or not user_data['wallet_address']:
                    raise HTTPException(
                        status_code=400,
                        detail="Please connect your wallet first"
                    )
                
                # Save blockchain review record
                blockchain_review_id = str(uuid.uuid4())
                await cur.execute("""
                    INSERT INTO blockchain_reviews (
                        id, review_id, user_id, booking_id, user_wallet, review_hash,
                        contract_address, verification_status, blockchain_network
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    blockchain_review_id, review_id, current_user['id'], booking_id, user_data['wallet_address'],
                    f"review_hash_{review_id}", os.environ.get('CONTRACT_ADDRESS_REVIEWS', ''),
                    'verified', 'sepolia'
                ))
                
                # Update review blockchain status
                await cur.execute("""
                    UPDATE reviews 
                    SET blockchain_verified = %s
                    WHERE id = %s
                """, (True, review_id))
                
                # Award bonus loyalty points for verified review
                await cur.execute("""
                    UPDATE loyalty_points 
                    SET points_balance = points_balance + 25, 
                        total_earned = total_earned + 25,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = %s
                """, (current_user['id'],))
                
                return {
                    "success": True,
                    "review_id": review_id,
                    "verification_status": "verified",
                    "bonus_points_awarded": 25
                }
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Include the router in the main app
app.include_router(api_router)

@app.on_event("startup")
async def startup_event():
    await init_db()
    await create_missing_tables()
    print("Database connection initialized and tables created")

async def create_missing_tables():
    """Create missing tables if they don't exist"""
    try:
        pool = await get_db()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                # Create provider_destinations table
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS provider_destinations (
                        id VARCHAR(255) PRIMARY KEY,
                        provider_id VARCHAR(255) NOT NULL,
                        destination_id VARCHAR(255) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (provider_id) REFERENCES providers(id) ON DELETE CASCADE,
                        FOREIGN KEY (destination_id) REFERENCES destinations(id) ON DELETE CASCADE,
                        UNIQUE KEY unique_provider_destination (provider_id, destination_id)
                    )
                """)
                
                # Add city_origin column to bookings table if it doesn't exist
                try:
                    await cur.execute("""
                        ALTER TABLE bookings ADD COLUMN city_origin VARCHAR(100) DEFAULT NULL COMMENT 'City of origin for the booking'
                    """)
                    print("Added city_origin column to bookings table")
                except Exception as e:
                    if "Duplicate column name" not in str(e):
                        print(f"Error adding city_origin column: {str(e)}")
                
                # Create indexes with error handling
                try:
                    await cur.execute("""
                        CREATE INDEX idx_provider_destinations_provider 
                        ON provider_destinations(provider_id)
                    """)
                except Exception as idx_e:
                    if "Duplicate key name" not in str(idx_e):
                        print(f"Warning: Could not create index idx_provider_destinations_provider: {str(idx_e)}")
                
                try:
                    await cur.execute("""
                        CREATE INDEX idx_provider_destinations_destination 
                        ON provider_destinations(destination_id)
                    """)
                except Exception as idx_e:
                    if "Duplicate key name" not in str(idx_e):
                        print(f"Warning: Could not create index idx_provider_destinations_destination: {str(idx_e)}")
                
                # Create chat_logs table
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS chat_logs (
                        id VARCHAR(255) PRIMARY KEY,
                        user_id VARCHAR(255),
                        session_id VARCHAR(255),
                        message TEXT NOT NULL,
                        response TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                        INDEX idx_chat_logs_user_session (user_id, session_id),
                        INDEX idx_chat_logs_created (created_at)
                    )
                """)
                
                # Update itineraries table structure if needed
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS itineraries (
                        id VARCHAR(255) PRIMARY KEY,
                        user_id VARCHAR(255),
                        destination VARCHAR(255) NOT NULL,
                        days INT NOT NULL,
                        budget DECIMAL(10,2) NOT NULL,
                        content TEXT,
                        preferences JSON,
                        generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                    )
                """)
                
                print("Missing tables created successfully")
    except Exception as e:
        print(f"Error creating tables: {str(e)}")

@app.on_event("shutdown")  
async def shutdown_event():
    global db_pool
    if db_pool:
        db_pool.close()
        await db_pool.wait_closed()
    print("Database connection closed")
    
