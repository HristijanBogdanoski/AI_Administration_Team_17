from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import timedelta

from app.db.session import get_db
from app.schemas.auth import UserRegister, UserLogin, Token, UserRegisterResponse
from app.services.user_service import create_user, authenticate_user
from app.core.security import create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/register", response_model=UserRegisterResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserRegister, db: Session = Depends(get_db)):
    """
    Register a new user.
    
    - **email**: User's email address (must be unique)
    - **full_name**: User's full name
    - **password**: Password (minimum 8 characters)
    
    Returns the created user object (without password).
    """
    # Create user
    db_user = create_user(db, user_data)
    
    if db_user is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    return db_user


@router.post("/login", response_model=Token)
async def login(credentials: UserLogin, db: Session = Depends(get_db)):
    """
    Login with email and password.
    
    - **email**: User's email address
    - **password**: User's password
    
    Returns a JWT access token.
    """
    user = authenticate_user(db, credentials.email, credentials.password)
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email, "user_id": user.id},
        expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}
