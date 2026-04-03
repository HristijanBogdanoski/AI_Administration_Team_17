from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import timedelta

from app.db.session import get_db
from app.schemas.auth import (
    UserRegister,
    UserLogin,
    Token,
    UserRegisterResponse,
    UserRoleUpdate,
    UserDeleteRequest,
    UserResponse,
    MessageResponse,
)
from app.services.user_service import create_user, authenticate_user, update_user_role, delete_user_by_email
from app.core.security import create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES, require_admin

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
        data={"sub": user.email, "user_id": user.id, "role": user.role},
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/admin/ping", dependencies=[Depends(require_admin)])
async def admin_ping():
    return {"message": "Admin access granted"}


@router.post("/admin/set-role", response_model=UserResponse)
async def set_user_role(
    role_data: UserRoleUpdate,
    db: Session = Depends(get_db),
    _admin=Depends(require_admin),
):
    user = update_user_role(db, role_data.email, role_data.role.value)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return user


@router.delete("/admin/remove-user", response_model=MessageResponse)
async def remove_user(
    user_data: UserDeleteRequest,
    db: Session = Depends(get_db),
    admin=Depends(require_admin),
):
    if user_data.email == admin.email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot delete your own admin account",
        )

    deleted = delete_user_by_email(db, user_data.email)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    return {"message": f"User {user_data.email} deleted"}
