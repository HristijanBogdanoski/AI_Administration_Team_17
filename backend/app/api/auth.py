from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta

from app.db.session import get_db
from app.schemas.auth import (
    UserRegister,
    UserLogin,
    Token,
    UserRegisterResponse,
    UserRole,
    UserResponse,
    MessageResponse,
    UserSelfUpdateRequest,
)
from app.services.user_service import (
    create_user,
    authenticate_user,
    get_user_by_embg,
    update_user_role,
    delete_user_by_email,
    get_all_users,
)
from app.core.security import create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES, require_admin
from app.core.security import get_current_user, verify_password, hash_password
from app.models.user import User

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserRegisterResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserRegister, db: Session = Depends(get_db)):
    """
    Register a new user.
    
    - **email**: User's email address (must be unique)
    - **full_name**: User's full name
    - **embg**: Optional unique social security number (13 digits)
    - **password**: Password (minimum 8 characters)
    
    Returns the created user object (without password).
    """
    if user_data.embg:
        existing_embg = get_user_by_embg(db, user_data.embg)
        if existing_embg is not None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="EMBG already registered"
            )

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


@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    """OAuth2-compatible token endpoint used by Swagger Authorize."""
    user = authenticate_user(db, form_data.username, form_data.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email, "user_id": user.id, "role": user.role},
        expires_delta=access_token_expires,
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/admin/health", dependencies=[Depends(require_admin)], summary="Check admin access")
async def admin_health():
    return {"message": "Admin access granted"}


@router.get("/admin/users", response_model=list[UserResponse])
async def list_users(
    db: Session = Depends(get_db),
    _admin=Depends(require_admin),
):
    return get_all_users(db)


@router.put("/admin/users/{email}/role", response_model=UserResponse)
async def set_user_role(
    email: str,
    role: UserRole,
    db: Session = Depends(get_db),
    _admin=Depends(require_admin),
):
    user = update_user_role(db, email, role.value)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return user


@router.delete("/admin/users/{email}", response_model=MessageResponse)
async def remove_user(
    email: str,
    db: Session = Depends(get_db),
    admin=Depends(require_admin),
):
    if email == admin.email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot delete your own admin account",
        )

    deleted = delete_user_by_email(db, email)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    return {"message": f"User {email} deleted"}


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """Get current authenticated user's profile."""
    return current_user


@router.put("/me", response_model=UserResponse)
async def update_me(
    payload: UserSelfUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Update current authenticated user's account.

    - email can be changed if unique
    - password can be changed only with current_password
    - embg can be set only if currently empty and must be unique
    """

    changed = False

    if payload.email is not None and payload.email != current_user.email:
        existing_email = db.query(User).filter(User.email == payload.email).first()
        if existing_email is not None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered",
            )
        current_user.email = payload.email
        changed = True

    if payload.new_password is not None:
        if payload.current_password is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is required to set a new password",
            )
        if not verify_password(payload.current_password, current_user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect",
            )
        current_user.hashed_password = hash_password(payload.new_password)
        changed = True

    if payload.embg is not None:
        if current_user.embg is not None and payload.embg != current_user.embg:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="EMBG cannot be changed once set",
            )

        if current_user.embg is None:
            existing_embg = db.query(User).filter(User.embg == payload.embg).first()
            if existing_embg is not None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="EMBG already registered",
                )
            current_user.embg = payload.embg
            changed = True

    if changed:
        db.commit()
        db.refresh(current_user)

    return current_user
