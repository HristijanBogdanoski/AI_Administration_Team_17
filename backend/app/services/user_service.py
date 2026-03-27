from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from app.models.user import User
from app.core.security import hash_password, verify_password
from app.schemas.auth import UserRegister


def create_user(db: Session, user_data: UserRegister) -> User | None:
    """
    Create a new user in the database.
    Returns the created user or None if email already exists.
    """
    # Check if email already exists
    existing_user = get_user_by_email(db, user_data.email)
    if existing_user:
        return None
    
    try:
        db_user = User(
            email=user_data.email,
            full_name=user_data.full_name,
            hashed_password=hash_password(user_data.password),
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user
    except IntegrityError:
        db.rollback()
        return None


def authenticate_user(db: Session, email: str, password: str) -> User | None:
    """
    Authenticate a user by email and password.
    Returns the user if credentials are valid, None otherwise.
    """
    user = get_user_by_email(db, email)
    if not user:
        return None
    
    if not verify_password(password, user.hashed_password):
        return None
    
    return user


def get_user_by_email(db: Session, email: str) -> User | None:
    """Get a user by email"""
    return db.query(User).filter(User.email == email).first()


def get_user_by_id(db: Session, user_id: int) -> User | None:
    """Get a user by ID"""
    return db.query(User).filter(User.id == user_id).first()
