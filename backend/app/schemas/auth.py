from enum import Enum

from pydantic import BaseModel, EmailStr, Field


class UserRegister(BaseModel):
    """Schema for user registration"""
    email: EmailStr
    full_name: str = Field(..., min_length=1, max_length=255)
    embg: str | None = Field(default=None, min_length=13, max_length=13)
    password: str = Field(..., min_length=8)
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "full_name": "John Doe",
                "embg": "1234567890123",
                "password": "securepassword123"
            }
        }


class UserLogin(BaseModel):
    """Schema for user login"""
    email: EmailStr
    password: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "securepassword123"
            }
        }


class Token(BaseModel):
    """Schema for JWT token response"""
    access_token: str
    token_type: str = "bearer"
    
    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer"
            }
        }

class UserRole(str, Enum):
    user = "user"
    admin = "admin"


class UserRoleUpdate(BaseModel):
    """Schema for updating a user's role"""
    email: EmailStr
    role: UserRole

    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "role": "admin"
            }
        }


class UserDeleteRequest(BaseModel):
    """Schema for deleting a user by email"""
    email: EmailStr

    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com"
            }
        }


class MessageResponse(BaseModel):
    """Simple message response schema"""
    message: str

class UserResponse(BaseModel):
    """Schema for user response (no password)"""
    id: int
    email: str
    full_name: str | None
    embg: str | None
    role: UserRole
    
    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": 1,
                "email": "user@example.com",
                "full_name": "John Doe"
            }
        }


class UserRegisterResponse(UserResponse):
    """Schema for registration response"""
    pass


class UserSelfUpdateRequest(BaseModel):
    """Schema for authenticated user self-update"""
    email: EmailStr | None = None
    current_password: str | None = Field(default=None, min_length=8)
    new_password: str | None = Field(default=None, min_length=8)
    embg: str | None = Field(default=None, min_length=13, max_length=13)

    class Config:
        json_schema_extra = {
            "example": {
                "email": "new_email@example.com",
                "current_password": "currentpassword123",
                "new_password": "newsecurepassword123",
                "embg": "1234567890123"
            }
        }
