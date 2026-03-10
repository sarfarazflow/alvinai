from pydantic import BaseModel, EmailStr


class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    email: str
    password: str
    full_name: str = ""
    department: str = ""


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
