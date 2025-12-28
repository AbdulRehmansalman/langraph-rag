from fastapi import APIRouter, Depends, Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import timedelta
from app.models.schemas import (
    UserCreate, UserLogin, User, Token, EmailVerification,
    ForgotPassword, ResetPassword, ChangePassword
)
from app.core.security import verify_password, get_password_hash, create_access_token, verify_token
from app.core.config import settings
from app.services.supabase_client import supabase_client
from app.services.email_service import email_service
from app.core.logging import get_logger

router = APIRouter()
security = HTTPBearer()
logger = get_logger(__name__)


@router.post("/register", response_model=dict)
async def register(user: UserCreate):
    """Register a new user with email verification."""
    # Check if user already exists
    existing_user = supabase_client.table("users").select("*").eq("email", user.email).execute()
    if existing_user.data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Hash password and create user
    hashed_password = get_password_hash(user.password)
    new_user = {
        "email": user.email,
        "hashed_password": hashed_password,
        "full_name": user.full_name,
        "timezone": user.timezone or "UTC",
        "is_active": False,
        "email_verified": False
    }

    result = supabase_client.table("users").insert(new_user).execute()
    if not result.data:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )

    # Send verification email (non-blocking failure)
    try:
        await email_service.send_verification_email(user.email, user.full_name)
    except Exception as e:
        logger.warning(f"Email sending failed for {user.email}: {e}")
        return {
            "message": "User created. Email verification failed - please contact support.",
            "user_id": result.data[0]["id"]
        }

    return {
        "message": "User created successfully. Please check your email for verification code.",
        "user_id": result.data[0]["id"]
    }


@router.post("/login", response_model=Token)
async def login(request: Request, user: UserLogin):
    """Authenticate user and return access token."""
    if not user.email or not user.password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email and password are required"
        )

    # Get user from database
    db_user = supabase_client.table("users").select("*").eq("email", user.email).execute()

    if not db_user.data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )

    user_data = db_user.data[0]

    # Check email verification
    if not user_data.get("email_verified", False):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Please verify your email before logging in"
        )

    # Verify password
    if not verify_password(user.password, user_data["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )

    # Check if user is active
    if not user_data.get("is_active", False):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account is deactivated"
        )

    # Update timezone if provided
    if user.timezone:
        try:
            supabase_client.table("users").update({
                "timezone": user.timezone
            }).eq("id", user_data["id"]).execute()
        except Exception:
            pass  # Non-critical failure

    # Create access token
    access_token = create_access_token(
        data={"sub": user_data["id"]},
        expires_delta=timedelta(minutes=settings.access_token_expire_minutes)
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user_data["id"],
            "email": user_data["email"],
            "full_name": user_data["full_name"],
            "created_at": user_data["created_at"],
            "is_active": user_data["is_active"],
            "email_verified": user_data.get("email_verified", False)
        }
    }


@router.get("/me", response_model=User)
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user."""
    payload = verify_token(credentials.credentials)
    user_id = payload.get("sub")

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

    db_user = supabase_client.table("users").select("*").eq("id", user_id).execute()

    if not db_user.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    user_data = db_user.data[0]
    return User(
        id=user_data["id"],
        email=user_data["email"],
        full_name=user_data["full_name"],
        created_at=user_data["created_at"],
        is_active=user_data["is_active"]
    )


@router.post("/verify-email", response_model=dict)
async def verify_email(verification: EmailVerification):
    """Verify user email with OTP."""
    is_valid = await email_service.verify_otp(verification.email, verification.otp, "verification")

    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OTP"
        )

    result = supabase_client.table("users").update({
        "email_verified": True,
        "is_active": True
    }).eq("email", verification.email).execute()

    if not result.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    return {"message": "Email verified successfully"}


@router.post("/resend-verification", response_model=dict)
async def resend_verification(email_request: ForgotPassword):
    """Resend email verification OTP."""
    db_user = supabase_client.table("users").select("*").eq("email", email_request.email).execute()

    if not db_user.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    user_data = db_user.data[0]

    if user_data.get("email_verified", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email is already verified"
        )

    await email_service.send_verification_email(email_request.email, user_data.get("full_name"))
    return {"message": "Verification email sent successfully"}


@router.post("/forgot-password", response_model=dict)
async def forgot_password(request: ForgotPassword):
    """Send password reset OTP."""
    db_user = supabase_client.table("users").select("*").eq("email", request.email).execute()

    # Don't reveal if email exists
    if db_user.data:
        user_data = db_user.data[0]
        try:
            await email_service.send_password_reset_email(request.email, user_data.get("full_name"))
        except Exception:
            pass

    return {"message": "If the email exists, a password reset code has been sent"}


@router.post("/reset-password", response_model=dict)
async def reset_password(request: ResetPassword):
    """Reset password with OTP."""
    is_valid = await email_service.verify_otp(request.email, request.otp, "password_reset")

    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OTP"
        )

    db_user = supabase_client.table("users").select("*").eq("email", request.email).execute()

    if not db_user.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    user_data = db_user.data[0]
    hashed_password = get_password_hash(request.new_password)

    supabase_client.table("users").update({
        "hashed_password": hashed_password
    }).eq("email", request.email).execute()

    try:
        await email_service.send_password_changed_notification(request.email, user_data.get("full_name"))
    except Exception:
        pass

    return {"message": "Password reset successfully"}


@router.post("/change-password", response_model=dict)
async def change_password(
    request: ChangePassword,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Change password for authenticated user."""
    payload = verify_token(credentials.credentials)
    user_id = payload.get("sub")

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

    db_user = supabase_client.table("users").select("*").eq("id", user_id).execute()

    if not db_user.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    user_data = db_user.data[0]

    if not verify_password(request.current_password, user_data["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )

    hashed_password = get_password_hash(request.new_password)
    supabase_client.table("users").update({
        "hashed_password": hashed_password
    }).eq("id", user_id).execute()

    try:
        await email_service.send_password_changed_notification(user_data["email"], user_data.get("full_name"))
    except Exception:
        pass

    return {"message": "Password changed successfully"}
