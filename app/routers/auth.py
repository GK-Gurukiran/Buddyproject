"""
Authentication API routes: registration, login, user management.
"""

import bcrypt
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import User, UserRole, get_db
from app.models.schemas import UserRegister, UserLogin, TokenResponse, UserResponse
from app.auth.jwt_handler import create_access_token
from app.auth.dependencies import get_current_user, require_role
from app.tenants.manager import TenantManager

router = APIRouter(prefix="/auth", tags=["Authentication"])


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserRegister, db: AsyncSession = Depends(get_db)):
    """
    Register a new user within a tenant.
    The tenant must already exist (identified by tenant_slug).
    """
    # Check if email already exists
    existing = await db.execute(select(User).where(User.email == user_data.email))
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    # Find the tenant
    tenant_mgr = TenantManager(db)
    tenant = await tenant_mgr.get_tenant_by_slug(user_data.tenant_slug)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant '{user_data.tenant_slug}' not found",
        )

    if not tenant.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tenant is deactivated",
        )

    # Check user limit
    can_add = await tenant_mgr.check_user_limit(tenant.id)
    if not can_add:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tenant has reached maximum user limit",
        )

    # Create the user
    hashed_pw = hash_password(user_data.password)
    user = User(
        email=user_data.email,
        username=user_data.username,
        hashed_password=hashed_pw,
        tenant_id=tenant.id,
        role=UserRole.USER,
    )
    db.add(user)
    await db.flush()
    await db.refresh(user)

    return user


@router.post("/login", response_model=TokenResponse)
async def login(credentials: UserLogin, db: AsyncSession = Depends(get_db)):
    """
    Authenticate a user and return a JWT token.
    """
    result = await db.execute(select(User).where(User.email == credentials.email))
    user = result.scalar_one_or_none()

    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )

    token = create_access_token(
        data={
            "user_id": user.id,
            "tenant_id": user.tenant_id,
            "role": user.role.value,
            "email": user.email,
        }
    )

    return TokenResponse(access_token=token)


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """Get the currently authenticated user's profile."""
    return current_user


@router.get("/users", response_model=list[UserResponse])
async def list_users(
    current_user: User = Depends(require_role("super_admin", "tenant_admin")),
    db: AsyncSession = Depends(get_db),
):
    """
    List users. Tenant admins see their own tenant's users.
    Super admins see all users.
    """
    if current_user.role == UserRole.SUPER_ADMIN:
        result = await db.execute(select(User).order_by(User.created_at.desc()))
    else:
        result = await db.execute(
            select(User)
            .where(User.tenant_id == current_user.tenant_id)
            .order_by(User.created_at.desc())
        )

    return list(result.scalars().all())
