"""
Tenant management API routes.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import User, UserRole, get_db
from app.models.schemas import TenantCreate, TenantResponse
from app.auth.dependencies import get_current_user, require_role
from app.tenants.manager import TenantManager

router = APIRouter(prefix="/tenants", tags=["Tenants"])


@router.post("/", response_model=TenantResponse, status_code=status.HTTP_201_CREATED)
async def create_tenant(
    tenant_data: TenantCreate,
    current_user: User = Depends(require_role("super_admin")),
    db: AsyncSession = Depends(get_db),
):
    """Create a new tenant organization. Super admin only."""
    tenant_mgr = TenantManager(db)
    tenant = await tenant_mgr.create_tenant(tenant_data)
    await db.refresh(tenant)
    return tenant


@router.get("/", response_model=list[TenantResponse])
async def list_tenants(
    current_user: User = Depends(require_role("super_admin")),
    db: AsyncSession = Depends(get_db),
):
    """List all tenants. Super admin only."""
    tenant_mgr = TenantManager(db)
    return await tenant_mgr.list_tenants()


@router.get("/{tenant_id}", response_model=TenantResponse)
async def get_tenant(
    tenant_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a specific tenant. Users can only see their own tenant."""
    if current_user.role != UserRole.SUPER_ADMIN and current_user.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this tenant",
        )

    tenant_mgr = TenantManager(db)
    tenant = await tenant_mgr.get_tenant_by_id(tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found",
        )
    return tenant


@router.delete("/{tenant_id}", response_model=TenantResponse)
async def deactivate_tenant(
    tenant_id: str,
    current_user: User = Depends(require_role("super_admin")),
    db: AsyncSession = Depends(get_db),
):
    """Deactivate a tenant. Super admin only."""
    tenant_mgr = TenantManager(db)
    return await tenant_mgr.deactivate_tenant(tenant_id)
