"""
Tenant management: creation, lookup, and isolation enforcement.
"""

from typing import Optional, List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException, status

from app.models.database import Tenant, User
from app.models.schemas import TenantCreate


class TenantManager:
    """Handles all tenant-related operations with data isolation."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_tenant(self, tenant_data: TenantCreate) -> Tenant:
        """Create a new tenant organization."""
        # Check slug uniqueness
        existing = await self.db.execute(
            select(Tenant).where(Tenant.slug == tenant_data.slug)
        )
        if existing.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Tenant with slug '{tenant_data.slug}' already exists",
            )

        tenant = Tenant(
            name=tenant_data.name,
            slug=tenant_data.slug,
            max_users=tenant_data.max_users,
        )
        self.db.add(tenant)
        await self.db.flush()
        return tenant

    async def get_tenant_by_slug(self, slug: str) -> Optional[Tenant]:
        """Look up a tenant by their unique slug."""
        result = await self.db.execute(select(Tenant).where(Tenant.slug == slug))
        return result.scalar_one_or_none()

    async def get_tenant_by_id(self, tenant_id: str) -> Optional[Tenant]:
        """Look up a tenant by ID."""
        result = await self.db.execute(select(Tenant).where(Tenant.id == tenant_id))
        return result.scalar_one_or_none()

    async def list_tenants(self) -> List[Tenant]:
        """List all tenants (super_admin only)."""
        result = await self.db.execute(select(Tenant).order_by(Tenant.created_at.desc()))
        return list(result.scalars().all())

    async def check_user_limit(self, tenant_id: str) -> bool:
        """Check if a tenant has room for more users."""
        tenant = await self.get_tenant_by_id(tenant_id)
        if not tenant:
            return False
        user_count = await self.db.execute(
            select(User).where(User.tenant_id == tenant_id)
        )
        current_count = len(list(user_count.scalars().all()))
        return current_count < tenant.max_users

    async def deactivate_tenant(self, tenant_id: str) -> Tenant:
        """Soft-delete a tenant by deactivating it."""
        tenant = await self.get_tenant_by_id(tenant_id)
        if not tenant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tenant not found",
            )
        tenant.is_active = False
        await self.db.flush()
        return tenant
