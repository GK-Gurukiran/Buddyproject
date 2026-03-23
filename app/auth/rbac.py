"""
Role-Based Access Control (RBAC) system.
Defines permissions per role and provides decorators for route protection.
"""

from functools import wraps
from typing import List
from fastapi import HTTPException, status
from app.models.database import UserRole


# ──────────────────────────────────────────────
# Permission Matrix
# ──────────────────────────────────────────────

ROLE_PERMISSIONS = {
    UserRole.SUPER_ADMIN: [
        "manage_tenants", "manage_all_users", "view_all_data",
        "manage_own_users", "chat", "view_memory", "manage_memory",
        "use_voice", "view_sessions", "delete_sessions",
    ],
    UserRole.TENANT_ADMIN: [
        "manage_own_users", "chat", "view_memory", "manage_memory",
        "use_voice", "view_sessions", "delete_sessions",
    ],
    UserRole.USER: [
        "chat", "view_memory", "use_voice", "view_sessions",
    ],
    UserRole.VIEWER: [
        "view_sessions",
    ],
}


def has_permission(role: str, permission: str) -> bool:
    """Check if a role has a specific permission."""
    try:
        user_role = UserRole(role)
    except ValueError:
        return False
    return permission in ROLE_PERMISSIONS.get(user_role, [])


def require_permissions(permissions: List[str]):
    """
    Dependency factory: returns a checker that raises 403 if the user
    lacks any of the required permissions.

    Usage in routes:
        @router.get("/admin", dependencies=[Depends(require_permissions(["manage_tenants"]))])
    """
    def checker(current_user=None, role: str = ""):
        for perm in permissions:
            if not has_permission(role, perm):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission denied. Required: {perm}"
                )
    return checker


def check_tenant_access(user_tenant_id: str, resource_tenant_id: str, user_role: str) -> bool:
    """
    Verify a user can access a resource belonging to a specific tenant.
    Super admins can access everything; others only their own tenant.
    """
    if user_role == UserRole.SUPER_ADMIN.value:
        return True
    return user_tenant_id == resource_tenant_id
