from enum import Enum
from typing import Set, List, Dict, Any
from dataclasses import dataclass

class Role(Enum):
    """User roles in the FinSolve system"""
    EMPLOYEE = "employee"
    ENGINEERING = "engineering" 
    FINANCE = "finance"
    MARKETING = "marketing"
    HR = "hr"
    C_LEVEL = "c_level"
    ADMIN = "admin"  # NEW: Explicit admin role

class Department(Enum):
    """Document departments for access control"""
    GENERAL = "general"
    ENGINEERING = "engineering"
    FINANCE = "finance"
    MARKETING = "marketing"
    HR = "hr"
    EXECUTIVE = "executive"
    IT = "it"  # NEW: IT department for admins

@dataclass
class Permission:
    """Represents permissions for a role"""
    allowed_departments: Set[Department]
    access_level: int  # 1=basic, 2=departmental, 3=cross-dept, 4=full

# RBAC Configuration: Role -> Permissions mapping
ROLE_PERMISSIONS = {
    Role.EMPLOYEE: Permission(
        allowed_departments={Department.GENERAL},
        access_level=1
    ),
    Role.ENGINEERING: Permission(
        allowed_departments={Department.GENERAL, Department.ENGINEERING},
        access_level=2
    ),
    Role.FINANCE: Permission(
        allowed_departments={Department.GENERAL, Department.FINANCE},
        access_level=2
    ),
    Role.MARKETING: Permission(
        allowed_departments={Department.GENERAL, Department.MARKETING},
        access_level=2
    ),
    Role.HR: Permission(
        allowed_departments={Department.GENERAL, Department.HR},
        access_level=2
    ),
    Role.C_LEVEL: Permission(
        allowed_departments={
            Department.GENERAL, Department.ENGINEERING, Department.FINANCE,
            Department.MARKETING, Department.HR, Department.EXECUTIVE
        },
        access_level=4
    ),
    # NEW: Admin role permissions – full access including IT department
    Role.ADMIN: Permission(
        allowed_departments={
            Department.GENERAL, Department.ENGINEERING, Department.FINANCE,
            Department.MARKETING, Department.HR, Department.EXECUTIVE, Department.IT
        },
        access_level=4
    )
}

# Document access mapping: filename -> department and access info
DOCUMENT_CATEGORIES = {
    # General access documents (all employees)
    "employee_handbook.md": {
        "department": Department.GENERAL,
        "type": "handbook",
        "access_level": 1
    },
    "company_policies.md": {
        "department": Department.GENERAL,
        "type": "policies", 
        "access_level": 1
    },
    
    # Finance documents
    "financial_summary.md": {
        "department": Department.FINANCE,
        "type": "report",
        "access_level": 2
    },
    "budget_analysis.csv": {
        "department": Department.FINANCE,
        "type": "data",
        "access_level": 2
    },
    
    # HR documents
    "sample_hr_data.csv": {
        "department": Department.HR,
        "type": "data",
        "access_level": 2
    },
    "enhanced_hr_data.csv": {
        "department": Department.HR,
        "type": "data",
        "access_level": 2
    },
    
    # Engineering documents
    "engineering_projects.csv": {
        "department": Department.ENGINEERING,
        "type": "data",
        "access_level": 2
    },
    "technical_docs.md": {
        "department": Department.ENGINEERING,
        "type": "documentation",
        "access_level": 2
    },
    
    # Marketing documents
    "marketing_campaigns.csv": {
        "department": Department.MARKETING,
        "type": "data",
        "access_level": 2
    },
    
    # Executive documents (C-Level only)
    "strategic_plan_2025_2027.md": {
        "department": Department.EXECUTIVE,
        "type": "strategic",
        "access_level": 4
    },
    # NEW: Admin-only / IT documents
    "system_config.md": {
        "department": Department.IT,
        "type": "configuration",
        "access_level": 4
    },
    "admin_manual.pdf": {
        "department": Department.IT,
        "type": "manual",
        "access_level": 4
    },
    "security_audit.csv": {
        "department": Department.IT,
        "type": "audit",
        "access_level": 4
    }
}

# Demo users configuration (keep your existing users)
# Line 138-177 in roles.py
DEMO_USERS: Dict[str, Dict[str, Any]] = {
    # Standard employees
    "finance.lead@finsolve.com": {
        "password": "password123",
        "name": "John Doe",
        "role": "finance",
        "department": "Finance"
    },
    "hr.lead@finsolve.com": {
        "password": "password123", 
        "name": "Jane Smith",
        "role": "hr",
        "department": "Human Resources"
    },
    "peter.pandey@finsolve.com": {
        "password": "password123",
        "name": "Peter Pandey", 
        "role": "engineering",
        "department": "Engineering"
    },
    "marketing.lead@finsolve.com": {
        "password": "password123",
        "name": "Sarah Wilson",
        "role": "marketing", 
        "department": "Marketing"
    },

    # CEO with admin privileges (C-Level)
    "tony.sharma@finsolve.com": {
        "password": "password123",
        "name": "David Brown",
        "role": "c_level",
        "department": "Executive",
        "is_admin": True,
        "admin_level": "c_level"
    },

    # Generic employee
    "employee@finsolve.com": {
        "password": "password123",
        "name": "General Employee",
        "role": "employee",
        "department": "General"
    },

    # Explicit test admins
    "admin@finsolve.com": {
        "password": "admin123",
        "name": "System Administrator", 
        "role": "admin",
        "department": "IT",
        "is_admin": True,
        "admin_level": "admin"
    },
    "demo_admin@finsolve.com": {
        "password": "demo123",
        "name": "Demo Administrator",
        "role": "admin", 
        "department": "IT",
        "is_admin": True,
        "admin_level": "admin"
    }
}
def can_access_document(user_role: Role, document_name: str) -> bool:
    """ 
    Check if a user role can access a specific document
    
    Args:
        user_role: The user's role enum
        document_name: Name of the document file
        
    Returns:
        bool: True if user can access document, False otherwise
    """
    if user_role not in ROLE_PERMISSIONS:
        return False
    
    # If document not in our mapping, default to general access
    if document_name not in DOCUMENT_CATEGORIES:
        user_permissions = ROLE_PERMISSIONS[user_role]
        return Department.GENERAL in user_permissions.allowed_departments
    
    # Check specific document access
    doc_info = DOCUMENT_CATEGORIES[document_name]
    doc_department = doc_info["department"]
    user_permissions = ROLE_PERMISSIONS[user_role]
    
    return doc_department in user_permissions.allowed_departments

def get_accessible_documents(user_role: Role) -> List[str]:
    """
    Get list of document names accessible to a specific role
    
    Args:
        user_role: The user's role enum
        
    Returns:
        List[str]: List of accessible document filenames
    """
    accessible = []
    user_permissions = ROLE_PERMISSIONS.get(user_role)
    
    if not user_permissions:
        return []
    
    for doc_name, doc_info in DOCUMENT_CATEGORIES.items():
        if doc_info["department"] in user_permissions.allowed_departments:
            accessible.append(doc_name)
    
    return accessible

def get_access_denied_message(user_role: Role) -> str:
    """
    Get appropriate access denied message for a role
    
    Args:
        user_role: The user's role enum
        
    Returns:
        str: Contextual message for the role
    """
    messages = {
        Role.EMPLOYEE: "As a general employee, you have access to company policies and general information. For departmental data, please contact the relevant department.",
        Role.ENGINEERING: "You have access to engineering documentation and general company information. For other departmental data, please contact the appropriate team.",
        Role.FINANCE: "You have access to financial reports and general company information. For other departmental data, please contact the appropriate team.",
        Role.MARKETING: "You have access to marketing data and general company information. For other departmental data, please contact the appropriate team.",
        Role.HR: "You have access to HR data and general company information. For other departmental data, please contact the appropriate team.",
        Role.C_LEVEL: "You have full access to all company information.",
        Role.ADMIN: "You have full administrative access to the system."
    }
    
    return messages.get(user_role, "Please contact your administrator for access information.")

def has_access(user_role: Role, document_department: Department) -> bool:
    """
    Check if user role has access to documents from a specific department
    
    Args:
        user_role: The user's role enum
        document_department: The document's department enum
        
    Returns:
        bool: True if user has access, False otherwise
    """
    user_permissions = ROLE_PERMISSIONS.get(user_role)
    if not user_permissions:
        return False
    
    return document_department in user_permissions.allowed_departments

# === Helper functions for admin checks ===

def is_admin_user(role: Role, email: str | None = None) -> bool:
    """Return True if the user has administrative privileges"""
    if role in {Role.ADMIN, Role.C_LEVEL}:  # Role-based admin
        return True

    if email:
        return email in {
            "david.brown@finsolve.com",
            "admin@finsolve.com",
            "demo_admin@finsolve.com"
        }
    return False


def get_admin_level(role: Role, email: str | None = None) -> str:
    """Return string admin level for UI / logging purposes"""
    if role == Role.C_LEVEL:
        return "c_level"
    if role == Role.ADMIN:
        return "admin"

    if email == "david.brown@finsolve.com":
        return "c_level"
    if email in {"admin@finsolve.com", "demo_admin@finsolve.com"}:
        return "admin"
    return "none"

# Test function for debugging
def test_rbac_system():
    """Test the RBAC system with various role/document combinations"""
    print("🧪 Testing RBAC System")
    print("=" * 50)
    
    test_cases = [
        (Role.EMPLOYEE, "employee_handbook.md", True),
        (Role.EMPLOYEE, "sample_hr_data.csv", False),
        (Role.EMPLOYEE, "financial_summary.md", False),
        (Role.HR, "sample_hr_data.csv", True),
        (Role.HR, "financial_summary.md", False),
        (Role.FINANCE, "financial_summary.md", True),
        (Role.FINANCE, "sample_hr_data.csv", False),
        (Role.C_LEVEL, "sample_hr_data.csv", True),
        (Role.C_LEVEL, "financial_summary.md", True),
        (Role.C_LEVEL, "strategic_plan_2025_2027.md", True),
    ]
    
    for role, document, expected in test_cases:
        result = can_access_document(role, document)
        status = "✅" if result == expected else "❌"
        print(f"{status} {role.value} → {document}: {result} (expected: {expected})")
    
    print(f"\n📋 Role Access Summary:")
    for role in Role:
        accessible_docs = get_accessible_documents(role)
        print(f"{role.value}: {accessible_docs}")

if __name__ == "__main__":
    test_rbac_system()
