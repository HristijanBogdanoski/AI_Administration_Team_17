from enum import Enum

class ServiceCategory(str, Enum):
    documents = "documents"
    taxes = "taxes"
    social = "social"
    business = "business"
    education = "education"
    utilities = "utilities"