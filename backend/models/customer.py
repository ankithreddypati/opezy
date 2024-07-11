from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from uuid import UUID, uuid4

class Customer(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()), alias="customerId")
    name: str = Field(alias="customerName")
    email: Optional[str] = Field(default=None, alias="customerEmail")
    phone: Optional[str] = Field(default=None, alias="customerPhone")
