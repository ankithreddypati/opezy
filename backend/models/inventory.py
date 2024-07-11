from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID, uuid4
from datetime import datetime

class Inventory(BaseModel):
    id: UUID = Field(default_factory=uuid4, alias="itemId")
    name: str = Field(alias="InventoryName")
    cost: float = Field(alias="itemCost")
    description: Optional[str] = Field(None, alias="itemDescription")
    image_path: Optional[str] = Field(None, alias="imagePath")
    expiry_date: Optional[datetime] = Field(None, alias="expiryDate")



