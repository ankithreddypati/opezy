from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from uuid import UUID, uuid4
from .ingredient import Ingredient


class Product(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()), alias="productId")
    name: str = Field(alias="productName")
    category: str = Field(alias="productCategory")  
    base_price: float = Field(alias="productPrice")
    description: Optional[str] = Field(default=None, alias="productDescription")
    is_vegetarian: Optional[bool] = Field(default=False, alias="isVegetarian")
    available_add_ons: Optional[List[Ingredient]] = Field(default=[], alias="availableAddOns")