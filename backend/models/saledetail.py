from pydantic import BaseModel, Field
from typing import Optional, List
from .ingredient import Ingredient

class SaleDetail(BaseModel):
    product_id: str = Field(alias="productId")
    purchased_price: float = Field(alias="purchasedPrice")
    quantity: int = Field(alias="quantity")
    selected_add_ons: List[Ingredient] = Field(default=[], alias="selectedAddOns")  
    special_instructions: Optional[str] = Field(default=None, alias="specialInstructions")  
