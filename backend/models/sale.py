from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from uuid import UUID, uuid4
from .saledetail import SaleDetail

class Sale(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()), alias="saleId")
    customer_id: str = Field(alias="customerId")
    date: datetime = Field(alias="saleDate")
    sale_details: List[SaleDetail] = Field(default=[], alias="saleDetails")
    total_amount: float = Field(alias="totalAmount")

