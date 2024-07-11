from datetime import datetime
from pydantic import BaseModel, Field
from uuid import uuid4


class Expense(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()), alias="_id")
    date: datetime = Field(alias="expenseDate")
    category: str = Field(alias="expenseCategory")
    amount: float = Field(alias="expenseAmount")
    description: str = Field(alias="expenseDescription")
