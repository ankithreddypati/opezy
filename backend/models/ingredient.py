from pydantic import BaseModel, Field
from typing import List, Optional
from uuid import uuid4

class Ingredient(BaseModel):
    name: str
    extra_cost: Optional[float] = 0.0