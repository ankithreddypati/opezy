from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from uuid import UUID, uuid4

class Feedback(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()), alias="feedbackId")
    comment: Optional[str] = Field(alias="feedbackComment")
    date: datetime = Field(alias="feedbackDate")

