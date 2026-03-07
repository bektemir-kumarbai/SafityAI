from pydantic import BaseModel


class DetectResponse(BaseModel):
    action_type: str