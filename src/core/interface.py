from typing import List, Literal, TypedDict

from pydantic import BaseModel, Field


class Entity(TypedDict):
    label: str
    text: str


def define_medical_entities_class(labels):
    """Define the medical entities class"""

    # define the medical entity class
    class MedicalEntity(BaseModel):
        label: Literal.__getitem__(tuple(labels)) = Field(...)
        text: str = Field(..., min_length=1)

    # define the medical entity list class
    class MedicalEntityList(BaseModel):
        entities: List[MedicalEntity] = Field(...)

    return MedicalEntityList
