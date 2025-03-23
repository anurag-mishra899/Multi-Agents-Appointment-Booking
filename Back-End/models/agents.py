from models.tools import DateModel, DateTimeModel, IdentificationNumberModel
from langchain_core.pydantic_v1 import constr, BaseModel, Field, validator
from typing import Optional

# Primary Assistant
class ToPrimaryBookingAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle flight updates and cancellations."""

    request: str = Field(
        description="Any necessary followup questions the update flight assistant should clarify before proceeding."
    )


class ToGetInfo(BaseModel):
    """Get information of doctor availability via name or specialization"""

    desired_date: DateModel = Field(
        description="The desired date for booking"
    )
    specialization: Optional[str] = Field(
        default=None, description="The desired specialization of the doctor"
    )
    doctor_name: Optional[str] = Field(
        default=None, description="The desired doctor name for booking"
    )
    request: str = Field(
        description="Any additional information or requests from the user regarding the appointment."
    )



class ToAppointmentBookingAssistant(BaseModel):
    """Transfer work to a specialized assistant to handle hotel bookings."""

    date:DateTimeModel = Field(
        description="The date for setting, cancel or rescheduling appointment"
    )
    identification_number: IdentificationNumberModel = Field(
        description="The id number of user."
    )
    doctor_number: str = Field(
        description="The name of the doctor"
    )
    request: str = Field(
        description="Any additional information or requests from the user regarding the hotel booking."
    )


class CompleteOrEscalate(BaseModel):
    """A tool to mark the current task as completed and/or to escalate control of the dialog to the main assistant,
    who can re-route the dialog based on the user's needs."""

    cancel: bool = True
    reason: str

    class Config:
        json_schema_extra = {
            "example": {
                "cancel": True,
                "reason": "User changed their mind about the current task.",
            },
            "example 2": {
                "cancel": True,
                "reason": "I have fully completed the task.",
            },
            "example 3": {
                "cancel": False,
                "reason": "I need to search the user's date and time for more information.",
            },
        }