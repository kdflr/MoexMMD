from pydantic import BaseModel
from datetime import datetime


class Candle(BaseModel):
    open: float
    close: float
    high: float
    low: float
    volume: int
    timestamp: datetime


class DateRange(BaseModel):
    date_from: datetime
    date_to: datetime