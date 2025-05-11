from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from backend.models import DateRange, Candle
from backend.connection import candle_collection
from datetime import datetime as dt


app = FastAPI()

@app.post('/add_batch')
def add_candle_batch(batch: list[Candle]):
    candle_lst = [candle.model_dump() for candle in batch]
    candle_collection.insert_many(candle_lst)
    
    return {'message': f'Добавлены свечи за {batch[0].timestamp.isoformat()}'}


@app.post('/candles', response_model=list[Candle])
def retrieve_candles(date_range: DateRange):
    query = {
        "timestamp": {
            "$gte": date_range.date_from,
            "$lte": date_range.date_to
        }
    }
    candles = candle_collection.find(query)

    return candles