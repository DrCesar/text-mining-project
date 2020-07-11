from typing import List

from fastapi import FastAPI, Query

from src.models.train import train
from src.models.predict import predict

app = FastAPI()



@app.get("/")
async def test():
    return { 'result': 'hello world' }

@app.get('/train')
async def train_model():
    train()

    return { 'result': 'model trained' }


@app.get('/predict')
async def predict_review(reviews: List[str] = Query(..., description='Reviews to process')):
    print(reviews)
    predictions = predict(reviews)

    response = [
        {
            'id': idx + 1,
            'review': review,
            'prediction': sentiment
        }
        for idx, (review, sentiment) in enumerate(zip(reviews, predictions))
    ]

    return response
