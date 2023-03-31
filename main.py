from fastapi import FastAPI
from pydantic import BaseModel
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizerFast
import tensorflow as tf


class MentalHealthInput(BaseModel):
    description: str


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Welcome to DIRAD API"}


@app.post("/analyze")
async def analyze_mental_health(health_input: MentalHealthInput):

    model_path = '../model/analysis-model'

    # Load the saved model from the local directory
    model = TFDistilBertForSequenceClassification.from_pretrained(model_path)

    # Load the tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        'distilbert-base-uncased')

    # user input
    text = health_input.description

    # Tokenize the text
    inputs = tokenizer(text, truncation=True,
                       padding=True, return_tensors='tf')

    # Make predictions
    outputs = model(inputs)
    prediction = tf.argmax(outputs.logits, axis=1).numpy()
    
    return str(prediction[0])
