from fastapi import FastAPI, Form
from http import HTTPStatus
from project_name.predict_model import make_prediction
from project_name.models.model import Model

app = FastAPI()

@app.get("/")
def root():
    """ Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@app.get("/prompt/")
async def generate_text(prompt: str = Form()):
    model = Model(model_version='models/MLOps data')
    predicted_text = make_prediction(model, prompt, 100)

    return predicted_text