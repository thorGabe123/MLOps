from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from project_name.predict_model import make_prediction
from project_name.models.model import Model

app = FastAPI()

@app.get("/health")
def health():
    """ Health check."""
    response = {
        "message": "OK",
        "status-code": 200,
    }
    return response

@app.get("/", response_class=HTMLResponse)
async def get_prompt():
    html_content = """
    <html>
    <head>
        <title>Text Generator</title>
    </head>
    <body>
        <h1>Text Generator</h1>
        <form action="/generated/" method="get">
            <label for="prompt">Enter Prompt:</label>
            <input type="text" id="prompt" name="prompt" required>
            <button type="submit">Generate Text</button>
        </form>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/generated/", response_class=HTMLResponse)
async def generate_text(prompt: str = Query(..., title="Prompt")):
    model = Model(model_version='models/MLOps data')
    predicted_text = make_prediction(model, prompt, 100)

    html_content = f"""
    <html>
    <head>
        <title>Text Generator</title>
    </head>
    <body>
        <h3>Input</h3>
        <p>{prompt}</p>
        <h3>Generated text</h3>
        <p>{predicted_text}</p>
    </body>
    </html>
    """

    return HTMLResponse(content=html_content)
