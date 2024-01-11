import sys
sys.path.append('C:\\Users\\min\\Documents\\GitHub\\MLOps')
from project_name.predict_model import make_prediction
from project_name.models.model import Model


def test_pred():
    """
    Testing the make_prediction function to make sure,
    that it can make predictions given a tokenizer, model and an input prompt.
    """

    model = Model(model_version='gpt2')
    prompt = "How can predictions be tested"
    predicted_text = make_prediction(model, prompt, 20)

    assert len(predicted_text) > len(prompt), "No prediction added to the prompt"
