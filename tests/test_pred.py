from project_name.predict_model import make_prediction
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def test_pred():
    """
    Testing the make_prediction function to make sure,
    that it can make predictions given a tokenizer, model and an input prompt.
    """

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    prompt = "How can predictions be tested"
    predicted_text = make_prediction(model, tokenizer, prompt, 20)

    assert len(predicted_text) > len(prompt), "No prediction added to the prompt"