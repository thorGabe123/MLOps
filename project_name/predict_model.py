import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from models.model import Model
import random



def make_prediction(model, tokenizer, input_prompt, max_output_length):
    indexed_tokens = tokenizer.encode(input_prompt)
    tokens_tensor = torch.tensor([indexed_tokens])
    prediction = model.generate(inputs=tokens_tensor)

    return prediction


if __name__ == '__main__':
    model = Model()
    tokenizer = model.tokenizer
    prompt = "What is the fastest car in the"
    predicted_text = make_prediction(model, tokenizer, prompt, 20)
    print(predicted_text)
