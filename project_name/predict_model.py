import torch
from models.model import Model



def make_prediction(model, input_prompt, max_output_length):
    indexed_tokens = tokenizer.encode(input_prompt)
    tokens_tensor = torch.tensor([indexed_tokens])
    prediction = model.generate(inputs=tokens_tensor, 
                                max_output_length=max_output_length,
                                num_return_sequences=1)

    return prediction


if __name__ == '__main__':
    model = Model()
    tokenizer = model.tokenizer
    prompt = "What is the fastest car in the"
    predicted_text = make_prediction(model, prompt, 200)
    print(predicted_text[0])
