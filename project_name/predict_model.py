import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import random


def make_prediction(model, tokenizer, input_prompt, max_output_length):
    indexed_tokens = tokenizer.encode(input_prompt)
    tokens_tensor = torch.tensor([indexed_tokens])

    prediction = model.generate(inputs=tokens_tensor, max_new_tokens=max_output_length,
                                do_sample=True, top_k=50, top_p=0.95, pad_token_id=tokenizer.eos_token_id)
    prediction = tokenizer.decode(((prediction.tolist())[0]))

    return prediction


if __name__ == '__main__':
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    prompt = "What is the fastest car in the"
    predicted_text = make_prediction(model, tokenizer, prompt, 20)
    print(predicted_text)
