import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel



def make_prediction(model, tokenizer, text, output_length):


    indexed_tokens = tokenizer.encode(text)

    tokens_tensor = torch.tensor([indexed_tokens])

    model.eval()

    # If using GPU
    # tokens_tensor = tokens_tensor.to('cuda')
    # model.to('cuda')

    softmax = torch.nn.Softmax(dim=-1)  # Softmax function to convert logits to probabilities

    with torch.no_grad():
        for i in range(output_length):
            outputs = model(tokens_tensor)
            predictions = outputs[0]

            # Apply softmax to convert logits to probabilities
            probabilities = softmax(predictions[0, -1, :])

            # Get the token with the highest probability
            predicted_index = torch.argmax(probabilities).item()

            #predicted_probability = torch.max(probabilities).item()
            #predicted_token = tokenizer.decode(predicted_index)
            #print(f"Predicted token: {predicted_token}, Probability: {predicted_probability}")

            tokens_tensor = torch.cat((tokens_tensor, torch.tensor([[predicted_index]])), dim=1)

    predicted_text = tokenizer.decode(tokens_tensor[0].tolist())
    return predicted_text


if __name__ == '__main__':
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    text = "What is the fastest car in the"
    predicted_text = make_prediction(model, tokenizer, text, 10)
    print(predicted_text)
