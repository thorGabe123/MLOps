import torch
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup
from omegaconf import OmegaConf

# loading
config = OmegaConf.load("project_name/config.yaml")

lr = config["hyperparameters"]["learning_rate"]
eps = config["hyperparameters"]["epsilon"]


class Model(torch.nn.Module):
    def __init__(self, lr=lr, eps=eps, model_version="gpt2"):
        super().__init__()

        # Load the GPTModel
        configuration = GPT2Config.from_pretrained(model_version, output_hidden_states=False)
        self.model = GPT2LMHeadModel.from_pretrained(model_version, config=configuration)
        # Load the GPT tokenizer.
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            model_version, bos_token="<|startoftext|>", eos_token="<|endoftext|>", pad_token="<|pad|>"
        )  # gpt2-medium)
        # Setting parameters
        self.lr = lr
        self.epsilon = eps

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, token_type_ids=None)
        return outputs

    def resize_token_embeddings(self, size):
        return self.model.resize_token_embeddings(size)

    def generate(self, inputs=None, bos_token_id=None, max_output_length=200, num_return_sequences=1):
        # Use the generate method from the GPTmodel model
        generated_text = []
        sample_outputs = self.model.generate(
            inputs=inputs,
            bos_token_id=bos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            max_length=max_output_length,
            top_p=0.90,
            num_return_sequences=num_return_sequences,
        )
        for sample_output in sample_outputs:
            generated_text.append(self.tokenizer.decode(sample_output, skip_special_tokens=True))
        return generated_text

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr, eps=self.epsilon)

    def configure_scheduler(self, num_warmup_steps, num_training_steps):
        return get_linear_schedule_with_warmup(
            self.configure_optimizers(), num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )

    def save_model(self, output_dir):
        # Save the model state dictionary
        self.model.save_pretrained(output_dir)
        # Save the tokenizer
        self.tokenizer.save_pretrained(output_dir)

        print("Saving model to %s" % output_dir)
