import torch
import random
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup
from omegaconf import OmegaConf

# loading
config = OmegaConf.load('project_name/config.yaml')

lr=config['hyperparameters']['learning_rate']
eps=config['hyperparameters']['epsilon']

class Model(torch.nn.Module):
    def __init__(self, lr=lr, eps=eps):
        super().__init__()
        
        # Load the GPTModel
        configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
        self.model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)
        # Load the GPT tokenizer.
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>') #gpt2-medium)
        # Setting parameters
        self.lr = lr
        self.epsilon = eps

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, oken_type_ids=None)
        return outputs
    
    def generate(self, inputs,):
        # Use the generate method from the GPTmodel model

        sample_outputs = self.model.generate(
                                    inputs=inputs,
                                    pad_token_id=self.tokenizer.eos_token_id,
                                    do_sample=True,   
                                    top_k=50, 
                                    max_length = 200,
                                    top_p=0.95, 
                                    num_return_sequences=1)
        
        generated_text = self.tokenizer.decode((sample_outputs.tolist())[0], skip_special_tokens=True)
        return generated_text

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr, eps = self.epsilon)
    
    def configure_scheduler(self, num_warmup_steps, num_training_steps):
        return get_linear_schedule_with_warmup(
                                            self.configure_optimizers(), 
                                            num_warmup_steps = num_warmup_steps, 
                                            num_training_steps = num_training_steps)
        # def common_step(self, batch, batch_idx):
    #     outputs = self(**batch)
    #     loss = outputs.loss

    #     return loss
    
    # def training_step(self, batch, batch_idx):
    #     loss = self.common_step(batch, batch_idx)     
    #     # logs metrics for each training_step,
    #     # and the average across the epoch
    #     self.log("training_loss", loss)

    #     return loss

    # def validation_step(self, batch, batch_idx):
    #     loss = self.common_step(batch, batch_idx)     
    #     self.log("validation_loss", loss, on_epoch=True)

    #     return loss

    # def test_step(self, batch, batch_idx):
    #     loss = self.common_step(batch, batch_idx)     

    #     return loss
