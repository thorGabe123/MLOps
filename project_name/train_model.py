from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, GPT2LMHeadModel, DataCollatorForLanguageModeling


model_checkpoint = "gpt2"

def tokenize_function(examples):
    return tokenizer(examples["text"])


print("Loading dataset")
dataset = load_dataset('izumi-lab/open-text-books')

print("Tokenizing text")
tokenizer = GPT2Tokenizer.from_pretrained(model_checkpoint, bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
tokenized_datasets = dataset.map(tokenize_function, remove_columns=["text"])

# Define Data Collator for Language Modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define Training Arguments
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=1000,
    evaluation_strategy="steps",
)

training_args = TrainingArguments(
    "model1-open-text-books",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    num_train_epochs=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
)

print("Begin training")
trainer.train()
print("Done")



