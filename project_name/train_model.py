import os
import time
import torch
import random
import datetime
import pandas as pd
from torch.utils.data import DataLoader, random_split, RandomSampler, SequentialSampler
from project_name.models.model import Model
from project_name.data.dataclass import GPT2Dataset
from omegaconf import OmegaConf

# loading
config = OmegaConf.load("project_name/config.yaml")

parameter = {
    "epochs": config["hyperparameters"]["epochs"],
    "learning_rate": config["hyperparameters"]["learning_rate"],
    "warmup_steps": config["hyperparameters"]["warmup_steps"],
    "epsilon": config["hyperparameters"]["epsilon"],
    "batch_size": config["hyperparameters"]["batch_size"],
    # this produces sample output every 100 steps
    "sample_every": config["hyperparameters"]["sample_every"],
}


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


def dataloader(tokenizer, batch_size):
    df = pd.read_csv("data/processed/processed_data.csv") 
    dataset = GPT2Dataset(df['text'], tokenizer, max_length=768)
    print(len(dataset))
    
    # Split into training and validation sets
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print("{:>5,} training samples".format(train_size))
    print("{:>5,} validation samples".format(val_size))

    # Create the DataLoaders for our training and validation datasets.
    # We'll take training samples in random order.
    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=batch_size,  # Trains with this batch size.
    )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
        val_dataset,  # The validation samples.
        sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
        batch_size=batch_size,  # Evaluate with this batch size.
    )
    return train_dataloader, validation_dataloader


def train():
    # Saving loss
    training_stats = []

    total_t0 = time.time()

    # A GPT model with arguments "lr" of learning rate & "eps" of epsilon
    model = Model(lr=parameter["learning_rate"], eps=parameter["epsilon"])
    tokenizer = model.tokenizer

    # Load the data
    train_dataloader, valid_dataloader = dataloader(tokenizer, parameter["batch_size"])

    # Tell pytorch to run this model on the GPU.
    device = torch.device("cuda")
    model = model.to(device)
    model.cuda()

    # this step is necessary because I've added some tokens (bos_token, etc) to the embeddings
    # otherwise the tokenizer and model tensors won't match up
    model.resize_token_embeddings(len(tokenizer))

    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * parameter["epochs"]
    # Create the optimizer(AdamW)
    optimizer = model.configure_optimizers()
    # Create the learning rate scheduler.
    # This changes the learning rate as the training loop progresses
    scheduler = model.configure_scheduler(num_warmup_steps=parameter["warmup_steps"], num_training_steps=total_steps)

    for epoch_i in range(0, parameter["epochs"]):
        # ========================================
        #               Training
        # ========================================
        print("")
        print("======== Epoch {:} / {:} ========".format(epoch_i + 1, parameter["epochs"]))
        print("Training...")

        # Save the starting training time for epoch
        t0 = time.time()
        total_train_loss = 0

        # start training mode
        model.train()

        for step, batch in enumerate(train_dataloader):
            loss = batch_train(model, batch)
            batch_loss = loss.item()
            total_train_loss += batch_loss

            # Get sample every x batches.
            if step % parameter["sample_every"] == 0:
                elapsed = format_time(time.time() - t0)
                print(
                    "  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.".format(
                        step, len(train_dataloader), batch_loss, elapsed
                    )
                )
                sample(model)

            loss.backward()
            optimizer.step()
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================

        print("")
        print("Running Validation...")
        avg_val_loss, validation_time = valid(model, valid_dataloader)
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                "epoch": epoch_i + 1,
                "Training Loss": avg_train_loss,
                "Valid. Loss": avg_val_loss,
                "Training Time": training_time,
                "Validation Time": validation_time,
            }
        )

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

    # ========================================
    #               Save
    # ========================================
    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    output_dir = "models/"

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model.save_model(output_dir)


def batch_train(model, batch):
    b_input_ids = batch[0]
    b_labels = batch[0]
    b_masks = batch[1]

    model.zero_grad()

    outputs = model(input_ids=b_input_ids, labels=b_labels, attention_mask=b_masks)
    loss = outputs[0]

    return loss


def sample(model):
    model.eval()

    sample_outputs = model.generate(bos_token_id=random.randint(1, 30000), num_return_sequences=3)
    for i, sample_output in enumerate(sample_outputs):
        print("{}: {}".format(i, sample_output))

    model.train()


def valid(model, valid_dataloader):
    t0 = time.time()

    model.eval()

    total_eval_loss = 0

    # Evaluate data for one epoch
    for batch in valid_dataloader:
        b_input_ids = batch[0]
        b_labels = batch[0]
        b_masks = batch[1]

        with torch.no_grad():
            outputs = model(input_ids=b_input_ids, attention_mask=b_masks, labels=b_labels)

            loss = outputs[0]
            print(loss)

        batch_loss = loss.item()
        total_eval_loss += batch_loss

    avg_val_loss = total_eval_loss / len(valid_dataloader)

    validation_time = format_time(time.time() - t0)
    return avg_val_loss, validation_time


if __name__ == "__main__":
    train()
