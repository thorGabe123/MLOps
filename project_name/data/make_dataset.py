from datasets import load_dataset
import pandas as pd


def get_data():
    dataset = load_dataset("izumi-lab/open-text-books")
    return dataset


def process_data(rawDataset):
    dataset = pd.DataFrame(rawDataset)

    array_of_books = []
    for a in dataset["train"]:
        array_of_books.append(a["text"])

    df = pd.DataFrame(array_of_books, columns=["text"])

    df.dropna(inplace=True)  # remove NA values
    return df


if __name__ == "__main__":
    # Get the data and process it
    df = process_data(get_data())
    df.to_csv("data/processed/processed_data.csv", index=False)
    pass
