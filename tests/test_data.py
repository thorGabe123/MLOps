import pandas.core.frame
from project_name.data.make_dataset import get_data, process_data


def test_data():
    """
    Testing function to get raw data and the following process_data function to ensure,
    that the dataset is downloaded and processed correctly
    """
    dataset = get_data()
    assert len(dataset['train']['text']) > 1000, "Raw dataset missing text - Under 1000 text entries"
    for i in range(500): # Testing first 500 entries
        assert len(dataset['train']['text'][i]) > 0, f"No data in the {i} entry in the raw data"

    processed_data = process_data(dataset)
    assert processed_data.shape[0] > 1000, "Processed dataset missing text - Under 1000 text entries"
    assert type(processed_data) == pandas.core.frame.DataFrame, "Not returning DataFrame"
    for i in range(500): # Testing first 500 entries
        assert len(processed_data['text'][i]) > 0, f"No data in the {i} entry in the processed data"


