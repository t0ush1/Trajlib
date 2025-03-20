from trajlib.data.data_reader.chengdu import read_data_chengdu
from trajlib.data.data import GPSData, GridData


def create_data(config):
    data_config = config["data_config"]

    if data_config["data_name"] == "chengdu":
        raw_data = read_data_chengdu(data_config["data_path"], data_config["data_size"])

    if data_config["data_form"] == "gps":
        data = GPSData(raw_data)
    elif data_config["data_form"] == "grid":
        data = GridData(raw_data, step=data_config["grid_step"])
        config["model_config"]["vocab_size"] = len(data.grid)

    return data
