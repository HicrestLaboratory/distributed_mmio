import os
import yaml

config_file_url = "config.yaml"

import colors


def read_config_file():
    if not os.path.exists(config_file_url):
        raise Exception(
            f"{colors.color_red('config.yaml')} is not present. Refer to the config.example.yaml file to generate your custom config.yaml file"
        )

    with open(config_file_url, "r") as config_file:
        config = yaml.safe_load(config_file)
        print(f"{colors.color_green('All config')}")
        print(config)
        print()

    return config


def create_datasets_dir():
    config = read_config_file()
    if not "path" in config:
        raise Exception(
            f"{colors.color_red('path')} key is required. Refer to the config.example.yaml file"
        )

    data_dir_path = config["path"]

    if not os.path.isdir(data_dir_path):
        os.mkdir(data_dir_path)
        print(f"{colors.color_green(data_dir_path)} directory created")
        print()


def get_data_dir_path():
    config = read_config_file()
    if not "path" in config:
        raise Exception(
            f"{colors.color_red('path')} key is required. Refer to the config.example.yaml file"
        )

    data_dir_path = config["path"]
    return data_dir_path
