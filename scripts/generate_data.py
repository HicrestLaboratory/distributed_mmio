import os
import sys
import subprocess
import shutil
import yaml


# TODO:
# 2. Look at the include folder
# 3. Add generator for Matrix Market
# 4. Writhe the README file


import colors

config_file_url = "config.yaml"
data_dir_path = "datasets"
file_name = "graph_binary"


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


def read_graph_500_config(config, size):
    if not size in config:
        raise Exception(
            f"{colors.color_red('size')} key is not configured properly. Refer to the config.example.yaml file"
        )

    if not "generators" in config[size]:
        raise Exception(
            f"{colors.color_red('generators')} key is not configured properly. Refer to the config.example.yaml file"
        )

    if not "graph500" in config[size]["generators"]:
        raise Exception(
            f"{colors.color_red('graph500')} key is not configured properly. Refer to the config.example.yaml file"
        )

    graph_500_config = config[size]["generators"]["graph500"]

    if not "scale" in graph_500_config:
        raise Exception(
            f"{colors.color_red('scale')} key is required. Refer to the config.example.yaml file"
        )
    if not "edge_factor" in graph_500_config:
        raise Exception(
            f"{colors.color_red('edge_factor')} key is required. Refer to the config.example.yaml file"
        )

    return (graph_500_config["scale"], graph_500_config["edge_factor"])


def show_sizes_menu():
    print()
    print("Choose the size of data to produce:")
    print()
    print("1. small")
    print("2. large")
    print("3. Exit program")
    print()


def get_dataset_size():
    while True:
        show_sizes_menu()
        size = input("Enter your choice (1-3): ").strip()
        if size == "1":
            return "small"
        elif size == "2":
            return "large"
        elif size == "3":
            exit()
        else:
            print("Invalid choice. Please enter a number from 1 to 3.")


def show_sources_menu():
    print()
    print("Choose your source of data:")
    print()
    print("1. generators")
    print("2. suite sparce matrix list")
    print("3. suite sparce matrix ssgetpy util")
    print("4. Exit program")
    print()


def show_generators_menu():
    print()
    print("Choose what generator of data to use:")
    print()
    print("1. graph500")
    print("2. Matrix Market")
    print("3. Exit program")
    print()


def create_datasets_dir():
    if not os.path.isdir(data_dir_path):
        os.mkdir(data_dir_path)
        print(f"{colors.color_green(data_dir_path)} directory created")
        print()


def set_env():
    os.environ["REUSEFILE"] = "1"
    os.environ["TMPFILE"] = file_name
    os.environ["SKIP_BFS"] = "1"


def unset_env():
    del os.environ["REUSEFILE"]
    del os.environ["TMPFILE"]
    del os.environ["SKIP_BFS"]


def generate_graph_500(scale, edge_factor):
    graph500_dir_path = "generators/graph500"
    if not os.listdir(graph500_dir_path):
        raise Exception(
            "graph 500 submodule is required: remember to run 'git submodule update --init --recursive'"
        )

    set_env()

    working_dir_path = f"{graph500_dir_path}/src"

    try:
        print(f"Start compiling graph500_reference_bfs...")
        print()
        subprocess.run(
            ["make", "graph500_reference_bfs"], cwd=working_dir_path, check=True
        )
        print()
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed: {e}")
        sys.exit(1)

    try:
        print(
            f"Start generating a graph with (scale, edge factor) = ({scale}, {edge_factor})..."
        )
        print()
        subprocess.run(
            ["./graph500_reference_bfs", f"{scale}", f"{edge_factor}"],
            cwd=working_dir_path,
            check=True,
        )
        print()
    except subprocess.CalledProcessError as e:
        print(f"Graph generation failed: {e}")
        sys.exit(1)

    create_datasets_dir()
    unset_env()

    print(f"{colors.GREEN}Graph generated in the {data_dir_path} folder{colors.RESET}")

    source_path = f"{working_dir_path}/{file_name}"
    destination_path = f"{data_dir_path}/{file_name}"

    shutil.move(source_path, destination_path)


def main():
    size = get_dataset_size()
    config = read_config_file()
    while True:
        show_sources_menu()
        choice = input("Enter your choice (1-4): ").strip()
        if choice == "1":
            print("generators option has been selected")
            show_generators_menu()
            choice = input("Enter your choice (1-3): ").strip()
            if choice == "1":
                (scale, edge_factor) = read_graph_500_config(config, size)
                print(scale, edge_factor)
                generate_graph_500(scale, edge_factor)
            print()
        elif choice == "2":
            print("suite sparce matrix list option has been selected")
        elif choice == "3":
            print("suite sparce matrix ssgetpy util has been selected")
        elif choice == "4":
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please enter a number from 1 to 4.")


if __name__ == "__main__":
    main()
