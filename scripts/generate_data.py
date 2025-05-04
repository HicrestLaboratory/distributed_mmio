import os
import sys
import subprocess
import shutil
import platform

import colors

data_dir_path = "data"
file_name = "graph_binary"

def show_menu():
    print()
    print("Choose the kind of data to generate:")
    print()
    print("1. graph500")
    print("2. Matrix Market")
    print("3. Exit")
    print()


def create_data_dir():
    if not os.path.isdir(data_dir_path):
        os.mkdir(data_dir_path)
        print("'data' directory created")


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
        raise Exception("graph 500 submodule is required: remember to run 'git submodule update --init --recursive'")


    set_env()

    working_dir_path = f"{graph500_dir_path}/src"

    # for {platform.uname()}
    try:
        print(f"Start compiling graph500_reference_bfs...")
        print()
        subprocess.run(["arch", "-arm64", "make", "graph500_reference_bfs"], cwd=working_dir_path,check=True)
        print()
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed: {e}")
        sys.exit(1)

    try:
        print(f"Start generating a graph with (scale, edge factor) = ({scale}, {edge_factor})...")
        print()
        subprocess.run(["arch", "-arm64", "./graph500_reference_bfs", scale, edge_factor], cwd=working_dir_path,check=True)
        print()
    except subprocess.CalledProcessError as e:
        print(f"Graph generation failed: {e}")
        sys.exit(1)

    create_data_dir()
    unset_env()

    print(f"{colors.GREEN}Graph generated in the data folder{colors.RESET}")

    source_path = f"{working_dir_path}/{file_name}"
    destination_path = f"{data_dir_path}/{file_name}"

    shutil.move(source_path, destination_path)


def main():
    while True:
        show_menu()
        choice = input("Enter your choice (1-2): ").strip()
        
        if choice == "1":
            print("graph500 generation has been selected")
            scale = input("Enter the scale of the problem: ").strip()
            edge_factor = input("Enter the edge factor of the problem: ").strip()
            print()
            generate_graph_500(scale, edge_factor)
        elif choice == "2":
            print("Generating text dataset...")
            # call yourv dataset generation logic here
        elif choice == "3":
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please enter a number from 1 to 3.")

if __name__ == "__main__":
    main()
    