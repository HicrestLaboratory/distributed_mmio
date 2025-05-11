import utils
import graph500_generator

# TODO:
# 2. Look at the include folder
# 3. Add generator for Matrix Market
# 4. Writhe the README file


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


def show_sizes_menu():
    print()
    print("Choose the size of data to produce:")
    print()
    print("1. small")
    print("2. large")
    print("3. Exit program")
    print()


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


def main():
    while True:
        size = get_dataset_size()
        config = utils.read_config_file()
        show_sources_menu()
        choice = input("Enter your choice (1-4): ").strip()
        if choice == "1":
            print("generators option has been selected")
            show_generators_menu()
            choice = input("Enter your choice (1-3): ").strip()
            if choice == "1":
                graph500_generator.generate(config, size)
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
