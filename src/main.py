from train import get_config
from alpha_zero import AlphaZero
import sys
import json
import ray


if __name__ == "__main__":

    with open("config.json", "r") as f:
        config = json.load(f)

    alpha_zero = AlphaZero(config)

    if len(sys.argv) == 2:
        num_workers = sys.argv[1]
        try:
            num_workers = int(num_workers)
        except ValueError:
            print("Invalid number of workers")
            sys.exit(1)
        alpha_zero.train(num_workers)
    else:
        print("\nWelcome to AlphaZero! Here's a list of options:")

        while True:
            # Configure running options
            options = [
                "Train",
                "Self-play",
                "Load pretrained model",
                "Exit",
            ]
            print()
            for i in range(len(options)):
                print(f"{i}. {options[i]}")

            print()

            choice = input("Enter a number to choose an action: ")
            valid_inputs = [str(i) for i in range(len(options))]
            while choice not in valid_inputs:
                choice = input("Invalid input, enter a number listed above: ")
            choice = int(choice)
            if choice == 0:
                num_workers = input("Enter the number of workers: ")
                while not num_workers.isdigit():
                    num_workers = input("Invalid input, enter a number: ")

                num_workers = int(num_workers)
                alpha_zero.train(num_workers)
            elif choice == 1:
                num_workers = input("Enter the number of workers: ")
                while not num_workers.isdigit():
                    num_workers = input("Invalid input, enter a number: ")

                num_workers = int(num_workers)
                alpha_zero.self_play(num_workers)

            print("\nDone")

    ray.shutdown()
