import numpy as np
import torch
from MPSPEnv import Env
import os
import pandas as pd
from NeuralNetwork import NeuralNetwork
from main import get_config
from EpisodePlayer import EpisodePlayer
import torch.multiprocessing as mp
from tqdm import tqdm


def get_benchmarking_data(path):
    """Go through all files in the directory and return a list of dictionaries with:
    N, R, C, seed, transportation_matrix, paper_result"""

    output = []
    df = pd.read_excel(
        os.path.join(path, "paper_results.xlsx"),
    )

    for file in os.listdir(path):
        if file.endswith(".txt"):
            with open(os.path.join(path, file), "r") as f:
                lines = f.readlines()
                N = int(lines[0].split(": ")[1])
                R = int(lines[1].split(": ")[1])
                C = int(lines[2].split(": ")[1])
                seed = int(lines[3].split(": ")[1])

                paper_result = df[
                    (df["N"] == N)
                    & (df["R"] == R)
                    & (df["C"] == C)
                    & (df["seed"] == seed)
                ]["res"].values

                assert len(paper_result) == 1
                paper_result = paper_result[0]

                output.append(
                    {
                        "N": N,
                        "R": R,
                        "C": C,
                        "seed": seed,
                        "transportation_matrix": np.loadtxt(lines[4:], dtype=np.int32),
                        "paper_result": paper_result,
                    }
                )

    return output


def transform_benchmarking_data(data):
    testset = []
    for instance in data:
        env = Env(
            instance["R"],
            instance["C"],
            instance["N"],
            skip_last_port=True,
            take_first_action=True,
            strict_mask=True,
        )
        env.reset_to_transportation(instance["transportation_matrix"])
        testset.append(env)
    return testset


def gpu_process(model, device, conn):

    model.eval()
    model.to(device)

    with torch.no_grad():
        while True:

            if not conn.poll():
                continue

            bay, flat_T = conn.recv()

            policy, value = model(
                torch.from_numpy(bay).to(device), torch.from_numpy(flat_T).to(device)
            )
            policy = policy.cpu()[0]
            value = value.cpu()[0]

            conn.send(
                (
                    torch.Tensor.numpy(policy, force=True).copy(),
                    torch.Tensor.numpy(value, force=True).copy(),
                )
            )


if __name__ == "__main__":
    config = get_config("config.json")
    model = NeuralNetwork(config=config, device="cpu")
    model.load_state_dict(torch.load("bigrun6.pt", map_location="cpu"))
    testset = get_benchmarking_data("benchmark/set_2")

    gpu_conn, cpu_conn = mp.Pipe()
    gpu_device = "cuda:1" if torch.cuda.is_available() else "mps"

    mp.Process(target=gpu_process, args=(model, gpu_device, gpu_conn)).start()
    for n in range(6, 17, 2):

        sub_testset = [e for e in testset if e["N"] == n]
        scores = []

        for e in tqdm(sub_testset):
            N = e["N"]
            R = e["R"]
            C = e["C"]
            seed = e["seed"]
            transportation_matrix = e["transportation_matrix"]

            env = Env(
                R,
                C,
                N,
                skip_last_port=True,
                take_first_action=True,
                strict_mask=True,
            )
            env.reset_to_transportation(transportation_matrix)
            player = EpisodePlayer(env, cpu_conn, config, deterministic=False)
            _, _, reshuffles, _ = player.run_episode()
            e["result"] = reshuffles
            scores.append(reshuffles)

        print(f"Average reshuffles for N={n}: {np.mean(scores)}")

        df = pd.DataFrame(testset)
        df.to_excel("results.xlsx")
