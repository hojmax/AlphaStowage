import numpy as np
import torch
from MPSPEnv import Env
import os
import pandas as pd
from main import get_config
from EpisodePlayer import EpisodePlayer
from Node import TruncatedEpisodeError
import torch.multiprocessing as mp
from Train import PretrainedModel
from tqdm import tqdm
from GPUProcess import GPUProcess
from multiprocessing.connection import Connection
import time


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


class BenchmarkLogger:
    def __init__(self, env_queue, result_queue, count=0):
        self.env_queue = env_queue
        self.result_queue = result_queue
        self.count = count
        self.results = []

    def loop(self):
        start_time = time.time()
        try:
            while True:
                time.sleep(240)
                while not self.result_queue.empty():
                    self.results.append(self.result_queue.get())

                processed = len(self.results)
                # Print processed out of total both in numbers and in pct, and time elapsed and estimated time remaining
                elapsed = time.time() - start_time
                avg_time = elapsed / processed if processed > 0 else 0
                remaining = (self.count - processed) * avg_time
                hours = remaining // 3600
                minutes = (remaining % 3600) // 60
                seconds = remaining % 60
                print(
                    f"Processed {processed}/{self.count} ({processed/self.count:.2%}) in {elapsed:.2f}s. Estimated time remaining: {hours:.0f}h {minutes:.0f}m {seconds:.0f}s"
                )

                # Calculate the average for each N from 4 to 16
                for N in range(4, 17, 2):
                    results = [r["result"] for r in self.results if r["N"] == N]
                    if len(results) == 0:
                        continue
                    avg = np.mean(results)
                    print(f"N={N} - Avg: {avg:.2f}, count: {len(results)}")
        finally:
            print("Benchmarking done")
            df = pd.DataFrame(self.results)
            df.to_excel("benchmark_results.xlsx", index=False)


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


class InferenceProcess:
    def __init__(
        self,
        env_queue: mp.Queue,
        result_queue: mp.Queue,
        conn: Connection,
        config: dict,
    ) -> None:
        self.conn = conn
        self.config = config
        self.env_queue = env_queue
        self.result_queue = result_queue

    def loop(self):
        while True:

            e = self.env_queue.get()

            env = Env(
                e["R"],
                e["C"],
                e["N"],
                skip_last_port=True,
                take_first_action=True,
                strict_mask=True,
            )
            env.reset_to_transportation(e["transportation_matrix"])

            try:
                player = EpisodePlayer(env, self.conn, self.config, deterministic=False)
                _, _, reshuffles, _, _ = player.run_episode()
            except TruncatedEpisodeError:
                reshuffles = -10000
            finally:
                env.close()

            e["result"] = -reshuffles
            self.result_queue.put(e)


def start_process_loop(process_class, *args, **kwargs):
    process = process_class(*args, **kwargs)
    process.loop()


if __name__ == "__main__":
    config = get_config("local_config.json")

    pretrained = PretrainedModel(
        wandb_run=config["wandb"]["pretrained_run"],
        wandb_model=config["wandb"]["pretrained_model"],
        artifact=config["wandb"]["artifact"],
        local_model=config["wandb"].get("local_model"),
    )

    testset = get_benchmarking_data("benchmark/set_2")

    gpu_device = "cuda:0" if torch.cuda.is_available() else "mps"

    workers = 2
    inference_pipes = [mp.Pipe() for _ in range(workers)]

    env_queue = mp.Queue()

    for e in testset:
        env_queue.put(e)

    result_queue = mp.Queue()

    processes = [
        mp.Process(
            target=start_process_loop,
            args=(
                GPUProcess,
                inference_pipes,
                None,
                gpu_device,
                pretrained,
                config,
            ),
        ),
        mp.Process(
            target=start_process_loop,
            args=(BenchmarkLogger, env_queue, result_queue, len(testset)),
        ),
    ] + [
        mp.Process(
            target=start_process_loop,
            args=(InferenceProcess, env_queue, result_queue, conn, config),
        )
        for i, (_, conn) in enumerate(inference_pipes)
    ]

    # Start processes
    for p in processes:
        p.start()

    # Join processes
    for p in processes:
        p.join()
