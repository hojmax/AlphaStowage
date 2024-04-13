import torch
from NeuralNetwork import NeuralNetwork
import time
from Logging import init_wandb_run, log_process_ps  # , log_memory_usage


def gpu_process(device, update_event, config, pipes):
    if config["train"]["log_wandb"]:
        init_wandb_run(config)

    model = NeuralNetwork(config, device).to(device)
    model.eval()
    bays = []
    flat_ts = []
    conns = []
    start_time = time.time()
    processed = 0
    avg_over = 1000
    i = 0

    with torch.no_grad():
        while True:
            if update_event.is_set():
                model.load_state_dict(
                    torch.load("shared_model.pt", map_location=model.device)
                )
                update_event.clear()

            i += 1
            if i % avg_over == 0:
                # log_memory_usage("gpu")
                processed_per_second = processed / (time.time() - start_time)
                log_process_ps(processed_per_second, config)
                processed = 0
                start_time = time.time()

            for parent_conn, _ in pipes:
                if not parent_conn.poll():
                    continue
                bay, flat_T = parent_conn.recv()
                bays.append(bay.clone())
                flat_ts.append(flat_T.clone())

                del bay, flat_T

                conns.append(parent_conn)

                if len(bays) < config["inference"]["batch_size"]:
                    continue

                processed += len(bays)
                bays = torch.cat(bays, dim=0)
                flat_ts = torch.stack(flat_ts)
                policies, values = model(bays.to(device), flat_ts.to(device))
                policies = policies.cpu()
                values = values.cpu()

                for conn, policy, value in zip(conns, policies, values):
                    conn.send((policy, value))

                del (bays, flat_ts, conns, policies, values)

                bays = []
                flat_ts = []
                conns = []
