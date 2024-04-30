import torch
from NeuralNetwork import NeuralNetwork
import time
from Logging import init_wandb_run, log_process_ps
import numpy as np
from Train import init_model

# import torch.multiprocessing as mp


# class GPUProcess:
#     def __init__(self, device, update_event, config, pipes):
#         self.device = device
#         self.update_event = update_event
#         self.config = config
#         self.pipes = pipes
#         self.model = NeuralNetwork(config, device).to(device)
#         self.model.eval()
#         self.reset_queue()

#     def reset_queue(self) -> None:
#         self.bays = []
#         self.flat_ts = []
#         self.conns = []

#     def check_for_model_update(self) -> None:
#         if self.update_event.is_set():
#             self.model.load_state_dict(
#                 torch.load("shared_model.pt", map_location=self.model.device)
#             )
#             self.update_event.clear()

#     def receive_data(self) -> None:
#         for parent_conn, _ in self.pipes:
#             if not parent_conn.poll():
#                 continue
#             bay, flat_T = parent_conn.recv()
#             self.bays.append(bay)
#             self.flat_ts.append(flat_T)
#             self.conns.append(parent_conn)


def gpu_process(pretrained, device, update_event, config, pipes):
    if config["train"]["log_wandb"]:
        init_wandb_run(config)

    model = init_model(config, device, pretrained)
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
                processed_per_second = processed / (time.time() - start_time)
                log_process_ps(processed_per_second, config)
                processed = 0
                start_time = time.time()

            for parent_conn, _ in pipes:
                if not parent_conn.poll():
                    continue
                bay, flat_T = parent_conn.recv()
                bays.append(bay.copy())
                flat_ts.append(flat_T.copy())

                del bay, flat_T

                conns.append(parent_conn)

            if len(bays) < config["inference"]["batch_size"]:
                continue

            processed += len(bays)
            bays = torch.from_numpy(np.concatenate(bays, axis=0)).to(device)
            flat_ts = torch.from_numpy(np.stack(flat_ts)).to(device)
            policies, values = model(bays.to(device), flat_ts.to(device))
            policies = policies.cpu()
            values = values.cpu()

            for conn, policy, value in zip(conns, policies, values):
                conn.send(
                    (
                        torch.Tensor.numpy(policy, force=True).copy(),
                        torch.Tensor.numpy(value, force=True).copy(),
                    )
                )

            del (bays, flat_ts, conns, policies, values)

            bays = []
            flat_ts = []
            conns = []
