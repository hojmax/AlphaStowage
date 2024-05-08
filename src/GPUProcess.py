from Train import init_model
import numpy as np
import torch
from Train import PretrainedModel
import torch.multiprocessing as mp
from typing import Union


class GPUProcess:
    def __init__(
        self,
        pipes: list,
        update_event: Union[mp.Event, None],
        device: torch.device,
        pretrained: PretrainedModel,
        config: dict,
    ) -> None:

        print("GPUProcess")
        self.device = device
        self.update_event = update_event
        self.config = config
        self.pipes = pipes
        self.model = init_model(config, device, pretrained)
        self.model.eval()
        self._reset_queue()

    def loop(self):
        with torch.no_grad():
            while True:
                if self.update_event is not None:
                    self._pull_model_update()
                self._receive_data()

                if self._queue_is_full():
                    policies, values = self._process_data()
                    self._send_data(policies, values)
                    self._reset_queue()

    def _reset_queue(self) -> None:
        self.bays = []
        self.flat_ts = []
        self.conns = []

    def _pull_model_update(self) -> None:
        if self.update_event.is_set():
            self.model.load_state_dict(
                torch.load("shared_model.pt", map_location=self.model.device)
            )
            self.update_event.clear()

    def _receive_data(self) -> None:
        for parent_conn, _ in self.pipes:
            if not parent_conn.poll():
                continue
            bay, flat_T = parent_conn.recv()
            self.bays.append(bay)
            self.flat_ts.append(flat_T)
            self.conns.append(parent_conn)

    def _queue_is_full(self) -> bool:
        return len(self.bays) >= self.config["inference"]["batch_size"]

    def _process_bays(self):
        bays = np.stack(self.bays)
        bays = torch.tensor(bays)
        bays = bays.unsqueeze(1)  # Add channel dimension
        bays = bays.to(self.device)
        return bays

    def _process_flat_ts(self):
        flat_ts = np.stack(self.flat_ts)
        flat_ts = torch.tensor(flat_ts)
        flat_ts = flat_ts.to(self.device)
        return flat_ts

    def _process_data(self) -> None:
        bays = self._process_bays()
        flat_ts = self._process_flat_ts()
        with torch.no_grad():
            policies, values, _ = self.model(bays, flat_ts)
            policies = policies.detach().cpu().numpy()
            values = values.detach().cpu().numpy()

        return policies, values

    def _send_data(self, policies, values):
        for conn, policy, value in zip(self.conns, policies, values):
            conn.send(
                (
                    policy,
                    value,
                )
            )
