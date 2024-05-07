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
        self.bays = None
        self.flat_ts = None
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
            if self.bays is None:
                self.bays = bay
            else:
                self.bays = torch.cat((self.bays, bay), dim=0)

            if self.flat_ts is None:
                self.flat_ts = flat_T
            else:
                self.flat_ts = torch.cat((self.flat_ts, flat_T), dim=0)

            self.conns.append(parent_conn)

    def _queue_is_full(self) -> bool:
        if self.bays is None or self.flat_ts is None:
            return False
        return self.bays.shape[0] >= self.config["inference"]["batch_size"]

    def _process_data(self) -> None:

        with torch.no_grad():
            bays = self.bays.to(self.device)
            flat_ts = self.flat_ts.to(self.device)
            policies, values = self.model(bays, flat_ts)
            policies = policies.detach().cpu()
            values = values.detach().cpu()

        return policies, values

    def _send_data(self, policies, values):
        for conn, policy, value in zip(self.conns, policies, values):
            conn.send(
                (
                    policy,
                    value,
                )
            )
