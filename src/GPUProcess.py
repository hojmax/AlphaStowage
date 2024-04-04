import torch
from NeuralNetwork import NeuralNetwork


def gpu_process(device, update_event, config, pipes):
    model = NeuralNetwork(config, device).to(device)
    model.eval()
    bays = []
    flat_ts = []
    conns = []

    with torch.no_grad():
        while True:
            if update_event.is_set():
                model.load_state_dict(
                    torch.load("shared_model.pt", map_location=model.device)
                )
                update_event.clear()

            for parent_conn, _ in pipes:
                if not parent_conn.poll():
                    continue
                bay, flat_T = parent_conn.recv()
                bays.append(bay)
                flat_ts.append(flat_T)
                conns.append(parent_conn)

                if len(bays) < config["inference"]["batch_size"]:
                    continue

                bays = torch.cat(bays, dim=0)
                flat_ts = torch.stack(flat_ts)
                policies, values = model(bays.to(device), flat_ts.to(device))
                policies = policies.cpu()
                values = values.cpu()

                for conn, policy, value in zip(conns, policies, values):
                    conn.send((policy, value))

                bays = []
                flat_ts = []
                conns = []
