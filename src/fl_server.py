# src/fl_server.py
import flwr as fl
from typing import Dict, Callable
import argparse

def weighted_average(metrics):
    # optional: funcție personalizată de agregare dacă vrei
    pass

def main(server_address="0.0.0.0:8080", rounds=5):
    # Strategy: FedAvg (default)
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,    # all clients
        fraction_eval=1.0,
        min_fit_clients=3,
        min_eval_clients=3,
        min_available_clients=3,
    )
    fl.server.start_server(server_address=server_address, config={"num_rounds": rounds}, strategy=strategy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--address', default="0.0.0.0:8080")
    parser.add_argument('--rounds', type=int, default=5)
    args = parser.parse_args()
    main(args.address, args.rounds)
