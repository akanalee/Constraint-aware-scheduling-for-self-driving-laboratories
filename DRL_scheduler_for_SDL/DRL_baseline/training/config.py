"""MultiPPO config for the SDL environment (Lei Kun baseline)."""
from dataclasses import dataclass


@dataclass
class MultiPPOConfig:
    """MultiPPO config, sized for SDL."""

    # Environment scale.
    n_j: int = 50
    n_m: int = 9
    n_machines: int = 70

    # Network.
    num_layers: int = 3
    neighbor_pooling_type: str = "sum"
    input_dim: int = 2
    hidden_dim: int = 128
    num_mlp_layers_feature_extract: int = 2
    num_mlp_layers_actor: int = 3
    hidden_dim_actor: int = 128
    num_mlp_layers_critic: int = 3
    hidden_dim_critic: int = 128
    learn_eps: bool = False

    # PPO.
    lr: float = 1e-4
    gamma: float = 0.99
    k_epochs: int = 5
    eps_clip: float = 0.2

    vloss_coef: float = 0.5
    ploss_coef: float = 1.0
    entloss_coef: float = 0.01

    # Training.
    batch_size: int = 1
    num_episodes: int = 1000

    decayflag: bool = True
    decay_step_size: int = 100
    decay_ratio: float = 0.75

    et_normalize_coef: float = 100.0

    graph_pool_type: str = "average"

    device: str = "cuda"

    Init: bool = True

    save_dir: str = "./saved_models"
    save_interval: int = 20


def get_default_config():
    return MultiPPOConfig()
