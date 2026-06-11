"""
 - SDLFSS
"""
from dataclasses import dataclass


@dataclass
class FSSConfig:
    """FSS - SDL"""

    #  (SDL)
    n_j: int = 40  # job (sdl_config.yaml)
    n_m: int = 9  # operation per job (7 + 2buffer)
    n_machines: int = 36  #  (SDL100+)

    #
    num_layers: int = 3  # GIN
    neighbor_pooling_type: str = "sum"  # sum/average/max
    input_dim: int = 5  # [is_scheduled, est_start/α, min_proc/α, urgency, n_compat/10]
    hidden_dim: int = 128  # 
    num_mlp_layers_feature_extract: int = 2  # feature extraction MLP
    num_mlp_layers_actor: int = 3  # actor MLP
    hidden_dim_actor: int = 128  # actor
    num_mlp_layers_critic: int = 3  # critic MLP
    hidden_dim_critic: int = 128  # critic
    learn_eps: bool = False  # GINepsilon

    # PPO
    lr: float = 3e-5  #
    gamma: float = 0.99  # 
    k_epochs: int = 5  # epoch
    eps_clip: float = 0.2  # PPO clip

    # 
    vloss_coef: float = 0.5  # value loss
    ploss_coef: float = 1.0  # policy loss
    entloss_coef: float = 0.01  # entropy loss

    # 
    batch_size: int = 1  # episode ()
    num_episodes: int = 1000  # episode

    # 
    decayflag: bool = True  # 
    decay_step_size: int = 100  #
    decay_ratio: float = 0.75  #

    # 
    et_normalize_coef: float = 100.0  # 

    # 
    graph_pool_type: str = "average"  # average/sum

    # Device
    device: str = "cuda"  # cuda/cpu

    # 
    Init: bool = True  # 

    # 
    save_dir: str = "./saved_models"  # 
    save_interval: int = 20  # 


def get_default_config():
    """"""
    return FSSConfig()