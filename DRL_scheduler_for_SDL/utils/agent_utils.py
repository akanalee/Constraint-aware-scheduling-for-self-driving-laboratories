"""
Agent Utils - 
action selectionevaluation
"""
import torch
from torch.distributions.categorical import Categorical


def select_action(p, candidate, memory, log_prob):
    """
    action - 
    Args:
        p:  [batch, n_candidates, 1]
        candidate: 
        memory: memory
        log_prob: log_prob
    """
    dist = Categorical(p.squeeze())
    s = dist.sample()
    if memory is not None:
        log_prob.append(dist.log_prob(s).cpu().tolist())

    action = []
    for i in range(s.size(0)):
        a = candidate[i][s[i]].cpu().tolist()
        action.append(a)
    return action, s


def select_action1(p, candidate):
    """
    actionlog_prob - batch_size=1
    :
        action: tensor [batch]
        s: sample indices [batch]
        log_a: log probabilities [batch]
    """
    dist = Categorical(p.squeeze())
    s = dist.sample()
    log_a = dist.log_prob(s)

    # s1D tensorbatch_size=1
    if s.dim() == 0:
        s = s.unsqueeze(0)
        log_a = log_a.unsqueeze(0)

    action = []
    for i in range(s.size(0)):
        a = candidate[i][s[i]]
        action.append(a)
    action = torch.stack(action, 0)

    return action, s, log_a


def select_action2(p):
    """
    action selection - batch_size=1
    
    """
    if torch.isnan(p).any():
        print(" [CRITICAL ERROR] NaN detected in machine selection!")
        print("    This should NOT happen after Smart Revert fix.")
        print("    Check env_adapter.py mask construction logic.")

        # uniformprocessing_times
        # 
        raise RuntimeError("NaN in machine probabilities - check mask construction!")

        # 
    dist = Categorical(p.squeeze())
    s = dist.sample()
    log_a = dist.log_prob(s)

    # s1D tensorbatch_size=1
    if s.dim() == 0:
        s = s.unsqueeze(0)
        log_a = log_a.unsqueeze(0)

    return s, log_a


def eval_actions(p, actions):
    """
    actions - 
    :
        log_probs: [batch]
        entropy: scalar
    """
    softmax_dist = Categorical(p)
    ret = softmax_dist.log_prob(actions).reshape(-1)
    entropy = softmax_dist.entropy().mean()
    return ret, entropy


def greedy_select_action(p, candidate):
    """Greedy op selection - action, index, log_a"""
    _, index = p.squeeze(-1).max(1) if p.dim() > 1 else p.squeeze(-1).max(0)

    if index.dim() == 0:
        index = index.unsqueeze(0)

    action = []
    for i in range(index.size(0)):
        a = candidate[i][index[i]]
        action.append(a)
    action = torch.stack(action, 0)

    return action, index  #  


def greedy_select_action2(p):
    """Greedy machine selection - """
    _, s = p.squeeze(-1).max(1) if p.dim() > 1 else p.squeeze(-1).max(0)

    if s.dim() == 0:
        s = s.unsqueeze(0)

    return s  #  


def sample_select_action(p, candidate):
    """
    Sample action for testing - batch_size=1
    """
    dist = Categorical(p.squeeze())
    s = dist.sample()

    # scandidate
    if s.dim() == 0:
        return candidate[0][s]  # batch_size=1
    else:
        return candidate[s]  # batch