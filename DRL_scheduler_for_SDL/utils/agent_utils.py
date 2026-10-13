import torch
from torch.distributions.categorical import Categorical


def select_action(p, candidate, memory, log_prob):
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
    dist = Categorical(p.squeeze())
    s = dist.sample()
    log_a = dist.log_prob(s)

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
    if torch.isnan(p).any():
        print(" [CRITICAL ERROR] NaN detected in machine selection!")
        print("    This should NOT happen after Smart Revert fix.")
        print("    Check env_adapter.py mask construction logic.")

        raise RuntimeError("NaN in machine probabilities - check mask construction!")

    dist = Categorical(p.squeeze())
    s = dist.sample()
    log_a = dist.log_prob(s)

    if s.dim() == 0:
        s = s.unsqueeze(0)
        log_a = log_a.unsqueeze(0)

    return s, log_a


def eval_actions(p, actions):
    softmax_dist = Categorical(p)
    ret = softmax_dist.log_prob(actions).reshape(-1)
    entropy = softmax_dist.entropy().mean()
    return ret, entropy


def greedy_select_action(p, candidate):
    _, index = p.squeeze(-1).max(1) if p.dim() > 1 else p.squeeze(-1).max(0)

    if index.dim() == 0:
        index = index.unsqueeze(0)

    action = []
    for i in range(index.size(0)):
        a = candidate[i][index[i]]
        action.append(a)
    action = torch.stack(action, 0)

    return action, index


def greedy_select_action2(p):
    _, s = p.squeeze(-1).max(1) if p.dim() > 1 else p.squeeze(-1).max(0)

    if s.dim() == 0:
        s = s.unsqueeze(0)

    return s


def sample_select_action(p, candidate):
    dist = Categorical(p.squeeze())
    s = dist.sample()

    if s.dim() == 0:
        return candidate[0][s]
    else:
        return candidate[s]
