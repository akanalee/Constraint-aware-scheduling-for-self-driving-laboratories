"""
Memory - 
episode
"""
import torch


class Memory:
    """Memory"""

    def __init__(self):
        self.adj_mb = []
        self.fea_mb = []
        self.candidate_mb = []
        self.mask_mb = []
        self.a_mb = []
        self.r_mb = []
        self.done_mb = []
        self.job_logprobs = []
        self.mch_logprobs = []
        self.mask_mch = []
        self.first_task = []
        self.pre_task = []
        self.action = []
        self.mch = []
        self.dur = []
        self.mch_time = []

    def clear_memory(self):
        """memory"""
        del self.adj_mb[:]
        del self.fea_mb[:]
        del self.candidate_mb[:]
        del self.mask_mb[:]
        del self.a_mb[:]
        del self.r_mb[:]
        del self.done_mb[:]
        del self.job_logprobs[:]
        del self.mch_logprobs[:]
        del self.mask_mch[:]
        del self.first_task[:]
        del self.pre_task[:]
        del self.action[:]
        del self.mch[:]
        del self.dur[:]
        del self.mch_time[:]


def adv_normalize(adv):
    """
     - 
    """
    std = adv.std()
    assert std != 0. and not torch.isnan(std), 'Need nonzero std'
    n_advs = (adv - adv.mean()) / (adv.std() + 1e-8)
    return n_advs