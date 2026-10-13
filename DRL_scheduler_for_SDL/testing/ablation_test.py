import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
import yaml

_testing_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(_testing_dir)
repo_root = os.path.dirname(project_root)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'training'))
sys.path.insert(0, _testing_dir)

import e2e_comparison as e2e
from ablation_components import (
    AblationAdapterNoBackward,
    AblationAdapterNoForward,
    AblationAdapterNoUrgency,
)
from config import get_default_config
from env_adapter import SDLEnvAdapter
from envs.sdl_env import SDLEnv
from ppo import PPO


class AblationComparison(e2e.BenchmarkComparison):
    GROUPS = ['erc_full', 'erc_no_forward', 'erc_no_backward', 'erc_no_urgency']

    GROUP_ADAPTER = {
        'erc_full':        SDLEnvAdapter,
        'erc_no_forward':  AblationAdapterNoForward,
        'erc_no_backward': AblationAdapterNoBackward,
        'erc_no_urgency':  AblationAdapterNoUrgency,
    }

    def __init__(self, env_config_path, model_paths, output_dir='./outputs/ablation',
                 ga_pop_size=20, ga_patience=20):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        with open(env_config_path, 'r', encoding='utf-8') as f:
            self.env_config_raw = yaml.safe_load(f)

        from configs import Config
        self.env_config = Config(self.env_config_raw)

        self.env = SDLEnv(self.env_config)

        self.ppo_config = get_default_config()
        self.ppo_config.n_machines = len(self.env.machines)
        self.ppo_config.n_j = self.env_config_raw.get('env_config', {}).get('n_jobs', 80)
        self.ppo_config.device = 'cpu'
        self.device = torch.device('cpu')

        self.ga_pop_size = ga_pop_size
        self.ga_patience = ga_patience

        self.ppo_baseline = None
        self.baseline_ppo_config = None

        self.group_ppo = {}
        for g in self.GROUPS:
            self.group_ppo[g] = self._load_ppo_for_group(g, model_paths[g])

        self.group_specs = {
            g: (self.GROUP_ADAPTER[g], self.group_ppo[g]) for g in self.GROUPS
        }

        self.ppo = self.group_ppo['erc_full']
        self.adapter = SDLEnvAdapter(self.env, self.ppo_config)

        print(f"\n{'#' * 60}")
        print(f"ERC-FSS component ablation initialized")
        print(f"  Machines: {len(self.env.machines)}")
        print(f"  Output Dir: {self.output_dir}")
        print(f"  Variants: {', '.join(self.GROUPS)}")
        print(f"{'#' * 60}\n")

    def _load_ppo_for_group(self, group, model_path):
        print(f"Loading [{group}] model from: {model_path}")
        ppo = PPO(self.ppo_config)
        checkpoint = torch.load(model_path, map_location='cpu')
        ppo.policy_job.load_state_dict(checkpoint['job_actor'])
        ppo.policy_mch.load_state_dict(checkpoint['mch_actor'])
        ppo.policy_old_job.load_state_dict(checkpoint['job_actor'])
        ppo.policy_old_mch.load_state_dict(checkpoint['mch_actor'])
        ppo.policy_job.eval()
        ppo.policy_old_job.eval()
        ppo.policy_mch.eval()
        ppo.policy_old_mch.eval()
        return ppo

    def run_episode_with_method(self, method='erc_full', seed=None,
                                ga_pop_size=20, ga_patience=20,
                                pregenerated_jobs=None, bo_delays=None):
        adapter_cls, ppo = self.group_specs[method]

        saved_adapter_cls = e2e.SDLEnvAdapter
        saved_ppo = self.ppo
        e2e.SDLEnvAdapter = adapter_cls
        self.ppo = ppo
        try:
            result = super().run_episode_with_method(
                'ERC-FSS', seed,
                ga_pop_size=ga_pop_size, ga_patience=ga_patience,
                pregenerated_jobs=pregenerated_jobs, bo_delays=bo_delays
            )
        finally:
            e2e.SDLEnvAdapter = saved_adapter_cls
            self.ppo = saved_ppo
        result['runtime_breakdown'].pop('fss_detail', None)
        return result

    def run_comparison(self, num_episodes=1, seeds=None, methods=None,
                       ga_pop_size=20, ga_patience=20):
        groups = list(methods) if methods else list(self.GROUPS)
        if seeds is None:
            seeds = [int(np.random.randint(0, 400000000)) for _ in range(num_episodes)]

        print(f"\n{'#' * 60}")
        print(f"Starting ERC-FSS component ablation")
        print(f"  Episodes: {len(seeds)}")
        print(f"{'#' * 60}\n")

        all_results = []

        for episode_idx, seed in enumerate(seeds):
            print(f"\n{'=' * 80}")
            print(f"Episode {episode_idx + 1}/{len(seeds)} - Seed: {seed}")
            print(f"{'=' * 80}")

            episode_results = {
                'episode_idx': episode_idx,
                'seed': seed,
                'timestamp': datetime.now().isoformat()
            }

            pregenerated_jobs, bo_delays = self._pre_generate_jobs(seed, n_jobs=80)

            for group in groups:
                print(f"\n{'-' * 60}")
                print(f"Running ablation variant: {group}  "
                      f"(adapter={self.GROUP_ADAPTER[group].__name__})")
                print(f"{'-' * 60}")
                try:
                    result = self.run_episode_with_method(
                        group, seed,
                        ga_pop_size=ga_pop_size, ga_patience=ga_patience,
                        pregenerated_jobs=pregenerated_jobs,
                        bo_delays=bo_delays
                    )
                    episode_results[group] = result
                except Exception as e:
                    print(f"{group} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    episode_results[group] = {
                        'error': str(e),
                        'makespan': float('inf')
                    }

            all_results.append(episode_results)
            self._save_episode_result(episode_results, episode_idx)

        from testing.summarize_results import summarize_ablation, print_ablation_summary
        summary = summarize_ablation(all_results)
        filepath = os.path.join(self.output_dir, 'summary.json')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print_ablation_summary(summary)
        print(f"\n Summary saved to: {filepath}")


def paper_trial_seeds():
    seeds = []
    for k in range(1, 11):
        path = os.path.join(repo_root, 'results', 'ablation', f'trial_{k}.json')
        with open(path, 'r', encoding='utf-8') as f:
            seeds.append(int(json.load(f)['seed']))
    return seeds


def main():
    import argparse

    parser = argparse.ArgumentParser(description='ERC-FSS component ablation on the SDL benchmark')
    parser.add_argument('--config', type=str,
                        default=os.path.join(project_root, 'configs', 'sdl_config.yaml'),
                        help='Environment config path')
    parser.add_argument('--model_full', type=str,
                        default=os.path.join(project_root, 'training', 'saved_models', 'ERC_FSS.pt'))
    parser.add_argument('--model_no_forward', type=str,
                        default=os.path.join(project_root, 'saved_models_ablation_no_forward', 'noforward.pt'))
    parser.add_argument('--model_no_backward', type=str,
                        default=os.path.join(project_root, 'saved_models_ablation_no_backward', 'nobackward.pt'))
    parser.add_argument('--model_no_urgency', type=str,
                        default=os.path.join(project_root, 'saved_models_ablation_no_urgency', 'nourgency.pt'))
    parser.add_argument('--num_episodes', type=int, default=1,
                        help='Number of scheduling trials with random seeds')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Run the trials with these seeds')
    parser.add_argument('--paper_seeds', action='store_true',
                        help='Run the ten trials reported in the paper '
                             '(seeds read from results/ablation)')
    parser.add_argument('--variants', type=str, nargs='+', default=AblationComparison.GROUPS,
                        choices=AblationComparison.GROUPS, help='Variants to run')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(repo_root, 'outputs', 'ablation'),
                        help='Output directory for results')

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Config not found: {args.config}")
        return

    model_paths = {
        'erc_full':        args.model_full,
        'erc_no_forward':  args.model_no_forward,
        'erc_no_backward': args.model_no_backward,
        'erc_no_urgency':  args.model_no_urgency,
    }
    missing = [(g, p) for g, p in model_paths.items() if not os.path.exists(p)]
    if missing:
        print("Missing model checkpoint(s):")
        for g, p in missing:
            print(f"   {g:16s}: {p}")
        return

    seeds = args.seeds
    if args.paper_seeds:
        seeds = paper_trial_seeds()

    runner = AblationComparison(
        env_config_path=args.config,
        model_paths=model_paths,
        output_dir=args.output_dir,
    )
    runner.run_comparison(
        num_episodes=args.num_episodes,
        seeds=seeds,
        methods=args.variants,
    )


if __name__ == "__main__":
    main()
