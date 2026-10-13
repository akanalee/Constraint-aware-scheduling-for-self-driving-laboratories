# Constraint-Aware Orchestration for Reliable Closed-Loop Discovery in Self-Driving Laboratories

Implementation of ERC-FSS (Experiment Release Controller with Flexible-Scaling Scheduler), the SDL benchmark, the ChemOS integration components and the result files.

## Repository structure

```
.
├── DRL_scheduler_for_SDL/
│   ├── configs/                 benchmark and case-study configurations
│   ├── envs/                    SDL environment
│   ├── models/                  FSS network
│   ├── training/                FSS training, ablation variants, trained models
│   ├── DRL_baseline/            conventional DRL baseline
│   ├── saved_models_ablation_*/ trained ablation models
│   ├── case_study/              experiment generation, reaction emulator wrappers
│   ├── utils/                   Gurobi feasibility check, runtime profiler, utilities
│   ├── testing/                 benchmark, ablation, statistics
│   └── plotting/                Gantt charts
├── chemos_integration/          ChemOS components, case-study drivers and figures
├── results/                     result files
│   ├── benchmark/               SDL benchmark, 10 trials
│   ├── ablation/                component ablation, 10 trials
│   ├── chemos_validation/       ChemOS-integrated validation, 5 trials
│   ├── case_study_alkox/        benzyl alcohol oxidation case study
│   └── case_study_suzuki/       Suzuki-Miyaura cross-coupling case study
├── auto_processing.py           ERC-FSS controller for the ChemOS-integrated validation
├── requirements.txt
└── environment.yml
```

New runs and figures are written to `outputs/`.

## Installation

Python 3.9:

```bash
conda env create -f environment.yml
conda activate erc-fss
```

or

```bash
pip install -r requirements.txt
```

A full Gurobi license is required (free academic licenses are available from Gurobi).

All commands are run from the repository root unless stated otherwise.

## Configuration

`DRL_scheduler_for_SDL/configs/sdl_config.yaml` defines the SDL benchmark:

- `env_config`: number of experiments (`n_jobs`) and T_max constraints `{from_step, to_step, max_interval}` (minutes);
- `process_steps`: instruments (`id`, `type`, `capacity`) and processing time (`base`, `std`, minutes) of each stage;
- `job_types`: operation sequence of an experiment;
- `atlas_config`: parameter space of the generated experiments.

`sdl_config_alkox.yaml` and `sdl_config_suzuki.yaml` in the same folder define the two case studies. PPO training settings are in `DRL_scheduler_for_SDL/training/config.py`.

Environment variables:

| Variable | Meaning |
|---|---|
| `CHEMOS_COMM_DIR` | communication directory shared by ChemOS and the ERC-FSS controller (default `chemos_comm/` in the repository root) |
| `CHEMOS_TIME_SCALE` | polling interval of the ChemOS schedule executor in real seconds per virtual second (default 0.01) |
| `SLACK_TOKEN` | Slack bot token |
| `OLYMPUS_SRC` | path to the `src` folder of Olympus |

## Trained models

| File | Used for |
|---|---|
| `DRL_scheduler_for_SDL/training/saved_models/ERC_FSS.pt` | SDL benchmark, ChemOS-integrated validation |
| `DRL_scheduler_for_SDL/training/saved_models/ALKOX.pt` | benzyl alcohol oxidation case study |
| `DRL_scheduler_for_SDL/training/saved_models/SUZUKI.pt` | Suzuki-Miyaura cross-coupling case study |
| `DRL_scheduler_for_SDL/DRL_baseline/training/saved_models/DRL_BASELINE.pt` | conventional DRL baseline |
| `DRL_scheduler_for_SDL/saved_models_ablation_no_backward/nobackward.pt` | ablation without backward deadline validation |
| `DRL_scheduler_for_SDL/saved_models_ablation_no_forward/noforward.pt` | ablation without forward feasibility check |
| `DRL_scheduler_for_SDL/saved_models_ablation_no_urgency/nourgency.pt` | ablation without constraint-induced urgency feature |

## Training

```bash
cd DRL_scheduler_for_SDL
python training/train.py
python training/train_ablation_no_backward.py
python training/train_ablation_no_forward.py
python training/train_ablation_no_urgency.py
python DRL_baseline/training/train_baseline.py
```

Checkpoints (`model_ep<N>.pt`) are written to `DRL_scheduler_for_SDL/saved_models/`, `DRL_scheduler_for_SDL/saved_models_ablation_*/` and `DRL_scheduler_for_SDL/DRL_baseline/training/saved_models/`.

## SDL benchmark

```bash
cd DRL_scheduler_for_SDL
python testing/e2e_comparison.py --paper_seeds                    # seeds of the trials in results/benchmark
python testing/e2e_comparison.py --num_episodes 10                # random seeds
python testing/e2e_comparison.py --seeds 93914750 99012345        # given seeds
python testing/e2e_comparison.py --paper_seeds --methods ERC-FSS MILP
```

Options: `--methods` (`ERC-FSS`, `Conventional DRL`, `GA`, `MILP`, `SPT`, `ChemOS Built-in Dispatch`), `--output_dir` (default `outputs/benchmark`), `--model`, `--baseline_model`, `--config`. One file `episode_<idx>_seed_<seed>.json` is written per trial.

Statistics, including the computation-time breakdown:

```bash
python testing/summarize_results.py benchmark ../outputs/benchmark
```

Add `--save summary.json` to save the summary.

### Ablation

```bash
cd DRL_scheduler_for_SDL
python testing/ablation_test.py --paper_seeds
python testing/summarize_results.py ablation ../outputs/ablation
```

### Gantt charts

```bash
cd DRL_scheduler_for_SDL
python plotting/convert_static_results.py ../results/benchmark/trial_1.json
```

The charts are written to `outputs/figures/`.

## Result files

| Folder in `results/` | Content | Statistics (run in `DRL_scheduler_for_SDL/`) |
|---|---|---|
| `benchmark/` | SDL benchmark, 10 trials | `python testing/summarize_results.py benchmark` |
| `ablation/` | component ablation, 10 trials | `python testing/summarize_results.py ablation` |
| `chemos_validation/` | ChemOS-integrated validation, 5 trials | `python testing/summarize_results.py chemos` |
| `case_study_alkox/` | benzyl alcohol oxidation case study | `python testing/summarize_results.py alkox` |
| `case_study_suzuki/` | Suzuki-Miyaura cross-coupling case study | `python testing/summarize_results.py suzuki` |

Figures:

```bash
cd DRL_scheduler_for_SDL
python plotting/convert_static_results.py ../results/benchmark/trial_1.json
python plotting/plot_gantt_chemos.py --json ../results/chemos_validation/trial_1.json
cd ..
python chemos_integration/plot_alkox_case_study.py
python chemos_integration/plot_suzuki_case_study.py
```

The figures are written to `outputs/figures/`.

## ChemOS integration

ChemOS and the ERC-FSS controller run as two processes and exchange files through `chemos_comm/` in the repository root.

### 1. Get ChemOS

```bash
git clone https://github.com/aspuru-guzik-group/ChemOS.git
```

Run ChemOS in the environment of this repository.

### 2. Modify ChemOS

ChemOS uses tabs for indentation; keep them when editing.

**`chemOS.py`**

(a) Add after `from Utilities.misc import Printer`:

```python
from chemos_integration.chemos_executor import ScheduleExecutor
from chemos_integration.file_protocol import init_comm_dirs, set_chemos_ready, is_episode_complete
```

(b) At the end of `ChemOS.__init__`, after `self.analyzer = Analyzer(settings = self.SETTINGS)`, add:

```python
		self.schedule_executor = ScheduleExecutor()
		self.drl_mode = False


	def start_schedule_executor(self):
		init_comm_dirs()
		set_chemos_ready()
		self.schedule_executor.start()
		self.drl_mode = True
```

(c) In `ChemOS._run`, change the start-request branch to:

```python
			if request['kind'] == 'start':
				self.request_handler.process_request(request)
				if not self.drl_mode:
					self.start_schedule_executor()
```

and add directly after the loop that pops `self.communicator.RECEIVED_REQUESTS`:

```python
		if self.drl_mode:
			if is_episode_complete():
				self._print('scheduling episode completed')
				self.drl_mode = False
			return
```

(d) In `ChemOS.run`, add `self.schedule_executor.stop()` after `self.bot_manager.shutdown()` in the `except (KeyboardInterrupt, SystemExit)` branch.

**`Communicator/communicator.py`**

Replace the ChatBot import with:

```python
from chemos_integration.command_bot import CommandBot as Bot
```

A message containing `experiment`, for example `run experiment`, starts a run.

**`Utilities/defaults.py`**

```python
'communicator': {'type': 'slack', 'port': 8080, 'channel_id': '<CHANNEL_ID>', 'bot_id': '<BOT_USER_ID>'},
```

**`Communicator/SlackInterface/slack.py`** (Windows)

Add `import time` to the imports and replace `SlackCommunicator._parse_message` with:

```python
	def _parse_message(self, file_name):
		if 'new_command' in file_name and 'pkl' in file_name:
			time.sleep(0.01)
			try:
				event = pickle.load(open(file_name, 'rb'))
			except EOFError:
				time.sleep(0.01)
				event = pickle.load(open(file_name, 'rb'))
			self._process_message(event['channel'], event['text'])
			os.remove(file_name)
```

**`Communicator/SlackInterface/run_slack_stream_template.py`**

Replace the last line with:

```python
	app.run(host = '127.0.0.1', port = {@PORT}, debug = False, use_reloader = False)
```

### 3. Slack and ngrok

1. Create a Slack app (https://api.slack.com/apps). Under **OAuth & Permissions**, add the bot token scopes `chat:write`, `files:write`, `channels:history` and `channels:read`, install the app and copy the **Bot User OAuth Token**.
2. Create a channel, invite the app and put the channel ID and the bot user ID into `Utilities/defaults.py`.
3. Start ngrok:
   ```bash
   ngrok http 8080
   ```
4. Under **Event Subscriptions**, enable events, set the request URL to `https://<your-ngrok-subdomain>.ngrok-free.app/slack` and subscribe to the bot event `message.channels` (the URL is verified while ChemOS is running).
5. In the terminal that starts ChemOS:
   ```bash
   export SLACK_TOKEN=xoxb-...        # Windows: set SLACK_TOKEN=xoxb-...
   ```

### 4. Run the ChemOS-integrated validation

Terminal 1:

```bash
ngrok http 8080
```

Terminal 2 (environment activated):

```bash
cd ChemOS
export PYTHONPATH=/path/to/this/repository      # Windows: set PYTHONPATH=C:\path\to\this\repository
python chemOS.py
```

Terminal 3:

```bash
python auto_processing.py --seed 44567943
```

Then post `run experiment` in the Slack channel right away (the controller waits at most `--wait-timeout` seconds, default 300). Use a new ChemOS process for every run. The result is written to `outputs/chemos_validation/`:

```bash
cd DRL_scheduler_for_SDL
python testing/summarize_results.py chemos ../outputs/chemos_validation
```

Seeds of the five validation trials: 44567943, 71935985, 54786065, 816265, 45321838.

## Closed-loop case studies

### Olympus

```bash
git clone https://github.com/aspuru-guzik-group/olympus.git
```

Edit the Olympus sources:

1. `src/olympus/__init__.py`: move `from .campaigns import Campaign, Observations, ParameterSpace` above `from .analyzer import Analyzer`.
2. `src/olympus/campaigns/campaign.py`: move the module-level imports `from olympus.datasets import Dataset` and `from olympus.emulators.emulator import Emulator` to the beginning of `Campaign.set_emulator_specs`:
   ```python
           from olympus.datasets import Dataset
           from olympus.emulators.emulator import Emulator
   ```

Then:

```bash
export OLYMPUS_SRC=/path/to/olympus/src     # Windows: set OLYMPUS_SRC=C:\path\to\olympus\src
```

### Benzyl alcohol oxidation

Start ngrok and a new ChemOS process (steps 3-4 above), then

```bash
python chemos_integration/alkox_auto_processing.py --base-seed 3723620
```

and post `run experiment` in the Slack channel. `--time-scale` (default 0.01) sets the real seconds per virtual second. Results are written to `outputs/case_study_alkox/`:

```bash
cd DRL_scheduler_for_SDL
python testing/summarize_results.py alkox ../outputs/case_study_alkox
cd ..
python chemos_integration/plot_alkox_case_study.py --results-dir outputs/case_study_alkox
```

### Suzuki-Miyaura cross-coupling

Start ngrok and a new ChemOS process, then

```bash
python chemos_integration/suzuki_auto_processing.py
```

and post `run experiment` in the Slack channel. Results are written to `outputs/case_study_suzuki/`:

```bash
cd DRL_scheduler_for_SDL
python testing/summarize_results.py suzuki ../outputs/case_study_suzuki
cd ..
python chemos_integration/plot_suzuki_case_study.py --results-dir outputs/case_study_suzuki
```

## License

MIT License (see `LICENSE`). ChemOS and Olympus are distributed under their own licenses.
