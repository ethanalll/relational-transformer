.DEFAULT_GOAL := help
DATA ?= $(HOME)/scratch/relrel-rt
VENV ?= $(CURDIR)/.venv
PYTHON ?= "$(VENV)/bin/python"
CARGO ?= cargo
MATURIN ?= "$(VENV)/bin/maturin"
SUITE ?= factorial
ARM ?= $(if $(filter sampler,$(SUITE)),family,global-quantile)
ARMS = $(if $(filter sampler,$(SUITE)),family direct,local-original local-quantile global-original global-quantile)
RUN ?= pilot
SEED ?= 0
DATA_SEED ?= 0
DEVICE ?= cuda:0
GPUS ?= 1
GPU_IDS ?=
EMBED_DEVICE ?= $(DEVICE)
EVAL ?= amazon
STEPS ?= 7629
BATCH ?= 128
CONTEXT ?= 1024
WIDTH ?= 128
CHECKPOINT_EVERY ?= 250
NATIVE_BIN ?= $(CURDIR)/rustler/target/release/rustler

experiment = $(PYTHON) -m rt.experiment
gpu_env = $(if $(strip $(GPU_IDS)),CUDA_VISIBLE_DEVICES="$(GPU_IDS)")
train_command = $(if $(filter 1,$(GPUS)),$(experiment),$(PYTHON) -m torch.distributed.run --standalone --local-addr 127.0.0.1 --nproc-per-node "$(GPUS)" --module rt.experiment)
data_command = $(PYTHON) -m rt.experiment_data
run_args = --data-root "$(DATA)" --suite "$(SUITE)" --arm "$(ARM)" --run "$(RUN)" --seed "$(SEED)" --data-seed "$(DATA_SEED)"
batch_args = --device "$(DEVICE)" --batch "$(BATCH)" --context "$(CONTEXT)" --width "$(WIDTH)"
prepare_args = --data-root "$(DATA)" --device "$(EMBED_DEVICE)" --native-bin "$(NATIVE_BIN)" $(if $(MODEL_PATH),--model-path "$(MODEL_PATH)")
eval_args = --eval "$(EVAL)" $(if $(EVAL_BATCHES),--eval-batches "$(EVAL_BATCHES)")

.PHONY: help setup native download-relbench preprocess-relbench preprocess-synthetic preprocess-arm preprocess-holdout train evaluate summarize check-model
help:
	@echo 'See EXPERIMENTS.md for setup, storage layout and example Slurm jobs.'
	@echo 'make setup native'
	@echo 'make download-relbench EVAL=amazon'
	@echo 'make preprocess-relbench EVAL=amazon'
	@echo 'make preprocess-synthetic SUITE=factorial'
	@echo 'make preprocess-holdout'
	@echo 'make train SUITE=factorial ARM=global-quantile SEED=0 GPUS=4 BATCH=128'
	@echo 'Set GPU_IDS=0,2 to select visible CUDA devices outside Slurm.'
	@echo 'make evaluate SUITE=factorial ARM=global-quantile SEED=0 EVAL=amazon'
	@echo 'make summarize SUITE=factorial EVAL=amazon'

setup:
	@test -x "$(VENV)/bin/python" || uv venv --python 3.12 "$(VENV)"
	uv pip install --python "$(VENV)/bin/python" -r requirements-experiments.txt

native:
	cd rustler && VIRTUAL_ENV="$(VENV)" $(MATURIN) develop --uv --release --locked
	cd rustler && PYO3_PYTHON="$(VENV)/bin/python" $(CARGO) build --release --locked --bin rustler

download-relbench:
	$(data_command) download --data-root "$(DATA)" --relbench "$(EVAL)"

preprocess-relbench:
	$(gpu_env) $(data_command) preprocess $(prepare_args) --suite relbench --relbench "$(EVAL)"

preprocess-synthetic:
	@set -e; for arm in $(ARMS); do $(MAKE) preprocess-arm SUITE="$(SUITE)" ARM="$$arm"; done

preprocess-arm:
	$(gpu_env) $(data_command) preprocess $(prepare_args) --suite "$(SUITE)" --split train --arm "$(ARM)"

preprocess-holdout:
	$(gpu_env) $(data_command) preprocess $(prepare_args) --suite holdout --split val --arm reference

train:
	$(gpu_env) $(train_command) train $(run_args) $(batch_args) --steps "$(STEPS)" --checkpoint-every "$(CHECKPOINT_EVERY)" $(if $(filter 1,$(RESUME)),--resume) $(if $(filter 1,$(COMPILE)),--compile)

evaluate:
	$(gpu_env) $(experiment) evaluate $(run_args) $(batch_args) $(eval_args) $(if $(CHECKPOINT),--checkpoint "$(CHECKPOINT)")

summarize:
	$(experiment) summarize $(run_args) $(eval_args)

check-model:
	$(gpu_env) $(experiment) check-model --device "$(DEVICE)" $(if $(CHECKPOINT),--checkpoint "$(CHECKPOINT)")
