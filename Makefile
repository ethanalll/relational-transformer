.DEFAULT_GOAL := help
DATA ?= $(HOME)/scratch/relrel-rt
VENV ?= $(CURDIR)/.venv
PYTHON ?= "$(VENV)/bin/python"
CARGO ?= cargo
MATURIN ?= "$(VENV)/bin/maturin"
CORPUS ?= global-quantile
HOLDOUT ?= holdout-val
CORPORA ?=
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
run_args = --data-root "$(DATA)" --corpus "$(CORPUS)" --run "$(RUN)" --seed "$(SEED)" --data-seed "$(DATA_SEED)"
batch_args = --device "$(DEVICE)" --batch "$(BATCH)" --context "$(CONTEXT)" --width "$(WIDTH)"
prepare_args = --data-root "$(DATA)" --device "$(EMBED_DEVICE)" --native-bin "$(NATIVE_BIN)" $(if $(MODEL_PATH),--model-path "$(MODEL_PATH)")
eval_args = --eval "$(EVAL)" --holdout "$(HOLDOUT)" $(if $(EVAL_BATCHES),--eval-batches "$(EVAL_BATCHES)")

.PHONY: help setup native download-relbench preprocess-relbench preprocess-synthetic train evaluate summarize check-model
help:
	@echo 'See EXPERIMENTS.md for setup, storage layout and example Slurm jobs.'
	@echo '  make setup native'
	@echo '  make download-relbench EVAL=amazon'
	@echo '  make preprocess-relbench EVAL=amazon'
	@echo '  make preprocess-synthetic CORPUS=global-quantile'
	@echo '  make train CORPUS=global-quantile SEED=0 RUN=pilot GPUS=4 BATCH=128'
	@echo '  make evaluate CORPUS=global-quantile SEED=0 RUN=pilot EVAL=amazon'
	@echo '  make evaluate CORPUS=global-quantile SEED=0 RUN=pilot EVAL=synthetic HOLDOUT=holdout-val'
	@echo '  make summarize RUN=pilot EVAL=amazon [CORPORA=local-orig,global-quant]'

setup:
	@test -x "$(VENV)/bin/python" || uv venv --python 3.12 "$(VENV)"
	uv pip install --python "$(VENV)/bin/python" -r requirements-experiments.txt

native:
	cd rustler && VIRTUAL_ENV="$(VENV)" $(MATURIN) develop --uv --release --locked
	cd rustler && PYO3_PYTHON="$(VENV)/bin/python" $(CARGO) build --release --locked --bin rustler

download-relbench:
	$(data_command) download --data-root "$(DATA)" --relbench "$(EVAL)"

preprocess-relbench:
	$(gpu_env) $(data_command) preprocess $(prepare_args) --source relbench --relbench "$(EVAL)"

preprocess-synthetic:
	$(gpu_env) $(data_command) preprocess $(prepare_args) --source synthetic --corpus "$(CORPUS)"

train:
	$(gpu_env) $(train_command) train $(run_args) $(batch_args) --steps "$(STEPS)" --checkpoint-every "$(CHECKPOINT_EVERY)" $(if $(filter 1,$(RESUME)),--resume) $(if $(filter 1,$(COMPILE)),--compile)

evaluate:
	$(gpu_env) $(experiment) evaluate $(run_args) $(batch_args) $(eval_args) $(if $(CHECKPOINT),--checkpoint "$(CHECKPOINT)")

summarize:
	$(experiment) summarize $(run_args) $(eval_args) $(if $(CORPORA),--corpora "$(CORPORA)")

check-model:
	$(gpu_env) $(experiment) check-model --device "$(DEVICE)" $(if $(CHECKPOINT),--checkpoint "$(CHECKPOINT)")
