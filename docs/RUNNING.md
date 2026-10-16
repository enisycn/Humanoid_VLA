# Run instructions

Linux, an NVIDIA GPU and EGL are used for the recorded experiments. Use Python 3.12 for simulation/policy and Python 3.9 for the optional predictor. Package versions are recorded in `requirements/`. These are the working environment versions, not a verified clean-machine installation.

## 1. Get the files

Install [Git LFS](https://git-lfs.com/) before cloning, or run `git lfs pull` after installation.

```bash
git clone https://github.com/enisycn/Humanoid_VLA.git
cd Humanoid_VLA
git lfs install --local
git lfs pull
```

`weights/final-expert.safetensors` is the fixed final expert. `previous-expert.safetensors` is its teacher/initialization. `forward-model.pt` is the optional predictor. The public frozen backbone is downloaded separately.

## 2. Environments

```bash
python3.12 -m venv .venv-policy
.venv-policy/bin/python -m pip install --upgrade pip
.venv-policy/bin/python -m pip install --no-deps -r requirements/policy.txt
mkdir -p vendor
git clone https://github.com/huggingface/lerobot.git vendor/lerobot
git -C vendor/lerobot checkout 200ee53596d464bd28f6595cdb31be69a2b5e379
.venv-policy/bin/python -m pip install --no-deps -e vendor/lerobot

python3.12 -m venv .venv-native
.venv-native/bin/python -m pip install --upgrade pip
.venv-native/bin/python -m pip install --no-deps -r requirements/native.txt
.venv-native/bin/python -m pip install --no-deps -e vendor/lerobot
.venv-native/bin/python scripts/setup_assets.py
.venv-policy/bin/python scripts/prepare_model.py --out checkpoints/backbone --public-out checkpoints/original
```

`--no-deps` preserves the observed versions. LeRobot's declared Torch limit differs from the tested Torch 2.11 environment, so a normal resolver install can replace the working Torch version. CUDA driver/wheel compatibility still depends on the computer. Install the EGL/OpenGL runtime through the operating system if headless rendering is unavailable.

The policy config references `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`. Download its config/processor files on the first online run; keep the same Hugging Face cache for later offline runs.

```bash
.venv-policy/bin/python scripts/check_install.py --part policy
.venv-native/bin/python scripts/check_install.py --part native
```

## 3. Evaluate the same layouts

The main three-layout comparison uses `configs/familiar-cases.json`. For that table, use `--cases configs/familiar-cases.json --indices 0 1 2`. The separate new-layout/predictor pilot uses `configs/evaluation-cases.json` and a five-command cadence. Use `--execute 10` for the primary table (the default), or `--execute 5` for the four-proposal predictor pilot.

List the fixed scenarios:

```bash
.venv-native/bin/python scripts/evaluate.py --list
```

Final policy:

```bash
MUJOCO_GL=egl .venv-native/bin/python scripts/evaluate.py \
  --indices 0 1 2 3 4 5 \
  --model-dir checkpoints/backbone \
  --expert weights/final-expert.safetensors \
  --policy-python .venv-policy/bin/python \
  --libero-config .cache/libero-config --output outputs/final
```

For the public baseline, use `--model-dir checkpoints/original` and omit `--expert`, preserving its own processors and saved precision. For the previous policy, use `--expert weights/previous-expert.safetensors`. Use a new output directory for each run. Every method uses the same four proposals per query and executes five commands. The control takes proposal zero; the predictor ranks those same proposals.

The output includes the trace, actually issued actions, model identity, proposals, camera frames and all terminal metrics. Failed runs remain in the output.

## 4. Check data and train

```bash
.venv-policy/bin/python scripts/train.py \
  --model-dir checkpoints/backbone --initial-expert weights/previous-expert.safetensors \
  --data training_data --output outputs/data-check --check-only

.venv-policy/bin/python scripts/train.py \
  --model-dir checkpoints/backbone --initial-expert weights/previous-expert.safetensors \
  --data training_data --output outputs/training --updates 1000
```

The data check loads no model and performs no updates. The training script uses the exported 64 records and the fixed batch24 schedule. It freezes the VLM and saves expert checkpoints, losses and optimizer/RNG state. The reported experiment paused after 200 updates and resumed to 1,000; the portable script runs continuously. The objective functions are extracted from the actual training code. Bitwise fresh-run equivalence is not claimed.

Plot the recorded loss:

```bash
.venv-native/bin/python scripts/plot_training.py \
  --csv results/training-loss.csv --out figures/training-loss.png
```

## 5. Optional forward predictor

Create a separate Python 3.9 environment using `requirements/world-model.txt`. Clone DINOv2 at the recorded revision:

```bash
python3.9 -m venv .venv-wm
.venv-wm/bin/python -m pip install --upgrade pip
.venv-wm/bin/python -m pip install --no-deps -r requirements/world-model.txt
git clone https://github.com/facebookresearch/dinov2.git vendor/dinov2
git -C vendor/dinov2 checkout 85a24602099d397264d5b30461ad7f3bfd726ca1
curl -L https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth -o checkpoints/dinov2_vits14_pretrain.pth
```

The worker verifies DINO checkpoint SHA-256 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`.

```bash
DINO_SOURCE="$PWD/vendor/dinov2" \
DINO_CHECKPOINT="$PWD/checkpoints/dinov2_vits14_pretrain.pth" \
MUJOCO_GL=egl .venv-native/bin/python scripts/evaluate.py \
  --indices 0 2 4 --execute 5 --model-dir checkpoints/backbone \
  --expert weights/final-expert.safetensors \
  --policy-python .venv-policy/bin/python --wm-python .venv-wm/bin/python \
  --world-model weights/forward-model.pt \
  --libero-config .cache/libero-config --output outputs/with-predictor
```

## 6. New phone recordings

Use the printable PDFs in `docs/`. Keep the camera fixed after calibration; measure actual reference spacing and marker sizes. Show reset ID12 between takes. Hand ID10 and plate ID11 provide motion; IDs14/15 mark obstacle layouts. ID13 anchors the stove position.

```bash
.venv-native/bin/python scripts/split_video.py --video recording.mov --out outputs/segments
```

Segmentation does not produce robot actions automatically. Calibrated tracking, a measured hand-to-contact offset, Panda execution and physical admission checks are separate steps. The actual calibration/tracking code is under `src/work/phone_video_prep`; the recorded retarget entrypoint and its limitations are documented in `method_source/`. Their archived workspace entrypoints are not standalone portable commands. The exported training records allow the training stage to run without repeating that manual preparation.

Summarize saved rollout metrics with `scripts/summarize.py --input outputs/final --out outputs/final-metrics.csv`. The original phone files are in `recordings/`; Git LFS is required.
