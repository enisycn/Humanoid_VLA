# Learning a plate push from video

I chose a simple LIBERO task: **push the plate to the front of the stove**. It lets me inspect the approach, contact and final position within a short project. I added a blocking obstacle and passages to test the same skill under layout changes.

The main comparison is **the original public SmolVLA checkpoint versus my final fine-tuned policy**. A small visual forward predictor is an optional extra. Intermediate checkpoints are kept outside the main comparison.

## From recording to robot data

I recorded four phone videos: normal pushes, losing contact and trying again, a single obstacle, and passages. ArUco markers identify the hand, plate, table, stove and reset periods. I used a checkerboard for camera calibration and measured the marker sizes and plate diameter.

I removed resets and the return-to-start motion, then mapped measured paths into Panda actions. Some direct transfers failed. The final library therefore also includes simulation examples and completed trajectories, with each source labelled separately. It contains **13 robot demonstrations, 19 continuous segments and 3,990 accepted action labels**. Older records provide replay during training.

[Recording and demonstration examples](docs/DATA.md) · [Printable markers](docs/real-video-markers-A4.pdf)

## Model and training

| Component | What I trained |
|---|---|
| SmolVLA vision-language backbone | Frozen: 489 tensors |
| Action expert and projections | Updated: 299 tensors |
| Teacher | Frozen; used only during training |
| Optional forward predictor | Frozen DINOv2 features, fitted PCA and ridge head |

The final stage uses **1,000 updates**, batch size 24 and learning rate 0.000001. Each batch has six new-data and eighteen replay contexts. I combine the official flow-matching loss with teacher velocity matching on replay. The [model history and data sources](docs/MODEL.md) explain initialization and earlier training.

![Training losses](figures/training-loss.png)

These are training losses on changing batches, not validation scores. Lower loss alone does not show better control.

## Evaluation

The policy sees two RGB views, the robot's eight state values and the same plate instruction in every layout. It receives no simulator object positions. I measure target-region arrival, actual obstacle contact, simulated time, total travelled path and contact breaks. Centre error and stopped drift are extra diagnostics.

The fixed final policy completes the familiar simple push, blocking obstacle and passage controls. Original-baseline comparisons use the same initial observations and geometry. New layout checks are reported separately, including failures.

| Task | Original baseline | Post-trained |
|---|---:|---:|
| Simple push | Fail | Pass |
| Passage | Fail | Pass |
| Blocking obstacle | Fail | Pass |

**Three matched development controls: 0/3 → 3/3.** One run per case; same initial observations and physics.

### Simple push

[![Baseline and post-trained policy](media/compare-simple.gif)](media/compare-simple.mp4)

### Passage

[![Baseline and post-trained policy](media/compare-passage.gif)](media/compare-passage.mp4)

### Blocking obstacle

[![Baseline and post-trained policy](media/compare-obstacle.gif)](media/compare-obstacle.mp4)

The GIFs play here; click for full MP4. Each side shows its own simulated time. After a recording ends, its last frame is held for display.

Two additional layouts also succeed while their matched original baselines fail: a shifted passage (five-command pilot) and a diagonal passage (ten-command queue). [All layout results](docs/NEW-LAYOUTS.md), including failures, are kept separately.

![Measured plate paths](figures/primary-paths.png)

[Baseline versus final results](docs/RESULTS.md) · [New layouts and predictor tests](docs/NEW-LAYOUTS.md) · [All control metrics](results/metrics.csv)

The recorded horizontal passage still fails: less obstacle contact does not mean reaching the goal. Larger starting changes also expose missed approaches. The predictor ranks four learned action proposals; it is not a collision guarantee. Better approach and contact-recovery data are future work. A [small native-task check](docs/RETENTION.md) retains the stove task and succeeds at opening a drawer; it does not establish retention of every pretrained skill.

## Run

[Setup, model download, training and evaluation commands](docs/RUNNING.md). Weights and training records use Git LFS. [Research notes](docs/RESEARCH.md) describe what I used from SmolVLA, Learning without Forgetting, DINOv2 and DINO-WM.
