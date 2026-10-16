# Research choices

| Source | What I used |
|---|---|
| [SmolVLA](https://huggingface.co/docs/lerobot/main/smolvla) | Freeze the VLM and train the action expert with the official masked flow loss. |
| [Learning without Forgetting](https://arxiv.org/abs/1606.09282) | Keep a frozen teacher to discourage changing old outputs; adapted here to continuous velocity fields, with replay. |
| [DINOv2](https://arxiv.org/abs/2304.07193) and [DINO-WM](https://arxiv.org/abs/2411.04983) | Frozen visual features and short action-conditioned prediction. My fitted ridge head is much smaller than DINO-WM; it is not a reproduction of that full method. |
| [DAgger](https://proceedings.mlr.press/v15/ross11a.html) | Motivation to inspect states reached after errors and add useful correction examples. I did not implement the full DAgger algorithm. |

A lower matching loss is not a guarantee of no forgetting. A low supervised loss is not a guarantee of successful closed-loop control. The report therefore keeps real rollout tests separate from training loss.
