# Method notes

The final model is a continuation of the previously successful post-trained policy. Personal recordings, simulation completions and earlier simulation sources are mixed; this is not a personal-data-only experiment. The original public model and both trained checkpoints are preserved locally with SHA-256 identities.

The teacher is used only during training. For each sampled batch, the supervised loss covers 24 training examples and the velocity-matching loss covers the 18 replay examples. Student and teacher receive identical flow time and noise. The coefficient is 1. Future 50-action chunks do not cross excluded label intervals.

Four phone recordings were retained without modifying their original bytes. Hand and plate markers have 20 mm black sides, table markers 35 mm. The measured plate diameter is 180 mm. IDs: hand 10, plate 11, reset 12, stove anchor 13, obstacles 14/15. Marker 13 is an anchor; the target is in front of it. Calibration frames are separate from demonstrations. A fixed table mapping and measured scale are used; no individual take is rotated to improve its outcome.

The horizontal passage uses a measured personal-video path and layout. The diagonal and vertical passage examples include disclosed simulation changes. Literal transfers of some obstacle paths failed and were excluded. Older right-obstacle examples have initial penetration; invalid commands and neighbouring training examples were masked. They are not whole-episode physics-clean demonstrations.

The forward predictor was fitted separately on 51 training episodes and four validation episodes, split by episode. It predicts 64 visual deltas, eight robot-state deltas and one goal-distance change five commands ahead. DINOv2 remains frozen. There are no bottle-pushing training episodes. Progress labels use simulator measurements during fitting only. A separate contact-risk attempt did not produce a reliable task improvement and was not adopted.

The loss graph is training loss on changing batches, not a validation curve. Lower loss does not establish better rollout success. See the [main paired results](RESULTS.md) and [layout variations](NEW-LAYOUTS.md) for the robot outcomes.

## Training and model terms

| Term | Meaning here |
|---|---|
| Training example / context | Two current images, robot state and instruction, paired with up to 50 future robot actions. It is one sampled moment, not a whole video. |
| Batch / update | A batch holds 24 examples. One update changes the trainable weights using that batch. The 1,000 updates repeatedly sample the available data. |
| Frozen | Weights are kept unchanged. Only the action-producing part is updated. |
| Tensor | A numerical array of model weights. The 489 frozen and 299 updated tensors are array counts, not layer or individual parameter counts. |
| Flow matching | SmolVLA's action-training method: it learns how to turn a noisy action sequence into a demonstrated action sequence. The left graph measures errors in this training prediction. |
| Teacher matching | On older examples, the trained policy also matches the fixed teacher's flow predictions using the same noise and flow time. This is a training penalty, not a controller added to the robot. |
| MSE | Mean squared error: squared prediction differences averaged over valid entries. Zero means a match; there is no universal threshold that proves skill retention. |
| DINOv2 | A pretrained image encoder that converts pixels to numerical visual features. Its weights stay fixed here. |
| PCA / ridge head | PCA compresses visual features to 64 numbers. A fitted linear ridge predictor uses these features, robot state and proposed actions to predict changes five commands ahead. |
| Action proposal / chunk | A sequence of robot commands proposed by the VLA. The optional predictor compares four proposals and selects one using predicted goal progress. |

The predictor outputs changes in 64 visual features, eight robot-state values and one goal-distance estimate. It does not output a reliable collision yes/no label. Its fitting uses simulator progress labels; during robot inference it receives images, robot state and proposed actions. The final predictor trials did not demonstrate an extra success gain.
