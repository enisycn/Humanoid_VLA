# Method notes

The final model is a continuation of the previously successful post-trained policy. Personal recordings, simulation completions and earlier simulation sources are mixed; this is not a personal-data-only experiment. The original public model and both trained checkpoints are preserved locally with SHA-256 identities.

The teacher is used only during training. For each sampled batch, the supervised loss covers 24 contexts and the velocity-matching loss covers the 18 replay contexts. Student and teacher receive identical flow time and noise. The coefficient is 1. Future 50-action chunks do not cross excluded label intervals.

Four phone recordings were retained without modifying their original bytes. Hand and plate markers have 20 mm black sides, table markers 35 mm. The measured plate diameter is 180 mm. IDs: hand 10, plate 11, reset 12, stove anchor 13, obstacles 14/15. Marker 13 is an anchor; the target is in front of it. Calibration frames are separate from demonstrations. A fixed table mapping and measured scale are used; no individual take is rotated to improve its outcome.

The horizontal passage uses a measured personal-video path and layout. The diagonal and vertical passage examples include disclosed simulation changes. Literal transfers of some obstacle paths failed and were excluded. Older right-obstacle examples have initial penetration; invalid commands and neighbouring contexts were masked. They are not whole-episode physics-clean demonstrations.

The forward predictor was fitted separately on 51 training episodes and four validation episodes, split by episode. It predicts 64 visual deltas, eight robot-state deltas and one goal-distance change five commands ahead. DINOv2 remains frozen. There are no bottle-pushing training episodes. Progress labels use simulator measurements during fitting only. A separate contact-risk attempt did not produce a reliable task improvement and was not adopted.

The loss graph is training loss on changing batches, not a validation curve. Lower loss does not establish better rollout success. See the complete machine-readable results for all four paired control outcomes.
