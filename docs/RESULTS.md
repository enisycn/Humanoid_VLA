# Results

These initial four pairs compare the previous trained policy with the final update. They do not compare against the original public model and are exposed development controls.

| Task | Previous video | Final video | Previous/final arrival |
|---|---|---|---|
| Simple push | [MP4](../media/previous-simple.mp4) | [MP4](../media/final-simple.mp4) | pass / pass |
| Passage | [MP4](../media/previous-passage.mp4) | [MP4](../media/final-passage.mp4) | pass / pass |
| Blocking obstacle | [MP4](../media/previous-obstacle.mp4) | [MP4](../media/final-obstacle.mp4) | pass / pass |
| Recorded horizontal passage | [MP4](../media/previous-horizontal-passage.mp4) | [MP4](../media/final-horizontal-passage.mp4) | fail / fail |

[Complete measurements](../results/metrics.csv). In these files B is previous and J is final. These identifiers are archival labels only.

The final policy preserves all three prior pushing successes in this small check. In the failed horizontal passage, robot-obstacle contact samples decrease from 74 to 0, but the goal-area gap remains about 188 mm. This is a limited contact improvement, not successful completion.

## Shared metrics

- Functional success: native target or fixed 50 mm near-arrival rule, correct route, no added-obstacle contact. The result must remain during ten external hold commands.
- Strict native goal is also reported separately. The 50 mm rule is a project criterion, not an official LIBERO score.
- Time: actual policy commands times 0.05 seconds. Failure time is the observation horizon, not a completion time.
- Gripper path: sum of every consecutive measured 3D displacement. Plate path uses consecutive XY positions. Neither is straight-line displacement.
- Contact breaks: loss after a recorded gripper-plate contact, before the external holds.
- Contacts: actual physics-substep pairs, not a count of separate impacts.
- Centre error, stopped drift, peak lift and inference latency are diagnostic measures.

Efficiency comparisons use pairs where both methods succeed. All complete results remain available, including failures. One rollout per case is not a statistical success-rate estimate.
