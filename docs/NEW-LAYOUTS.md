# Layout variations

Six fixed layout changes, plus one additional stove/goal shift. Each completed pair uses identical initial observations and geometry. These are local diagnostic checks, not a full benchmark.

| Layout | Original baseline | Post-trained | Commands per plan |
|---|---|---|---|
| Left Start | Fail ([video](../media/test-0-baseline.mp4)) | Fail ([video](../media/test-0-fine.mp4)) | 5 |
| Forward Right Start | Fail ([video](../media/layout-1-baseline.mp4)) | Fail ([video](../media/layout-1-fine.mp4)) | 10 |
| Shifted Block | Fail ([video](../media/test-2-baseline.mp4)) | Fail ([video](../media/test-2-fine.mp4)) | 5 |
| Round Obstacle | Fail ([video](../media/layout-3-baseline.mp4)) | Fail ([video](../media/layout-3-fine.mp4)) | 10 |
| Shifted Passage | Fail ([video](../media/test-4-baseline.mp4)) | Pass ([video](../media/test-4-fine.mp4)) | 5 |
| Diagonal Passage | Fail ([video](../media/layout-5-baseline.mp4)) | Pass ([video](../media/layout-5-fine.mp4)) | 10 |
| Stove moved right, 8 cm | Not evaluated | Fail ([video](../media/stove-shift-final.mp4)) | 10 |

The five-command rows use four proposals and take the first in the control. The ten-command rows use the stock single-proposal queue. These are separate matched protocols; their scores should not be pooled. Not completed means no terminal result before the deadline.

## Optional predictor

Matched five-command predictor trials: Left Start: fail; Shifted Block: fail. The shifted-passage predictor run was interrupted by the fixed time limit and is excluded from completion counts. There is no demonstrated predictor success gain in these final-checkpoint trials.

[All completed metrics](../results/seven-layouts.csv) · [Pilot traces](../results/new-layouts.json)

The stove and its target region moved together by 8 cm at reset. The VLA still received only images, robot state and language. The original and changed target regions have separate centres; this was not an invisible scoring-only target change.

The round-obstacle run ends 41.8 mm from the centre but is not collision-free. It is useful partial progress, not a clean success. Centre error alone did not cause this failure.
