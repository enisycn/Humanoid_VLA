# Small old-task check

Two native LIBERO task families, one initial state each. Original and final policies receive matched initial RGB, robot state and native physics. No objects are removed. Each completed success also holds for ten external commands.

| Native task | Original baseline | Final policy |
|---|---|---|
| Open middle drawer | Fail | Pass |
| Turn on stove | Pass | Pass |

The stove result is one directly measured retained native capability. The original drawer run fails, so the final drawer success is an additional successful check, not a retained baseline success. Two single trials cannot establish retention of the whole pretraining skill set. Public pretraining coverage is not inferred from task names.

[Raw results](../results/old-task-results.json)
