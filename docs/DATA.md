# Data notes

The path figure below shows robot demonstrations used for training, not learned-policy rollouts. Green is the start, orange the finish, red the target centre. Grey shapes are obstacles.

![Demonstration paths](../figures/demonstration-paths.png)

There are four original phone files and thirteen completed Panda replays. These counts have different meanings. The library combines measured paths, simulation completion and older simulation sources. Nineteen contiguous accepted ranges contain 3,990 labels. Unsafe ranges are excluded; future chunks never cross the gaps.

The source manifest records each origin, accepted action ranges and hashes. Original source files are preserved locally. The repository provides representative videos; full original phone footage is not needed to view results.

Printed markers: [table/hand/plate/reset/stove](real-video-markers-A4.pdf), [hand and obstacle page](obstacle-and-passage-markers-A4.pdf). Print at actual size. Measure the black square, excluding the white border.
