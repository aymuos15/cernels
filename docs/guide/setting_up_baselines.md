# Setting up baselines

When an op has no native-torch equivalent, the eager production op (e.g. `torchvision.ops.nms`) IS the reference — set it as `baseline` and leave `lib` unset.

