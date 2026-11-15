# Scheme-C 14-class view: ws split by target size

Source rows: `schemeC_pose23_full_v5` (white / 5 / unknown labels already dropped).

Every non-ws row (pose train+val, source_json) goes to train.
A ws capture session goes to train when the median of its per-image largest
object (long side on 480x640) is >= 24 px, else to val.
ws objects with a long side < 12 px are dropped and ws images left
without objects are removed. Non-ws rows keep every object.
