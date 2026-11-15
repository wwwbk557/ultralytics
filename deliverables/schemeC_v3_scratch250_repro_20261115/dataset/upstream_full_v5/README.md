# Scheme-C 14-class full view from pose23_current

Label-only: mixed images are kept and only white/5/unknown labels are dropped.
Objects with visibility=0 are dropped (no coordinate; local head has no visibility).
Visibility=1 objects are kept. Classes are mapped to local sentry/hero/engineer/3/4/outpost/base.
Keypoint order stays LT,RT,RB,LB; the third value is removed.
Split is unit-level: pose source-train/val dirs, source_json capture groups, workspace
capture sessions. Units seen on both sides of the strict proposal were consolidated.
