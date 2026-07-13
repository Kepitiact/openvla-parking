"""reasoning_data_gen — Step 0 of the reasoning-VLA pipeline.

Generates natural-language reasoning traces + counterfactual pairs that sit
*before* the trajectory in the assistant turn, grounded only in what the
perceiver can see (GT placeholder now, trained-UniAD later). Fact extraction is
pure rules; the teacher LLM only verbalizes a pre-computed fact record.

Public surface (kept small and stable so later steps drop in):
  schema         — special tokens, DECISION_SET, EntityRef, FactRecord, render/format helpers
  scene_record   — SceneRecord (perceivable scene), Frame bundle, from_gt / from_uniad(stub)
  fact_extractor — deterministic rules -> FactRecord
  verbalizer     — MockVerbalizer (default) / QwenVerbalizer (HAL, behind a flag)
  reconcile      — perception-vs-GT status
  counterfactual — stop-injection pairs
  validators     — entity/action/causal/schema gates (reused at eval time, Step 8)
  run_generate   — CLI: infos pkl + source + teacher + version -> versioned traces + manifest
"""
