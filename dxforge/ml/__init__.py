"""DxForge ML layer (scikit-learn only, no model downloads).

The ML layer *proposes*; the deterministic engine in ``dxforge.engine``
*disposes*. Modules:

  text       -- text normalisation + realistic noise injection
  dataset    -- synthetic, engine-labelled datasets (intents, outcomes, traces)
  features   -- text vectorisers and KB-derived tabular features
  slots      -- slot filling (DID / RID / sub-function) for the hybrid parser
  intent_model, outcome_model, defect_model -- training + evaluation
  inference  -- load trained artefacts, predict with confidence, hybrid parse
  monitor    -- prediction logging + drift / low-confidence monitoring
"""
SEED = 42
