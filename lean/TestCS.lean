import Mathlib.Analysis.InnerProductSpace.PiL2

example (v w : Fin 32 → ℝ) : ‖⟪v, w⟫‖ ≤ ‖v‖ * ‖w‖ := norm_inner_le_norm v w
