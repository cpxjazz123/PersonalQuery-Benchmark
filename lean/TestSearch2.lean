import Mathlib.Data.Finset.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

namespace Foo

example (s : Finset ℕ) (f g : ℕ → ℝ) : ∑ x ∈ s, (f x - g x) = ∑ x ∈ s, f x - ∑ x ∈ s, g x := by
  -- Try using the lemma
  exact Finset.sum_sub_distrib

end Foo
