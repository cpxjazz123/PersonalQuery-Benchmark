import Mathlib

namespace Foo

example (s : Finset ℕ) (f g : ℕ → ℝ) : ∑ x ∈ s, (f x - g x) = ∑ x ∈ s, f x - ∑ x ∈ s, g x := by
  exact Finset.sum_sub_distrib (s := s) (f := f) (g := g)

end Foo
