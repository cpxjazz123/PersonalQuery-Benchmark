import Mathlib

namespace Foo

-- Try various names
#check @Finset.sum_sub_distrib
#check @Finset.sum_neg_distrib
#check @Finset.sum_add_distrib

example (s : Finset ℕ) (f g : ℕ → ℝ) : ∑ x ∈ s, (f x - g x) = ∑ x ∈ s, f x - ∑ x ∈ s, g x := by
  exact Finset.sum_sub_distrib (s := s) (f := f) (g := g)

end Foo
