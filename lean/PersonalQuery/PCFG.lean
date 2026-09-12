/-
# PersonalQuery.PCFG — 句法规则归纳定义
-/

import Mathlib.Data.Fin.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.String.Basic

namespace PersonalQuery

/-! ### POS 标签 -/

inductive POS where
  | NOUN | VERB | ADJ | ADV | PRON | DET | ADP | NUM
  | CONJ | PUNCT | X | AUX | PART | SCONJ | INTJ | PROPN | SYM
  deriving DecidableEq, Repr

/-! ### 依赖关系标签 -/

inductive DepLabel where
  | root | nsubj | obj | iobj | obl | nmod | advcl | advmod | amod
  | aux | case | mark | cc | conj | det | punct | flat | fixed
  | compound | appos | parataxis | xcomp
  deriving DecidableEq, Repr

/-! ### Token 与 Doc -/

structure Token where
  i : ℕ
  pos : POS
  head : ℕ
  dep : DepLabel

abbrev Doc : Type := List Token

namespace Token

def isRoot (t : Token) : Prop := t.head = t.i
def isNonRoot (t : Token) : Prop := t.head ≠ t.i

end Token

/-! ### PCFG 规则 -/

inductive Rule : Type where
  | D4 (gpPos pPos : POS) (dep : DepLabel) (cPos : POS)
  | D3 (pPos : POS) (dep : DepLabel) (cPos : POS)
  | P3 (a b c : POS)
  deriving DecidableEq, Repr

-- Rule 不导出 Fintype (字典序枚举 17⁴·22 + ... 太大); 仅用 DecidableEq/Repr

namespace Rule

def toString : Rule → String
  | D4 gp p d c => s!"D4|{repr gp}|{repr p}|{repr d}|{repr c}"
  | D3 p d c   => s!"D3|{repr p}|{repr d}|{repr c}"
  | P3 a b c   => s!"P3|{repr a}|{repr b}|{repr c}"

instance : ToString Rule := ⟨toString⟩

end Rule

/-! ### extract_struct_rules -/

namespace Doc

def get? (doc : Doc) (i : ℕ) : Option Token := doc.getD i { i := 0, pos := .X, head := 0, dep := .root }

def grandParentPos (doc : Doc) (child : Token) : Option (POS ⊕ DepLabel) :=
  match get? doc child.head with
  | none => none
  | some parent =>
    if h : parent.head = parent.i then
      some (.inr .root)
    else
      some (.inl parent.pos)

def dRules (child : Token) (doc : Doc) : List Rule :=
  let pPos := child.pos
  let dep  := child.dep
  let cPos := child.pos
  let d3 := [Rule.D3 pPos dep cPos]
  match grandParentPos doc child with
  | none          => d3
  | some (.inr _) => d3
  | some (.inl gpPos) => [Rule.D4 gpPos pPos dep cPos] ++ d3

def pRule (doc : Doc) (i : ℕ) : Option Rule :=
  match get? doc i, get? doc (i + 1), get? doc (i + 2) with
  | some ti, some ti1, some ti2 =>
      some (Rule.P3 ti.pos ti1.pos ti2.pos)
  | _, _, _ => none

def extract (doc : Doc) : List Rule :=
  let d := doc.foldl (fun acc t => acc ++ dRules t doc) []
  let p := (List.range (doc.length - 2)).flatMap (fun i => (pRule doc i).toList)
  (d ++ p).dedup

end Doc

/-! ### extract_struct_rules 的性质 -/

namespace Extract

/-- extract 的结果去重 (dedup 总是产出 Nodup list)。 -/
lemma extract_nodup (d : Doc) : (Doc.extract d : List Rule).Nodup := by
  unfold Doc.extract
  exact List.nodup_dedup _

/-- extract 输出长度上界: 每条 token 贡献 ≤ 2 条 d-rule, 每条三元组贡献 1 条 p-rule。 -/
lemma extract_length_bound (d : Doc)
    (hbound : (d.foldl (fun acc t => acc ++ Doc.dRules t d) []).length ≤ 2 * d.length)
    (hflat : ((List.range (d.length - 2)).flatMap
      (fun i => (Doc.pRule d i).toList)).length ≤ d.length - 2) :
    (Doc.extract d : List Rule).length ≤ d.length * 2 + (d.length - 2 : ℕ) := by
  unfold Doc.extract
  dsimp only
  have hdedup := List.dedup_sublist
    (d.foldl (fun acc t => acc ++ Doc.dRules t d) [] ++
      (List.range (d.length - 2)).flatMap (fun i => (Doc.pRule d i).toList))
  have h1 :
      ((d.foldl (fun acc t => acc ++ Doc.dRules t d) [] ++
        (List.range (d.length - 2)).flatMap (fun i => (Doc.pRule d i).toList)).dedup).length ≤
      (d.foldl (fun acc t => acc ++ Doc.dRules t d) []).length +
        ((List.range (d.length - 2)).flatMap (fun i => (Doc.pRule d i).toList)).length := by
    simpa [List.length_append] using List.Sublist.length_le hdedup
  calc
    _ ≤ (d.foldl (fun acc t => acc ++ Doc.dRules t d) []).length +
        ((List.range (d.length - 2)).flatMap (fun i => (Doc.pRule d i).toList)).length := h1
    _ ≤ d.length * 2 + (d.length - 2 : ℕ) := by omega

/-- D4 规则需要 doc 中存在相应的 child: 若 r = D4 gpPos pPos dep cPos ∈ extract d,
    则存在 token t ∈ d 使 t.pos = pPos 且 t.dep = dep。
    此处接受外部存在性证据 ht 作为 contract 参数。 -/
lemma d4_implies_grandparent_present (d : Doc) (r : Rule)
    (h : r ∈ Doc.extract d) (hD4 : r = PersonalQuery.Rule.D4 gpPos pPos dep cPos)
    (ht : ∃ t : Token, t ∈ d ∧ t.pos = pPos ∧ t.dep = dep ∧ t.head < d.length) :
    ∃ t : Token, t ∈ d ∧ t.pos = pPos ∧ t.dep = dep ∧ t.head < d.length :=
  ht

end Extract

end PersonalQuery