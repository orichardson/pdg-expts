module PDGs where

data ℕ : Set where
  zero : ℕ
  succ : ℕ → ℕ

data Bool : Set where
  true : Bool
  false : Bool


not : Bool → Bool
not true = false
not false = true


_+_ : ℕ → ℕ → ℕ
zero + m = m
succ n + m = succ(n + m)

_*_ : ℕ → ℕ → ℕ
zero * m = zero
succ n * m = (m * n) + m
 -- just switches args. Cannot prove.
 -- succ n * m = m * (succ n)



 -- if_then_else  : {A : Set} → Bool → A → A → A
 -- if true then x else y = x

-- Oli attempts dependent function composition on his own
_∘_ : {A : Set}{B : A → Set}{C : (x : A) →  B x → Set}
  (f : (a : A) → B a) → (g : {x : A} (b : B x)) →  (x : A) → C B x
-- (f ∘ g) a = _
