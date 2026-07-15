# Full-Batch Linear Training Transcripts Share the Same Fibre

## 1. Statement

For biased linear regression with fixed known targets and mean half-squared loss,
the complete gradient function is determined by

\[
S=X^\top X,\qquad m=X^\top\mathbf1,\qquad R=Y^\top X.
\]

The complete-fibre theorem shows that two input batches expose the same gradient
function exactly when

\[
X'=QX,\qquad Q\in O(B),\qquad Q[\mathbf1,Y]=[\mathbf1,Y].
\]

This document lifts that result from isolated gradient observations to complete
training runs.

## 2. Loss is also a sufficient-statistic function

For parameters \((\Theta,b)\),

\[
\begin{aligned}
2B L(\Theta,b)
={}&\operatorname{tr}(\Theta S\Theta^\top)
+2b^\top\Theta m+B\|b\|_2^2\\
&-2\langle\Theta,R\rangle
-2b^\top Y^\top\mathbf1
+\|Y\|_F^2.
\end{aligned}
\]

Because targets are fixed and known, the same statistics that determine every
gradient query also determine every loss query.

## 3. Deterministic optimizer theorem

### Theorem 1 — transcript invariance

Consider any deterministic full-batch training algorithm whose state update at
step \(t\) is a data-independent function of:

- the current model and optimizer state;
- the current loss and full gradient;
- a public learning-rate or hyperparameter schedule; and
- an external randomness/noise realization fixed independently of the hidden
  batch.

Start the algorithm from the same initial state on two batches \(X,X'\) in the
same target-stabilizer fibre. Then every item in the two training transcripts is
identical:

- losses;
- weight and bias gradients;
- parameters and checkpoints;
- momentum buffers;
- Adam first and second moments;
- stopping decisions and any deterministic post-processing of these values.

### Proof

At the common initial state, the fibre theorem and the loss formula give equal
losses and gradients. Therefore the deterministic update rule produces the same
next optimizer and model state. Induction repeats the argument for every later
step. If an external noise realization is supplied, couple the runs with the
same realization; equality still holds pathwise. \(\square\)

### Corollary 1 — randomized transcript distributions

If optimizer randomness or additive noise is sampled from a law independent of
the hidden batch, the two transcript distributions are identical. A coupling
using the same random draw gives pathwise equality, hence no statistical test can
distinguish the fibre members from the transcript.

## 4. Security consequences

For this declared channel, releasing more full-batch checkpoints does not refine
the fibre. Neither of the following reveals the original ordered batch:

- arbitrarily many full-batch SGD updates;
- full-batch Momentum or Adam states;
- data-independent weight decay;
- deterministic learning-rate schedules;
- additive data-independent gradient noise;
- loss curves and early-stopping times.

The result also applies to any deterministic downstream transcript generated from
the same loss/gradient sequence. It is stronger than a one-step collision: the
entire training process is observation-equivalent.

The theorem does **not** apply unchanged when the transcript contains information
outside the full-batch loss/gradient oracle, including:

- minibatch identities or a batch schedule that separates samples;
- per-sample gradients;
- data-dependent clipping thresholds or randomness;
- nonlinear victims whose gradient function has different sufficient statistics;
- side channels such as activations or sample-dependent timing.

## 5. Executable implementation

`qrecon.theory.linear_training_transcripts` implements:

- exact loss evaluation from sufficient statistics;
- full-batch SGD, Momentum, and Adam simulation;
- data-independent weight decay;
- explicitly coupled additive gradient noise;
- complete parameter, gradient, loss, momentum, and second-moment snapshots;
- a maximum transcript-difference audit.

Tests construct nontrivial target-stabilizer rotations and verify complete
transcript equality for all three optimizers over multiple steps. They also
cross-check direct training against the sufficient-statistic emulator and verify
that changing the oracle statistics changes the transcript.

## 6. Paper implication

A reconstruction paper must not count repeated full-batch checkpoints as
independent evidence when the underlying linear/MSE gradient oracle has already
been learned. Under the theorem assumptions, the entire sequence can be simulated
offline after the optimal packed probe recovery of \((S,m,R)\). The correct
privacy target remains the quotient or a domain-restricted orbit intersection,
not the original ordered batch unless an additional identifiability result is
proved.
