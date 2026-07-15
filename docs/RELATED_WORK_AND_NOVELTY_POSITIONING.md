# Related Work and Novelty Positioning

## 1. Scope

Q-RECON's primary modern-model contribution is not another optimization heuristic
claiming lower reconstruction error. It is an exact information-theoretic boundary
for a declared class of multivariate forecasting models and private recovery targets.

This document separates that contribution from existing gradient-inversion attacks,
time-series attacks, Transformer leakage studies, and general equivariant-network
literature.

## 2. Gradient inversion attacks

### Deep Leakage from Gradients

Zhu, Liu, and Han, *Deep Leakage from Gradients* (arXiv:1906.08935), established
optimization-based reconstruction from shared gradients on vision and language tasks.
It demonstrates that gradients can contain enough information for accurate recovery;
it does not characterize exact observation fibres created by channel symmetries.

### Inverting Gradients

Geiping et al., *Inverting Gradients — How easy is it to break privacy in federated
learning?* (arXiv:2003.14053), introduced a strong cosine-based objective and proved
analytic recovery of inputs to fully connected layers under suitable conditions.
Q-RECON retains this family as a matched classical baseline. Its channel-permutation
result concerns cases where the entire gradient tuple is identical for multiple
labeled private objects, so no improved optimizer can resolve the missing order.

### Batch and adaptive attacks

GradInversion (Yin et al., arXiv:2104.07586) demonstrates high-fidelity recovery of
larger image batches. Learning-based adaptive inversion (Wu et al.,
arXiv:2210.10880) and Gradient Inversion Transcript (Chen and Liu,
arXiv:2505.20026) use auxiliary learning or generative priors to strengthen attacks.
These methods can change the prior and optimization success, but they cannot exceed a
Bayes ceiling once the side-information-conditioned observation channel is fixed.
Q-RECON therefore reports exact ordered recovery separately from orbit recovery and
numerical gradient consistency.

## 3. Transformer and time-series leakage

### Partial Transformer gradients

Li, Xu, and Dras, *Seeing the Forest through the Trees: Data Leakage from Partial
Transformer Gradients* (arXiv:2406.00999), shows empirically that a single Transformer
layer or a small linear component can leak language-model training data. This is
strong evidence that partial visibility does not generally imply privacy. Q-RECON's
result is complementary: for anonymous-channel equivariant forecasters, even the full
gradient cannot identify semantic channel order, and deterministic partial visibility
or randomized postprocessing cannot split that exact fibre.

### TS-Inverse

Meijer et al., *TS-Inverse: A Gradient Inversion Attack Tailored for Federated Time
Series Forecasting Models* (arXiv:2503.20952), studies time-series forecasting across
multiple models and datasets and combines a learned quantile initializer with trend,
periodicity, and resolution-aware losses. Q-RECON implements provenance-tracked
TS-Inverse-style objectives and a disjoint learned-quantile ablation as classical
baselines. The current GIFT-Eval PatchTST results demonstrate nontrivial but imperfect
reconstruction; they are not presented as outperforming TS-Inverse.

### Spatiotemporal attacks

Zheng et al., *Extracting Spatiotemporal Data from Gradients with Large Language
Models* (arXiv:2410.16121), develops priors and defenses for location-like
spatiotemporal data. It reinforces the importance of domain priors and side
information. Q-RECON's theorem explicitly parameterizes public channel metadata and
recovery equivalence because those priors can shrink or eliminate a permutation
orbit.

## 4. Forecasting architectures

PatchTST, *A Time Series is Worth 64 Words* (arXiv:2211.14730), uses temporal patches
and channel independence. iTransformer, *Inverted Transformers Are Effective for
Time Series Forecasting* (arXiv:2310.06625), embeds complete variable histories as
tokens and attends across variables.

Q-RECON does not claim to invent either architecture. It identifies a privacy-relevant
consequence of a specific architectural regime:

- shared parameters across anonymous channels;
- no channel-indexed embedding or head;
- no learned per-channel affine RevIN parameter;
- channel-symmetric loss.

Under these conditions the model is channel-permutation equivariant for every
parameter value, which induces an exact full-gradient orbit.

## 5. Equivariant-network literature

General equivariant-network work studies expressivity, geometry, sample efficiency,
and symmetry-preserving design. Examples include Kohn, Sattelberger, and Shahverdi,
*Geometry of Linear Neural Networks: Equivariance and Invariance under Permutation
Groups* (arXiv:2309.13736). Privacy-oriented equivariant designs also exist, such as
Zhang et al., *Rotation-Equivariant Neural Networks for Privacy Protection*
(arXiv:2006.13016), but they protect intermediate representations through a designed
random phase mechanism rather than deriving a gradient-observation fibre for standard
forecasting architectures.

The novelty claimed by Q-RECON is therefore not the algebraic fact that equivariant
functions respect group actions. It is the complete privacy consequence chain for
training-data reconstruction:

1. simultaneous input/target channel permutations preserve the loss as an identity
   in the model parameters;
2. the complete parameter-gradient tuple is identical;
3. duplicate private signatures determine the exact orbit size;
4. a declared public-label partition determines the residual subgroup and Bayes
   ceiling;
5. the same ceiling applies to adaptive classical and coherent quantum access;
6. deterministic optimizer transcripts and final model deltas remain identical;
7. deterministic postprocessing and data-independent randomized release channels
   cannot split the fibre;
8. the theorem is validated on immutable ETTm1, ETTm2, and ETTh1 windows for
   iTransformer and shared-head PatchTST, with channel-specific controls.

## 6. Claim comparison matrix

| Work family | Main object | Typical result | What Q-RECON adds |
|---|---|---|---|
| DLG / Inverting Gradients / GradInversion | vision or language gradients | constructive reconstruction attack | exact non-identifiability when multiple private objects induce the same full gradient |
| partial-Transformer leakage | selected language-model parameter gradients | empirical leakage from small modules | full-gradient equality and closure under partial visibility for a specific equivariant forecasting class |
| TS-Inverse | time-series observations and targets | stronger domain-tailored reconstruction objective | modern PatchTST baseline plus exact side-information-conditioned recovery ceilings |
| learned/generative inversion | auxiliary attack priors | improved empirical reconstruction | prior changes are separated from information absent from the observation channel |
| equivariant-network theory | group-respecting functions | architectural/geometry properties | gradient privacy theorem, orbit/Bayes accounting, optimizer/release closure, coherent-query consequence |

## 7. Novelty boundary

The following statements are intentionally excluded:

- all Transformers are permutation-private;
- semantic channel labels are always hidden;
- the unordered orbit representative is unrecoverable;
- clipping, quantization, or Gaussian noise is universally protective;
- Q-RECON obtains an end-to-end quantum speedup;
- Q-RECON currently outperforms TS-Inverse on reconstruction quality.

The focused defensible claim is:

> In anonymous-channel, permutation-equivariant forecasting Transformers trained with
> a channel-symmetric loss, exact labeled training data are identifiable only up to a
> side-information-conditioned channel orbit. This is an information-theoretic limit
> shared by classical and coherent quantum reconstruction, and it survives the
> declared optimizer and release transformations.

## 8. Pre-submission literature gate

Before paper submission, every related-work statement must be checked again against
new papers published after the artifact release date. The bibliography must cite the
archival venue version when one exists, and the paper must include a direct theorem
comparison rather than relying only on prose novelty claims.
