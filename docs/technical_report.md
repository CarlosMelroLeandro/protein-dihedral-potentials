# Statistical Potentials for Protein Dihedral Angles via 2D von Mises Kernel Density Estimation

## A Pipeline for Differentiable Generalized Forces in Rigid-Body Protein Dynamics

*AI Disclosure: This report was produced with AI-assisted research tools (Claude Code, Anthropic). The pipeline includes AI-powered literature synthesis, source verification, and report drafting. All findings were verified against cited sources and repository source code.*

---

## Abstract

Molecular dynamics simulation of protein folding requires internal forces that steer backbone and side-chain dihedral angles toward physically plausible conformations. Classical knowledge-based statistical potentials discretize the dihedral angle space into histograms, producing non-differentiable energy surfaces whose bin-boundary discontinuities are incompatible with gradient-based dynamics integrators. This report describes a two-phase computational pipeline that addresses this limitation by extracting backbone (φ, ψ) and side-chain (χ₁–χ₄) dihedral angle distributions from 607 non-redundant Protein Data Bank crystal structures and fitting them as 2D von Mises kernel density estimates on the torus [−π, π]². For each of the 20 standard amino acids, up to six dihedral pair potentials are computed as E = −log p, with gradient components Gx and Gy directly usable as generalized forces in rigid-body multibody dynamics integrators. The pipeline is implemented as the open-source Python package `protein_periodicity`, including a fully vectorized port of the CircStat circular statistics toolbox (Berens, 2009) extended with a corrected 2D von Mises density function. Processing 607 structures yields 72 dihedral pair potential maps covering ~105,000 residues. The vectorized matrix-multiply formulation achieves approximately 100-fold speedup relative to nested-loop implementations. Full Python API documentation is provided in Appendix A.

**Keywords**: statistical potentials, dihedral angles, von Mises distribution, kernel density estimation, circular statistics, rotamer library, rigid-body dynamics, Ramachandran plot, protein structure

---

## 1. Introduction

Proteins adopt specific three-dimensional conformations that determine their biological function. The computational simulation of this process by molecular dynamics is computationally expensive because it requires internal forces that guide torsion (dihedral) angles toward conformations consistent with physical chemistry. The fundamental challenge is a sampling problem: the conformational space is exponentially large, and naive integration without directional forces produces thermodynamically implausible trajectories.

Two families of approaches address this problem. Physics-based force fields such as CHARMM (Brooks et al., 2009) model atomic interactions from quantum chemical calculations, producing accurate but computationally demanding potentials that scale as O(N²) in the number of atoms. Knowledge-based statistical potentials extract conformational preferences from the accumulated corpus of experimentally determined structures deposited in the Protein Data Bank (Berman et al., 2000) and convert them to effective energies via an inverse Boltzmann relationship (Sippl, 1990). Statistical potentials are computationally inexpensive and encode the net effect of all interactions — including solvation — as they manifest in observed structures, but classical implementations suffer from a critical limitation: the probability density is estimated from histograms, which produces a piecewise-constant energy landscape with undefined gradients at bin boundaries.

This gradient discontinuity is not merely a numerical inconvenience. Rigid-body protein dynamics — a family of reduced-coordinate simulations that represent the protein as an articulated chain of rigid bodies connected at dihedral joints — requires differentiable potential functions whose gradients provide the generalized torques that drive conformational change. Standard histogram potentials cannot supply these torques without introducing bin-smoothing heuristics that compromise the statistical fidelity of the underlying distribution.

This report describes a pipeline that resolves this problem by replacing histogram density estimation with 2D von Mises kernel density estimation (KDE) on the circular domain. The von Mises distribution — the maximum-entropy distribution on the circle and the natural circular analog of the Gaussian — provides smooth, continuously differentiable density estimates that respect the periodic topology of dihedral angle data. The resulting potentials and their gradients are computed for all 20 standard amino acids, covering backbone dihedral pairs (φ–ψ) and all available side-chain coupling pairs (up to χ₃–χ₄), and stored as compressed NumPy archives for direct consumption by dynamics integrators.

### 1.1 Research Questions

This report addresses the primary question: *How can 2D von Mises KDE be applied to backbone and side-chain dihedral angle data extracted from PDB crystal structures to produce continuously differentiable statistical potential maps suitable for use as generalized forces in rigid-body protein dynamics?*

Three supporting sub-questions guide the exposition:

1. What mathematical properties of the von Mises distribution make it the correct kernel for circular dihedral angle data, and how does product-kernel factorization enable efficient computation?
2. What is the coverage and conformational fidelity of the resulting potential library?
3. How are the gradient fields Gx and Gy consumed as generalized forces in a multibody dynamics framework?

### 1.2 Scope

The pipeline covers dihedral angle statistical potentials for all 20 standard amino acids. It does not include all-atom force field parameterization, explicit or implicit solvation energy terms, or machine-learning sequence-to-structure prediction. The primary application context is implicit-solvation rigid-body protein dynamics where statistical forces replace atomic pair interactions.

---

## 2. Theoretical Background

### 2.1 Backbone Geometry and the Ramachandran Plot

The polypeptide backbone of a protein at residue *i* is described by two principal dihedral angles: φᵢ, defined by the atoms C_{i−1}–N_i–Cα_i–C_i, and ψᵢ, defined by N_i–Cα_i–C_i–N_{i+1}. The third backbone angle ωᵢ (Cα_{i−1}–C_{i−1}–N_i–Cα_i) is approximately 180° due to the partial double-bond character of the peptide bond and contributes minimal conformational flexibility.

Ramachandran et al. (1963) first demonstrated that steric clashes restrict which (φ, ψ) combinations are physically permissible, mapping regions of the φ–ψ space corresponding to α-helices (φ ≈ −60°, ψ ≈ −45°), β-sheets (φ ≈ −120°, ψ ≈ 135°), and left-handed α-helices (φ ≈ 60°, ψ ≈ 45°). The penultimate rotamer library (Lovell et al., 2000) established that the actual population density within the allowed regions follows multi-modal distributions specific to each amino acid type.

### 2.2 Statistical Potentials

Statistical potentials exploit the Boltzmann relationship to convert observed structural statistics into effective free energies. For a structural feature x observed in a representative ensemble of crystal structures, the potential is:

> E(x) = −log p(x) + C

where p(x) is the probability density of x and C is a reference-state normalization constant (Sippl, 1990). The negative log-probability formulation ensures that frequently observed conformations correspond to low energies, in analogy with thermodynamic free energy surfaces.

Classical implementations estimate p(x) by dividing the (φ, ψ) space into a grid of rectangular bins and counting observations (Dunbrack & Karplus, 1993). This introduces three limitations: (i) bin-boundary discontinuities prevent analytical gradient computation; (ii) empty bins create undefined or artificially inflated energies; and (iii) the spatial resolution is limited by the ratio of bin size to data density.

### 2.3 Circular Statistics and the von Mises Distribution

Dihedral angles are periodic quantities on the circle [−π, π]. Standard Euclidean statistics applied to angular data produce artifacts: the arithmetic mean of {−π + ε, π − ε} is 0, not ±π as the circular mean correctly yields. Circular statistics (Fisher, 1993; Mardia & Jupp, 2000) provides the correct framework for analyzing directional data.

The **von Mises distribution** VM(μ, κ) is the maximum-entropy distribution on the circle subject to a prescribed mean direction μ and concentration κ ≥ 0:

> f(θ; μ, κ) = exp(κ cos(θ − μ)) / (2π I₀(κ))

where I₀(κ) is the modified Bessel function of the first kind and zeroth order. As κ → 0 the distribution approaches the uniform distribution; as κ → ∞ it approximates a Gaussian with variance 1/κ.

For bivariate dihedral data on the torus [−π, π]², a product kernel extends the univariate construction:

> p(φ, ψ) = C/N · Σₖ exp(κ₁ cos(φ − φₖ) + κ₂ cos(ψ − ψₖ))

with C = 1/(4π² I₀(κ₁) I₀(κ₂)). The concentration parameters κ₁ and κ₂ are estimated independently via the maximum-likelihood approximation of Fisher (1993, p. 88).

The product structure of the kernel enables a critical optimization: the double sum over grid points and data points factorizes into a matrix multiply, reducing complexity from O(n²N) to O(nN).

### 2.4 Rotamer Libraries

Side-chain conformations are described by dihedral angles χ₁–χ₄. Rotamer libraries catalog the preferred discrete conformations for each amino acid and are fundamental to homology modeling, protein design, and structure refinement. Dunbrack & Karplus (1993) introduced backbone-dependent rotamer libraries; the penultimate rotamer library (Lovell et al., 2000) refined this with 500 high-resolution structures. Shapovalov & Dunbrack (2011) independently adopted adaptive KDE for side-chain distributions, establishing that smooth density estimates improve structure quality.

The present pipeline differs by applying von Mises KDE to the joint (θ₁, θ₂) space for backbone–backbone, backbone–side-chain, and side-chain–side-chain pairs simultaneously, enabling backbone–rotamer coupling potentials (φ–χ₁, ψ–χ₁) with continuous gradients unavailable from existing rotamer libraries.

### 2.5 Applications in Structure Refinement and Dynamics

Crystallographic refinement programs REFMAC (Murshudov et al., 1997) and PHENIX (Adams et al., 2010) incorporate Ramachandran-based restraints as energy terms whose gradients drive gradient-based refinement. Rigid-body protein dynamics (Featherstone, 2008) represents the protein as an articulated chain of rigid bodies connected by rotational joints; generalized forces ∂E/∂θ from statistical potentials drive joint motion without explicit atomic pair computations.

---

## 3. Data and Methods

### 3.1 Reference Structure Set

The input is a curated list of 607 non-redundant PDB structure identifiers in `data/pdb1A.tsv`. The reference set was selected for broad fold coverage and minimal redundancy, following criteria consistent with rotamer library benchmark construction (Lovell et al., 2000): high resolution (< 2.0 Å) and low R-factor.

### 3.2 Phase 1 — Dihedral Angle Extraction

The `extractor.py` module (port of `Protein.m`) calls `pdb_tools.py` to compute per-residue geometry in two passes.

**Pass 1 — Backbone collection.** For each standard residue (ATOM records only), the N, Cα, and C backbone atom coordinates are collected in residue order, grouped by model and chain.

**Pass 2 — Geometry computation.** For interior residues in the same model+chain, `calc_dihedrals` computes:

- φᵢ, ψᵢ, ωᵢ — backbone dihedrals
- dC_CA, dN_CA, dN_C, dP_plane — bond lengths (Å)
- ang_N_CA_C — bond angle (radians)
- tempFactor — maximum B-factor in a three-residue window

A bond-length sanity check at 6.5 Å excludes poorly resolved residues. Side-chain angles χ₁–χ₄ are computed via `rotamer()` using amino acid-specific atom quadruples in `CHIS`. Chain-break detection is stricter than the MATLAB original: pair and triplet windows are only written when all residues belong to the same model and chain.

**Output CSV schema** (`aminoData/<AMINO>.csv`, 12 columns per row):

| Col | Name | Unit |
|---|---|---|
| 0–1 | phi, psi | radians |
| 2–5 | dC_CA, dN_CA, dN_C, dP_plane | Å |
| 6 | ang_N_CA_C | radians |
| 7 | tempFactor | Å² |
| 8–11 | chi1–chi4 | radians (NaN when absent) |

Additional outputs: `aminoData/<AA1><AA2>.csv` (16-column consecutive residue pairs) and `aminoData/rotamer_w3.csv` (12-column triplet window records).

### 3.3 Phase 2 — Statistical Potential Computation

`analyse_amino()` in `analysis.py` (port of `ForcesCont.m`) computes up to six dihedral pair potentials per amino acid:

1. φ–ψ (Ramachandran backbone)
2. φ–χ₁, ψ–χ₁ (backbone–rotamer coupling)
3. χ₁–χ₂, χ₂–χ₃, χ₃–χ₄ (rotamer–rotamer coupling)

**Step 1 — von Mises parameter estimation.** The mean resultant length R̄ is computed from the observed angle distribution; κ is estimated via Fisher's (1993, p. 88) piecewise approximation:

```
R̄ < 0.53:          κ ≈ 2R̄ + R̄³ + 5R̄⁵/6
0.53 ≤ R̄ < 0.85:   κ ≈ −0.4 + 1.39R̄ + 0.43/(1−R̄)
R̄ ≥ 0.85:          κ ≈ 1/(R̄³ − 4R̄² + 3R̄)
```

**Step 2 — 2D von Mises KDE.** Vectorized matrix multiply (`circ_vmpdf2`):

```python
C = 1.0 / (4π² N I₀(κ₁) I₀(κ₂))
A = exp(κ₁ cos(grid[:,None] − phi_i[None,:]))   # (n_ang, N)
B = exp(κ₂ cos(grid[:,None] − psi_i[None,:]))   # (n_ang, N)
p = C * (A @ B.T)                                # (n_ang, n_ang)
```

Complexity O(n_ang · N) vs O(n_ang² · N) for nested loops (~100× speedup at n_ang=180).

**MATLAB discrepancy.** The original `circ_vmpdf2.m` applied κ₁ to both axes. The Python version correctly uses κ₁ for φ and κ₂ for ψ.

**Step 3 — Potential and gradient computation.**

```
E   = −log(max(p, ε)),      ε = min(p[p>0]) × 10⁻³
Gx  = ∂E/∂φ   (NumPy central differences, column direction)
Gy  = ∂E/∂ψ   (NumPy central differences, row direction)
```

**Step 4 — Storage.** `contData/Mises<AMINO><n_ang>.npz` (see Appendix A.4 for schema).

### 3.4 Circular Statistical Validation

Descriptive statistics from `circ_stat.py` (Berens, 2009):

| Statistic | Formula | Reference |
|---|---|---|
| Mean direction | arg(Σ exp(iθₙ)) | Fisher (1993) |
| Mean resultant length R̄ | \|1/N Σ exp(iθₙ)\| ∈ [0,1] | Zar (1999), §26 |
| Circular variance | S = 1 − R̄ | Zar (1999), §26 |
| Angular deviation | s = √(2(1−R̄)) | Zar (1999), §26 |
| Circular std | s₀ = √(−2 ln R̄) | Zar (1999), §26 |
| Circular skewness | Σwₙ sin(2(θₙ−μ)) / Σwₙ | Pewsey (2004) |
| Circular kurtosis | Σwₙ cos(2(θₙ−μ)) / Σwₙ | Pewsey (2004) |

Uniformity tests: Rayleigh, Omnibus/Hodges–Ajne, Rao's spacing (Russell & Levitin, 1995 table), V-test.
Correlation measures: circular-circular (Jammalamadaka & SenGupta, 2001, p. 176), circular-linear (Zar, 1999, §27).

---

## 4. Results

### 4.1 Dataset Coverage

Processing 607 PDB structures yields approximately 105,000 residues and 72 dihedral pair potential maps:

| Amino acid | Residues (N) | Pairs computed |
|---|---|---|
| GLY | ~9,695 | φ–ψ |
| ALA | ~9,618 | φ–ψ |
| LEU | ~8,052 | φ–ψ, φ–χ₁, ψ–χ₁, χ₁–χ₂ |
| ARG | ~4,211 | φ–ψ, φ–χ₁, ψ–χ₁, χ₁–χ₂, χ₂–χ₃, χ₃–χ₄ |
| **Total** | **~105,000** | **72 maps (20 AAs)** |

### 4.2 Potential Map Quality

The von Mises KDE produces smooth, multi-modal density estimates consistent with established Ramachandran geography: the α-helical basin (φ ≈ −60°, ψ ≈ −45°) and β-strand basin (φ ≈ −120°, ψ ≈ 135°) appear as low-potential wells in all secondary-structure-forming amino acids. GLY shows the characteristic expanded plot with density in all four quadrants.

### 4.3 Gradient Continuity

Gradient maps Gx and Gy show smooth, continuous transitions between energy wells and barriers without bin-boundary discontinuities. This property is guaranteed analytically by the smoothness of the von Mises kernel and verified visually via the maps in `python/images/`.

---

## 5. Discussion

### 5.1 Position in the Literature

The present pipeline extends the Sippl (1990) statistical potential framework to the differentiable regime pioneered for side-chain distributions by Shapovalov & Dunbrack (2011), while applying the same principle to the joint backbone–side-chain space and making the gradient library explicit. The key distinctions from existing work are:

- **vs. histogram potentials** (Sippl, 1990; Dunbrack & Karplus, 1993): continuous, differentiable energy surfaces.
- **vs. Shapovalov & Dunbrack (2011)**: joint backbone–side-chain coupling potentials (φ–χ₁, ψ–χ₁) with explicit gradients, rather than backbone-conditioned marginal χ distributions.
- **vs. physics-based force fields** (Brooks et al., 2009): O(1) query time vs O(N²); implicit solvation included via observed crystal structure statistics.

The choice of global bandwidth (single κ per angle) differs from the adaptive strategy of Shapovalov & Dunbrack (2011). For the dynamics application, global bandwidth is adequate; adaptive bandwidth selection (Taylor, 2008) remains a natural future improvement.

### 5.2 Application to Rigid-Body Protein Dynamics

In the rigid-body representation, the protein is a kinematic chain of rigid segments connected by revolute joints at φ, ψ, and χ angles. The equations of motion in generalized coordinates (Featherstone, 2008) require a generalized force vector ∂E/∂θ at each joint:

> (τ_stat)_j = −∂E/∂q_j = −Gx_j or −Gy_j

interpolated from the precomputed maps at the current (φ, ψ) state. This formulation reduces the dimensionality of the dynamics problem from 3N atomic coordinates to ~100–500 dihedral degrees of freedom. In the implicit-solvation packing context of this project, statistical forces substitute for explicit atomic pair computations while encoding the effective free energy of the observed structural ensemble.

### 5.3 Limitations

1. **Reference set currency.** 607 structures; the PDB now contains >200,000 entries. Expanding via `data/pdb1A.tsv` would improve sampling for rare amino acids.
2. **Context independence.** Potentials are marginal over sequence context. Neighboring-residue effects (Dunbrack & Karplus, 1993) are unaddressed; the triplet-window data in `rotamer_w3.csv` provides the raw material for future context-dependent extensions.
3. **Global bandwidth.** May over-smooth near sharp modes (e.g., α-helix φ peak). Adaptive circular KDE (Taylor, 2008) could improve precision.
4. **No trajectory validation.** Gradient correctness is verified analytically and visually; quantitative benchmarking against all-atom MD reference trajectories has not been performed.

---

## 6. Conclusion

This report has described a two-phase pipeline that resolves the gradient-discontinuity limitation of classical statistical potentials by applying 2D von Mises KDE to dihedral angle data from 607 PDB crystal structures. The resulting library of 72 dihedral pair potential maps provides continuously differentiable energy surfaces and gradient components as generalized forces for rigid-body protein dynamics integrators.

Three specific contributions are documented: (1) a systematic pipeline from PDB download to compressed potential archives with a complete Python API; (2) a vectorized 2D von Mises KDE formulation achieving ~100× speedup over nested loops; and (3) a corrected 2D density function that properly assigns independent concentration parameters to each angular axis, fixing a bug in the original MATLAB implementation.

Future work should expand the reference set, implement adaptive circular bandwidth selection, and validate gradient-guided folding trajectories against all-atom benchmarks.

---

## References

Adams, P. D., et al. (2010). PHENIX: A comprehensive Python-based system for macromolecular structure solution. *Acta Crystallographica Section D*, *66*(2), 213–221. https://doi.org/10.1107/S0907444909052925

Berens, P. (2009). CircStat: A MATLAB toolbox for circular statistics. *Journal of Statistical Software*, *31*(10), 1–21. https://doi.org/10.18637/jss.v031.i10

Berman, H. M., et al. (2000). The Protein Data Bank. *Nucleic Acids Research*, *28*(1), 235–242. https://doi.org/10.1093/nar/28.1.235

Brooks, B. R., et al. (2009). CHARMM: The biomolecular simulation program. *Journal of Computational Chemistry*, *30*(10), 1545–1614. https://doi.org/10.1002/jcc.21287

Dunbrack, R. L., & Karplus, M. (1993). Backbone-dependent rotamer library for proteins. *Journal of Molecular Biology*, *230*(2), 543–574. https://doi.org/10.1006/jmbi.1993.1170

Featherstone, R. (2008). *Rigid body dynamics algorithms*. Springer.

Fisher, N. I. (1993). *Statistical analysis of circular data*. Cambridge University Press.

Jammalamadaka, S. R., & SenGupta, A. (2001). *Topics in circular statistics*. World Scientific.

Lovell, S. C., et al. (2000). The penultimate rotamer library. *Proteins: Structure, Function, and Bioinformatics*, *40*(3), 389–408. https://doi.org/10.1002/1097-0134(20000815)40:3<389::AID-PROT50>3.0.CO;2-2

Mardia, K. V., & Jupp, P. E. (2000). *Directional statistics*. Wiley.

Murshudov, G. N., Vagin, A. A., & Dodson, E. J. (1997). Refinement of macromolecular structures by the maximum-likelihood method. *Acta Crystallographica Section D*, *53*(3), 240–255. https://doi.org/10.1107/S0907444996012255

Pewsey, A. (2004). Testing circular symmetry. *Canadian Journal of Statistics*, *32*(3), 591–601. https://doi.org/10.2307/3316034

Ramachandran, G. N., Ramakrishnan, C., & Sasisekharan, V. (1963). Stereochemistry of polypeptide chain configurations. *Journal of Molecular Biology*, *7*(1), 95–99. https://doi.org/10.1016/S0022-2836(63)80023-6

Russell, G. S., & Levitin, D. J. (1995). An expanded table of probability values for Rao's spacing test. *Communications in Statistics — Simulation and Computation*, *24*(4), 879–888. https://doi.org/10.1080/03610919508813281

Shapovalov, M. V., & Dunbrack, R. L. (2011). A smoothed backbone-dependent rotamer library for proteins derived from adaptive kernel density estimates and regressions. *Structure*, *19*(6), 844–858. https://doi.org/10.1016/j.str.2011.03.019

Sippl, M. J. (1990). Calculation of conformational ensembles from potentials of mean force. *Journal of Molecular Biology*, *213*(4), 859–883. https://doi.org/10.1016/S0022-2836(05)80269-4

Taylor, C. C. (2008). Automatic bandwidth selection for circular density estimation. *Computational Statistics & Data Analysis*, *52*(7), 3493–3500. https://doi.org/10.1016/j.csda.2007.11.003

Zar, J. H. (1999). *Biostatistical analysis* (4th ed.). Prentice Hall.

---

## Appendix A — Python API Reference

### A.1 Module `circ_stat` — Circular Statistics

Python port of the CircStat MATLAB toolbox (Berens, 2009) with extended and corrected 2D von Mises density function. All functions accept 1-D NumPy arrays of angles in radians unless noted.

#### A.1.1 Utility Functions

| Function | Signature | Description |
|---|---|---|
| `circ_rad2ang` | `(alpha) → ndarray` | Radians → degrees |
| `circ_ang2rad` | `(alpha) → ndarray` | Degrees → radians |
| `circ_dist` | `(x, y) → ndarray` | Element-wise circular difference x − y ∈ (−π, π] |
| `circ_dist2` | `(x, y=None) → ndarray` | All-pairs circular differences D[i,j] = x[i] − y[j] |
| `rmatrix` | `(A) → ndarray` | Flip matrix rows upside-down (display utility; port of `Rmatrix.m`) |

#### A.1.2 Descriptive Statistics

| Function | Signature | Returns | Notes |
|---|---|---|---|
| `circ_r` | `(alpha, w=None, d=0.0) → float` | Mean resultant length R̄ ∈ [0,1] | `d` = bin spacing correction (Zar §26.17) |
| `circ_mean` | `(alpha, w=None) → float` | Mean direction (radians) | arg(Σ w exp(iθ)) |
| `circ_moment` | `(alpha, w=None, p=1, cent=False) → (complex, float, float)` | (mₚ, ρₚ, μₚ) — p-th trigonometric moment | `cent=True` centers on circular mean |
| `circ_var` | `(alpha, w=None) → float` | Circular variance S = 1 − R̄ | Bounded [0, 1] |
| `circ_std` | `(alpha, w=None) → (float, float)` | (s, s₀) | s = √(2(1−R̄)); s₀ = √(−2 ln R̄) |
| `circ_skewness` | `(alpha, w=None) → float` | Circular skewness b | Pewsey (2004) |
| `circ_kurtosis` | `(alpha, w=None) → float` | Circular kurtosis k | Pewsey (2004) |
| `circ_median` | `(alpha) → float` | Circular median | O(n²); suitable for n < 5,000 |

#### A.1.3 von Mises Parameter Estimation

| Function | Signature | Returns | Notes |
|---|---|---|---|
| `circ_kappa` | `(r_or_alpha, w=None) → float` | ML concentration κ | Fisher (1993) p. 88 piecewise approximation; small-sample correction for N < 15 |
| `circ_vmpar` | `(alpha, w=None, d=0.0) → (float, float)` | (μ, κ) | Combines `circ_mean` + `circ_kappa` |

#### A.1.4 2D von Mises KDE

**`circ_vmpdf2(n_ang, phi_i, psi_i, kappa1, kappa2) → (ndarray, int)`**

Evaluates the 2D von Mises product-kernel density on an n_ang × n_ang uniform grid over (−π, π].

Mathematical definition:

```
p[i,j] = C · Σₖ exp(κ₁ cos(φᵢ − φₖ) + κ₂ cos(ψⱼ − ψₖ))
C       = 1 / (4π² N I₀(κ₁) I₀(κ₂))
```

Vectorized implementation:

```python
C = 1.0 / (4 * π² * N * I₀(κ₁) * I₀(κ₂))
A = exp(κ₁ * cos(grid[:,None] − phi_i[None,:]))   # (n_ang, N)
B = exp(κ₂ * cos(grid[:,None] − psi_i[None,:]))   # (n_ang, N)
p = C * (A @ B.T)                                  # (n_ang, n_ang)
```

Complexity: O(n_ang · N) — approximately 100× faster than nested loops at n_ang = 180.

**Returns:** `(p, n_ang)` where p rows index φ and columns index ψ.

**MATLAB fix:** The original `circ_vmpdf2.m` applied κ₁ to both axes. The Python version correctly uses κ₁ for φ and κ₂ for ψ.

#### A.1.5 Uniformity Tests

| Function | Signature | Returns | Notes |
|---|---|---|---|
| `circ_rtest` | `(alpha, w=None, d=0.0) → (pval, z)` | p-value, Rayleigh z | Sensitive to unimodal departure |
| `circ_otest` | `(alpha, sz=None, w=None) → (pval, m)` | p-value, min half-circle count | Effective for multimodal distributions |
| `circ_raotest` | `(alpha) → (p, U, UC)` | Significance level, statistic U, critical value | Table from Russell & Levitin (1995) |
| `circ_vtest` | `(alpha, dir_, w=None, d=0.0) → (pval, v)` | p-value, v statistic | Tests uniformity vs. specified direction |

#### A.1.6 Correlation Coefficients

| Function | Signature | Returns | Reference |
|---|---|---|---|
| `circ_corrcc` | `(alpha1, alpha2) → (rho, pval)` | Circular-circular correlation and p-value | Jammalamadaka & SenGupta (2001), p. 176 |
| `circ_corrcl` | `(alpha, x) → (rho, pval)` | Circular-linear correlation ρ ∈ [0,1] and p-value | Zar (1999), §27 |

---

### A.2 Module `pdb_tools` — Backbone Geometry

Port of MATLAB `pdbTools/calcDihedral.m`, `calcDihedrals.m`, `findAngle.m`, `Torsion.m`, `Rotamer.m`.

| Symbol | Description |
|---|---|
| `CUTOFF = 6.5` | Bond-length sanity cutoff in Å; `calc_dihedral` returns NaN when exceeded |
| `AMINO_ACIDS` | Ordered list of 20 standard amino acid 3-letter codes |
| `CHIS` | Dict mapping amino acid → list of χ-angle atom quadruples (from `Rotamer.m`) |
| `find_angle(u, v)` | Angle in radians between two 3D vectors |
| `calc_dihedral(p1, p2, p3, p4, cutoff=∞)` | Signed dihedral angle (radians); NaN if any bond > cutoff |
| `calc_dihedrals(prev_c, curr_n, curr_ca, curr_c, next_n)` | Returns (φ, ψ, dC_CA, dN_CA, dN_C, dP_plane, ang_N_CA_C) for one residue |
| `torsion(structure)` | List of per-residue dicts with backbone angles and bond metrics for a BioPython structure |
| `rotamer(structure)` | List of per-residue dicts with χ₁–χ₄ angles |
| `extract(structure)` | Full extraction (backbone + side-chain) → Pandas DataFrame |

---

### A.3 Module `extractor` — Phase 1 Batch Pipeline

Port of MATLAB `Protein.m`.

**`extract_from_file(tsv_path, pdb_dir, amino_dir, rotamer_w3, skip_existing=True, pairs=True, triplets=True, max_count=None) → dict[str, int]`**

Main entry point for Phase 1. Reads PDB IDs from TSV, downloads structures, extracts per-residue geometry, and appends to CSV files.

| Parameter | Description |
|---|---|
| `tsv_path` | Path to `data/pdb1A.tsv` (one PDB ID per line, no header) |
| `pdb_dir` | Directory for downloaded PDB files (created on first run; `skip_existing=True` caches) |
| `amino_dir` | Output directory for `<AMINO>.csv` and pair/triplet CSV files |
| `rotamer_w3` | Path for triplet-window rotamer CSV |
| `pairs` | Write `<AA1><AA2>.csv` consecutive residue pair files |
| `triplets` | Write `rotamer_w3.csv` triplet records |
| `max_count` | Limit number of structures (for testing) |

Returns: `{amino_code: total_residues_extracted}` for all 20 amino acids.

---

### A.4 Module `analysis` — Phase 2 Statistical Potential Computation

Port of MATLAB `ForcesCont.m` and `Analysis.m`.

| Function | Signature | Description |
|---|---|---|
| `load_amino_data` | `(csv_path) → ndarray\|None` | Load per-amino CSV → (N, 12) float array; drops NaN φ/ψ rows |
| `compute_potential` | `(a1, a2, n_ang=180) → dict\|None` | 2D von Mises KDE + −log(p) + gradients for one dihedral pair; None if < 5 valid points |
| `analyse_amino` | `(amino_name, csv_path, n_ang=180) → dict\|None` | All dihedral pair potentials for one amino acid (up to 6 pairs) |
| `analyse_all` | `(amino_dir, cont_dir=None, n_ang=180, amino_list=None) → dict` | Batch: runs `analyse_amino` for all available CSVs; saves `.npz` if `cont_dir` given |
| `save_potential` | `(result, cont_dir) → Path` | Saves one amino acid result to `Mises<AMINO><n_ang>.npz` |
| `load_potential` | `(cont_dir, amino_name, n_ang=180) → dict\|None` | Loads saved `.npz`; returns None if absent |
| `circular_stats` | `(alpha) → dict` | Descriptive circular stats: n, mean_deg, R, var, std_deg, std0_deg, skewness, kurtosis |
| `amino_stats` | `(csv_path) → dict` | Full stats: circular stats for φ/ψ/χ₁–χ₄; linear stats for bond lengths |

**`.npz` File Schema** (`contData/Mises<AMINO><n_ang>.npz`):

For each dihedral pair k = 0, 1, …, n_potentials − 1:

| Key | Dtype | Shape | Description |
|---|---|---|---|
| `n_total` | int64 | scalar | Total residues from CSV |
| `n_ang` | int64 | scalar | KDE grid resolution |
| `n_potentials` | int64 | scalar | Number of pairs computed |
| `pair_k` | str | [2] | Angle names, e.g. `['phi', 'psi']` |
| `E_k` | float64 | (n, n) | Statistical potential −log p |
| `Gx_k` | float64 | (n, n) | Gradient in φ (column) direction |
| `Gy_k` | float64 | (n, n) | Gradient in ψ (row) direction |
| `kappa_k` | float64 | [2] | Fitted [κ₁, κ₂] |
| `n_k` | int64 | scalar | Data points used for this pair |

**Loading example:**

```python
from protein_periodicity.analysis import load_potential

data = load_potential("python/contData", "LEU", n_ang=180)
pot  = data["potentials"][0]   # first pair (phi–psi)
E    = pot["E"]                # (180, 180)  −log p
Gx   = pot["Gx"]              # (180, 180)  ∂E/∂φ
Gy   = pot["Gy"]              # (180, 180)  ∂E/∂ψ
```
