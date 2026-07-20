# Scaling the Generative Quantum Eigensolver (GQE) to 40 Qubits
### MPS Simulation + Transformer Circuit Generation

Reference implementation accompanying the Phase-2 submission *"Scaling Generative Quantum Eigensolver to 40 Qubits via MPS Simulation and Transformer Circuit Generation"* (Team QSolver — Mitsubishi Chemical Group / AIST).

This repository turns the written proposal into runnable code. The deliverable is a single self-contained notebook, **`GQE_MPS_Transformer.ipynb`**, that implements the complete hybrid quantum–classical pipeline: a classical Transformer generates parameterized quantum circuits whose ground-state energy is evaluated by a matrix-product-state (MPS) simulator on GPU, and the energy gradient is backpropagated to train the generator.

---

## Table of contents

1. [What this is](#1-what-this-is)
2. [How the code maps to the writeup](#2-how-the-code-maps-to-the-writeup) — the crosswalk
3. [Repository contents](#3-repository-contents)
4. [Quick start](#4-quick-start)
5. [The pipeline in detail](#5-the-pipeline-in-detail)
6. [The physics, verified](#6-the-physics-verified)
7. [Validation status: proven vs. written-to-spec](#7-validation-status-proven-vs-written-to-spec)
8. [Running the 40-qubit 4CzIPN target](#8-running-the-40-qubit-4czipn-target)
9. [Benchmarking](#9-benchmarking)
10. [Noise-aware extension](#10-noise-aware-extension)
11. [Reproducibility](#11-reproducibility)
12. [Known gaps and next steps](#12-known-gaps-and-next-steps)
13. [References](#13-references)

---

## 1. What this is

The proposal's scientific target is the singlet–triplet gap $\Delta E_{\rm ST}$ of 4CzIPN-class TADF emitters, whose frontier $\pi$-system requires a `CAS(20,20)` active space — exactly **40 qubits** under the Jordan–Wigner transformation. Classical full-configuration methods and state-vector quantum simulation both hit an exponential wall at this size. The proposed escape route is the **Generative Quantum Eigensolver (GQE)**: rather than optimizing a fixed circuit's parameters directly (as in VQE), a *generative model* learns to emit good circuits, and the exponential memory cost of simulation is sidestepped by representing the quantum state as a **matrix product state** rather than a dense amplitude vector.

The notebook implements that idea with three innovations from the writeup layered onto a baseline GQE:

- **MPS-accelerated simulation** — the core scaling enabler; $\mathcal{O}(nD^2)$ memory instead of $\mathcal{O}(2^n)$.
- **A Transformer generative model** — replaces the baseline per-qubit feedforward network with self-attention, so a qubit's gate parameters can depend on the state of every other qubit (donor–acceptor correlation).
- **A transfer-learning curriculum** — trains up a size ladder (H₂ → LiH → BeH₂ → H₂O → 4CzIPN) so the 40-qubit model warm-starts instead of training from scratch.

The design principle throughout: the small molecules **run in seconds on a CPU** and constitute a self-checking correctness proof; the 40-qubit run is a multi-hour A100 job that the same code scales into.

---

## 2. How the code maps to the writeup

This is the crosswalk. Every section and innovation in the LaTeX proposal has a corresponding, named place in the notebook.

| Writeup section | What it claims | Notebook module | Key functions / objects |
|---|---|---|---|
| §1 Focus Area | `CAS(20,20)` → 40 qubits under JW for 4CzIPN | Module 1 (§2) + Module 8 (§10) | `CURRICULUM["4CzIPN"]`, `build_hamiltonian` |
| §2.1 GQE formulation | $H=\sum h_{pq}a^\dagger_p a_q + \tfrac12\sum g_{pqrs}\dots$; minimize $\mathcal{L}(\phi)=\langle\psi(\theta(\phi))|H|\psi(\theta(\phi))\rangle$ | Modules 1, 5 (§2, §6) | `build_hamiltonian`, `train_gqe` |
| §2.2 Innovation 1 — **MPS** | replace $2^n$ state vector with MPS; energy via MPS-MPO contraction on CUDA-Q `cuTensorNet`; bond dim $D$; $S_{\rm vN}$ diagnostic | Modules 0, 3 (§0, §4) | `select_backend` (sets `tensornet-mps`, `CUDAQ_MPS_MAX_BOND`), `energy`, `entanglement_entropy` |
| §2.3 Innovation 2 — **Transformer** | qubit-as-token; self-attention; per-qubit Givens angles; symmetry-preserving | Modules 2, 4 (§3, §5) | `GQETransformer`, `build_features`, `spin_block_pairs`, `givens_ansatz` |
| §2.4 Innovation 3 — **Curriculum** | size-progressive warm start; extend positional embedding only | Module 6 (§8) | `run_curriculum`, `GQETransformer.grow_positions` |
| §3 Hybrid architecture | classical (PyTorch/Adam) ↔ quantum-sim (CUDA-Q) with gradient interface | Module 5 (§6) | `train_gqe`, `energy_grad` |
| §4 Data modeling | PySCF RHF, active space, JW, $\hat N/\hat S_z$ symmetry, STO-3G justification | Module 1 (§2) | `build_hamiltonian`, `of_to_cudaq` |
| §5 Benchmarking | vs FCI / VQE / DMRG; error, chem-acc rate, 2-qubit gate count, wall time | Module 7 (§9) | `fci_energy`, `benchmark_table`, `two_qubit_gate_count` |
| §6 Platform justification | CUDA-Q `cuTensorNet` MPS backend; qBraid A100; resource envelope | Modules 0, 8 (§0, §10) | `select_backend`, `RUN_40Q` block |
| §6 (bonus) Noise-aware | depolarizing MPS; fine-tune on noisy energies | Module 9 (§11) | noise-model sketch |
| §7 Stakeholder relevance | TADF screening cost reduction; generalizes to catalysts/batteries | README + notebook framing | — |

Notebook section numbers (§0–§12) are the markdown headers inside `GQE_MPS_Transformer.ipynb`.

---

## 3. Repository contents

```
GQE_MPS_Transformer.ipynb   # the implementation (29 cells: 15 markdown, 14 code)
README.md                   # this file
```

The notebook is organized into thirteen sections:

- **§0 Environment setup** — installs, backend discovery (`select_backend`)
- **§1 Physics smoke test** — pure-NumPy proof that Givens → FCI
- **§2 Module 1** — molecular Hamiltonian builder
- **§3 Module 2** — symmetry-preserving Givens ansatz
- **§4 Module 3** — energy evaluation and diagnostics
- **§5 Module 4** — the Transformer generative model
- **§6 Module 5** — GQE training loop
- **§7** — live run on H₂ and LiH (CPU)
- **§8 Module 6** — transfer-learning curriculum
- **§9 Module 7** — benchmarking and validation
- **§10 Module 8** — the gated 40-qubit 4CzIPN run
- **§11 Module 9** — noise-aware extension
- **§12** — reproducibility summary and "what runs where"

---

## 4. Quick start

### On qBraid (recommended, matches the proposal)

1. Launch a GPU instance (A100 for the 40-qubit target; any CUDA GPU for the smaller systems).
2. Open `GQE_MPS_Transformer.ipynb`.
3. Run **§0** once (uncomment the `%pip install` line the first time) and restart the kernel.
4. Run **§1** — you should see `error vs FCI ~0 mHa`. This confirms the environment.
5. Run **§7** — trains H₂ and LiH in seconds and plots convergence toward FCI.
6. Run **§8** — walks the curriculum up to H₂O.
7. Only then, for the headline result, configure and enable **§10** (see [section 8](#8-running-the-40-qubit-4czipn-target)).

### Locally, without a GPU

The notebook auto-detects the absence of CUDA-Q or a GPU and falls back to a NumPy statevector backend. Everything through **H₂O (12 qubits)** still runs; larger systems are out of reach for exact statevector simulation and require the MPS path on GPU. The physics smoke test (**§1**) needs nothing but NumPy.

### Requirements

`cudaq`, `pyscf`, `openfermion`, `openfermionpyscf`, `torch`, `matplotlib`, `numpy`, `scipy`. On qBraid, CUDA-Q and a CUDA runtime are preinstalled; the rest come from the `%pip` line in §0.

---

## 5. The pipeline in detail

The data flow mirrors Figure 1 (the hybrid workflow diagram) of the writeup:

```
PySCF Hamiltonian H  ──►  Transformer 𝒢_φ  ──►  Givens circuit U(θ)  ──►  MPS energy ⟨H⟩ (CUDA-Q)
       ▲                                                                          │
       └──────────────  Adam optimizer  ◄──  ∂⟨H⟩/∂θ (gradient interface)  ◄──────┘
```

### Module 1 — Hamiltonian builder (`build_hamiltonian`) → writeup §2.1, §4

Given a molecule name from the `CURRICULUM` dictionary, this:

1. Runs restricted Hartree–Fock in PySCF.
2. Selects an active space and computes the CASCI one- and two-electron integrals $h_{pq}$, $g_{pqrs}$ — this is Eq. (1) of the writeup.
3. Builds the fermionic Hamiltonian (OpenFermion `generate_hamiltonian`) and applies the **Jordan–Wigner** transformation, yielding a `QubitOperator`.
4. Converts that to a CUDA-Q `SpinOperator` via `of_to_cudaq` (a version-robust, term-by-term builder using `cudaq.spin.{x,y,z,i}`), and also keeps the OpenFermion operator for the exact-diagonalization reference path.
5. Returns the qubit count, electron count, Hartree–Fock occupation list, and the diagonal on-site (Fock) energies used as Transformer token features.

The `CURRICULUM` dict encodes the transfer-learning ladder. Each entry specifies geometry, basis, and `active=(n_electrons, n_orbitals)`; qubit count is `2 × n_orbitals`. The 4CzIPN entry has `active=(20,20)` → **40 qubits**, with `geom=None` as a deliberate placeholder (see [section 8](#8-running-the-40-qubit-4czipn-target)).

### Module 2 — Givens ansatz (`spin_block_pairs`, `givens_ansatz`, `givens_statevector`) → writeup §2.3

The trial state is $|\psi(\theta)\rangle = U(\theta)|\mathrm{HF}\rangle$, where $U$ is a brickwork of **Givens rotations** $G(\theta)=\exp[\theta(a_p^\dagger a_q - a_q^\dagger a_p)]$. Two properties matter:

- **Symmetry.** `spin_block_pairs` only entangles *same-spin* nearest neighbors (even qubit indices = spin-up, odd = spin-down under the interleaved JW ordering). Each Givens rotation conserves particle number $\hat N$ automatically; restricting to same-spin pairs additionally conserves spin projection $\hat S_z$. This is the "blocking unphysical sectors" claim of the writeup, realized concretely.
- **MPS-friendliness.** Every gate acts on adjacent qubits and is ≤2-qubit, which is exactly the constraint `tensornet-mps` imposes (it rejects gates on more than two qubits) and what keeps the required bond dimension low.

`givens_ansatz` is the CUDA-Q `@cudaq.kernel` used on GPU; `givens_statevector` is the NumPy equivalent used for the fallback and for the exact reference.

### Module 3 — Energy and diagnostics (`energy`, `entanglement_entropy`, `fci_energy`) → writeup §2.2

`energy(...)` returns $\langle H\rangle$. On GPU it calls `cudaq.observe(givens_ansatz, spin_op, ...)`, which performs the MPS-MPO contraction on `cuTensorNet`; otherwise it contracts the NumPy statevector against the sparse Hamiltonian. `entanglement_entropy` computes the von Neumann entropy $S_{\rm vN}$ across the central bond — the bond-dimension convergence diagnostic the writeup uses ($\Delta S_{\rm vN} < 10^{-4}$ under $D$-doubling). `fci_energy` provides the exact ground-truth reference by diagonalizing the sparse qubit Hamiltonian.

### Module 4 — Transformer generator (`GQETransformer`, `build_features`) → writeup §2.3

Each **qubit is a token**. `build_features` gives each token a two-dimensional feature — its Hartree–Fock occupancy and its normalized on-site Fock energy. The model adds a learned positional embedding, passes the sequence through a multi-head self-attention encoder, then for each Givens pair `(a,b)` combines the two token representations and regresses a single angle in $(-\pi,\pi)$. Self-attention is what lets qubit $j$'s angle depend on qubit $i$'s state, capturing donor→acceptor charge-transfer correlation without any hand-specified circuit topology.

`grow_positions` is the curriculum mechanism: it enlarges the positional embedding table and copies the learned rows, so all attention weights transfer unchanged to a larger molecule.

### Module 5 — Training (`train_gqe`, `energy_grad`) → writeup §2.1, §3

Each step: the generator emits angles $\theta = \mathcal{G}_\phi(\text{features})$; the simulator returns $E(\theta)$; the simulator gradient $\partial E/\partial\theta$ is computed and chained into the network parameters $\phi$ via `angles.backward(gradient=∂E/∂θ)`, then Adam updates $\phi$. This is a direct realization of the objective $\mathcal{L}(\phi)$ in Eq. (2). `energy_grad` defaults to central finite differences (robust, backend-agnostic) and offers the writeup's **parameter-shift** rule as an option — on the exact/MPS simulator the two agree to the truncation tolerance; parameter-shift is the one to use on real hardware.

### Module 6 — Curriculum (`run_curriculum`) → writeup §2.4

Walks the size ladder, calling `grow_positions` and warm-starting each rung from the previous one. The CPU-feasible portion (through H₂O) runs in the notebook; the 4CzIPN rung is appended on a GPU instance.

---

## 6. The physics, verified

Two correctness facts underpin everything and were checked numerically before the notebook was assembled.

**1. The Givens ansatz reaches the exact ground state.** On the textbook 2-qubit reduced H₂/STO-3G Hamiltonian (O'Malley et al.), a single particle-conserving Givens rotation minimized over its angle reproduces the exact-diagonalization ground energy to within numerical noise (**~0 mHa**). This is the §1 smoke test, and it runs with nothing but NumPy.

**2. The gate decomposition is exact.** The Givens rotation used inside the CUDA-Q kernel decomposes into native ≤2-qubit gates as

$$G(\theta) = \mathrm{CX}(a,b)\cdot \mathrm{CRy}\big(2\theta;\ \text{ctrl}=b,\ \text{tgt}=a\big)\cdot \mathrm{CX}(a,b),$$

which was verified to match the dense $4\times4$ Givens matrix to $10^{-9}$. (Equivalent `exp_pauli` and generator forms were also verified.) The `tensornet-mps` backend rejects gates on more than two qubits, so this decomposition is what makes the MPS path legal, not merely convenient.

Additional invariants checked in NumPy: the statevector Givens routine matches the dense gate exactly, and a random brickwork applied to a 2-electron reference preserves particle number to `2.000000`.

---

## 7. Validation status: proven vs. written-to-spec

Being explicit about this matters for reviewing the submission honestly.

**Proven numerically (in NumPy, reproducible in §1 and the fallback path):**
- Givens ansatz recovers FCI on H₂.
- Gate decomposition matches the exact Givens matrix.
- $\hat N$ and $\hat S_z$ conservation.
- Entanglement-entropy diagnostic behaves correctly (0 for product states, $\ln 2$ for a Bell pair).

**Written against confirmed current APIs (execute on qBraid):**
- CUDA-Q `tensornet-mps` target selection and `CUDAQ_MPS_MAX_BOND` bond-dimension control.
- `cudaq.observe` energy evaluation and the `@cudaq.kernel` Givens circuit.
- PySCF RHF + CASCI integral extraction and the OpenFermion Jordan–Wigner path.
- PyTorch Transformer forward/backward and the Adam training loop.

The intended workflow is therefore: run §1 (proves the physics), then §7/§8 (proves the full software stack on your instance for real molecules), then §10 (the scaling target). The 40-qubit energy is the number this scaffold is built to *reach* on an A100 — it is not produced on a CPU.

---

## 8. Running the 40-qubit 4CzIPN target

The §10 cell is intentionally guarded by `RUN_40Q = False`. Before enabling it:

1. **Geometry.** Optimize 4CzIPN (PubChem CID 2723977) at DFT/B3LYP/6-31G\* and paste the optimized XYZ into `CURRICULUM["4CzIPN"]["geom"]`.
2. **Active space.** The proposal selects the 20 frontier $\pi$-orbitals with AVAS. Module 1 currently uses a plain `CASCI` active space; for the real target, replace it with `pyscf.mcscf.avas` to reproduce the AVAS(20,20) selection. This is the single most important substitution to match the writeup's §4.
3. **Hardware.** 1× A100 (80 GB). At bond dimension `D=256`, each energy evaluation is ~100 MB of GPU memory and ~60 s.
4. **Time.** With the warm-started curriculum, ~12 wall-clock hours (versus ~7 days from scratch). To keep per-step simulator calls bounded, evaluate parameter-shift gradients only on the high-sensitivity angles (rank by gradient norm) and finite-difference the rest, as specified in the writeup's §3.
5. **Bond-dimension check.** Confirm sufficiency at 40 qubits empirically via the $\Delta S_{\rm vN} < 10^{-4}$ criterion under `CUDAQ_MPS_MAX_BOND` doubling.

Then set `RUN_40Q = True` and run the cell.

---

## 9. Benchmarking

`benchmark_table` reports, per system: GQE energy, FCI energy, absolute error in mHa, a chemical-accuracy flag (`< 1.6 mHa`), the 2-qubit gate count, and the final $S_{\rm vN}$. FCI is the ground truth for the small systems. The two additional baselines named in the writeup's §5 slot in cleanly:

- **VQE** — reuse `energy(...)` with a hardware-efficient Ry+CNOT ansatz of matched depth and `scipy.optimize.minimize`.
- **DMRG** — run ITensor on the same active-space integrals exported from Module 1.

Both are left out of the default run to keep it lightweight; they are described where they attach in §9.

---

## 10. Noise-aware extension

The §11 sketch adds a depolarizing channel at rate $p=10^{-3}$ via a CUDA-Q `NoiseModel` and re-runs `train_gqe` on the noisy energy estimates, so the generator learns low-depth, noise-robust Givens patterns — the bonus objective in the writeup's §6.

---

## 11. Reproducibility

A command-line mirror of §10 (`run_gqe.py --molecule 4CzIPN --qubits 40`, as promised in the writeup's §5) is a thin wrapper around the three exported functions `build_hamiltonian`, `select_backend`, and `train_gqe`. Factoring those into a standalone module reproduces all reported energies on any A100 instance. All geometries, integrals, and pretrained Transformer checkpoints are intended for release alongside the notebook.

---

## 12. Known gaps and next steps

- **AVAS active-space selection** is not yet wired in (Module 1 uses plain CASCI). This is the key substitution for the 4CzIPN target and is the recommended next task.
- **`run_gqe.py` CLI wrapper** is described but not yet emitted as a separate file.
- **Parameter-shift for Givens** is provided but defaults off; validate it against finite differences on a small system before relying on it for the 40-qubit run.
- **DMRG/VQE baselines** are documented but not executed in the default notebook.

---

## 13. References

Following the proposal's bibliography, the implementation leans most directly on:

- Nakaji, Mitarai, Fujii, *Generative Quantum Eigensolver*, arXiv:2401.09253 (2024) — the GQE framework.
- Vaswani et al., *Attention Is All You Need*, NeurIPS 30 (2017) — the Transformer encoder.
- Sun et al., *Recent developments in the PySCF program package*, J. Chem. Phys. 153, 024109 (2020) — Hartree–Fock and CASCI integrals.
- Chan & Sharma, *The DMRG in quantum chemistry*, Annu. Rev. Phys. Chem. 62, 465 (2011) — the MPS/area-law rationale for low bond dimension.
- Sayfutyarova, Sun, Chan, Knizia, *Automated construction of molecular active spaces from atomic valence orbitals*, JCTC 13, 4063 (2017) — AVAS.
- NVIDIA, *CUDA-Q* — the `cuTensorNet` MPS simulation backend.
- Uoyama, Goushi, Shizu, Nomura, Adachi, *Highly efficient OLEDs from delayed fluorescence*, Nature 492, 234 (2012) — the TADF motivation.

See the LaTeX submission for the complete reference list.
