# Literature Notes — C5 Information-Capacity Bound

Working notes to support derivation of the Holevo+Fano bound in `c5-info-bound.tex`.

## Abbas et al. 2021 — "The power of quantum neural networks"

**Main claim:** The *effective dimension* $d_{\gamma,n}$, a Fisher-information-based capacity measure, yields a data-dependent generalization bound that is tighter than VC-dimension arguments, and well-designed quantum neural networks can achieve strictly higher effective dimension than classical networks with the same number of trainable parameters.

**Key inequality:** Definition 3.1 (Eq. 2) defines the effective dimension of a parametric model class $\mathcal{M}_\Theta$ over a $d$-dimensional parameter space and $n$ data samples as
\[
d_{\gamma, n}(\mathcal{M}_\Theta) \;:=\; 2 \, \frac{\log\!\left( \frac{1}{V_\Theta} \int_\Theta \sqrt{\det\!\left( \mathrm{id}_d + \frac{\gamma n}{2\pi \log n} \hat{F}(\theta)\right)} \, \mathrm{d}\theta \right)}{\log\!\left(\frac{\gamma n}{2\pi \log n}\right)},
\]
where $\hat{F}(\theta)$ is the (normalized) Fisher information matrix. Theorem 3.2, Eq. (5), then gives, for a loss bounded in $[-B/2, B/2]$ and under Lipschitz / full-rank regularity assumptions, a bound on the tail of the uniform generalization gap:
\[
\mathbb{P}\!\left(\sup_\theta |R(\theta) - R_n(\theta)| \ge 4 M \sqrt{\frac{2\pi \log n}{\gamma n}} \right) \;\le\; c_{d,\Lambda} \left( \frac{\gamma n^{1/\alpha}}{2\pi \log n^{1/\alpha}} \right)^{d_{\gamma,n}/2} \exp\!\left( -\frac{16 M^2 \pi \log n}{B^2 \gamma}\right).
\]

**Relevance to us:** Effective dimension is a sharper capacity proxy than parameter count for our 4-qubit VQC with StronglyEntanglingLayers. It rank-orders quantum vs. classical heads trained on the same projection+loss. However it bounds the *generalization gap* between train and test risk, not the achievable test accuracy itself; it says nothing about the ceiling imposed by information destroyed at the measurement step. Holevo+Fano bound the *information floor* regardless of parameter count or Fisher spectrum. The two bounds are therefore complementary: Abbas tells us how well the capacity-to-data ratio generalizes; Holevo tells us how much signal the architecture can carry at all.

**Caveats:** The bound assumes (i) the Fisher matrix $\hat F(\theta)$ has full rank everywhere (violated on barren plateaus and at symmetry points), (ii) Lipschitz continuity of the parameterized density map, and (iii) a loss bounded in an interval. Practically, numerical estimation of $\det(\hat F)$ on a 4-qubit circuit requires Monte Carlo sampling of $\Theta$, which is noisy. The bound is also uniform over $\Theta$; tighter data-dependent versions exist but are not used here. Finally, the result is a *high-probability bound*, not an exact accuracy ceiling — it does not replace Holevo for our purposes.

## Caro et al. 2022 — "Generalization in quantum ML from few training data"

**Main claim:** The generalization error of a quantum machine learning model with $T$ parameterized local quantum channels (gates) trained on $N$ i.i.d. samples is in $\mathcal{O}(\sqrt{T \log T / N})$ with high probability (Theorem 1, Eq. 5, p. 3), and improves to $\mathcal{O}(\sqrt{K \log(MT) / N} + \sum_{k>K} M\Delta_k)$ when only $K \ll T$ gates change substantially during optimization (Theorem 3, Eq. 7, p. 3).

**Key inequality:** Theorem 1 (Basic QMLM bound), paraphrased: for a quantum learning model with $T$ trainable two-qubit local channels trained on $N$ i.i.d. samples $S_N$, with high probability over $S_N$,
\[
\mathrm{gen}(\alpha^*) \;\in\; \mathcal{O}\!\left( \sqrt{\frac{T \log T}{N}} \right).
\]
The refined Theorem (light cone / few-changed-gates) replaces $T$ by $K$, the number of gates whose parameter moves substantially during training.

**Relevance to us:** Our 4-qubit circuit uses a StronglyEntanglingLayers ansatz with on the order of $T \approx 12$-$36$ trainable two-qubit+single-qubit gates, trained on $N \approx 10^3$-$10^4$ routing examples. Caro's bound scales as $\sqrt{T \log T / N} \sim 0.06$-$0.36$ for these values (natural log, constants hidden in $\mathcal{O}$), i.e., essentially negligible compared to the Holevo+Fano accuracy floor our paper derives. This is exactly why our paper argues the bottleneck is *capacity*, not *sample complexity*: with realistic $N$ we are nowhere near Caro's $\sqrt{T \log T / N}$ regime.

**Caveats:** The bound assumes each trainable channel acts locally (bounded locality $k$), the loss is bounded, and training produces the empirical risk minimizer $\alpha^*$. It does not distinguish quantum from classical capacity — a classical neural network with the same number of parameters obeys an analogous Rademacher-style $\sqrt{\text{params}/N}$ bound. Hence Caro is a *sufficient* condition for good generalization but does not predict *achievable risk*; it cannot be used to argue quantum advantage or disadvantage on its own.

## Du et al. 2020 — "Expressive power of parameterized quantum circuits"

**Main claim:** Multilayer parameterized quantum circuits (MPQCs) with $O(\mathrm{poly}(N))$ single-qubit and CNOT gates on $N$ qubits strictly dominate several classical generative models in expressive power, under the complexity-theoretic assumption that the polynomial hierarchy does not collapse; the expressive ordering is $\mathrm{MPQC} > \mathrm{DBM} > \mathrm{long\text{-}range\ RBM} > \mathrm{short\text{-}range\ RBM}$ (Theorem 1, p. 6). TPQCs do not appear in Theorem 1; they appear only in the bond-dimension relation of Eq. 13 ($E_\mathrm{MPQC} = E_\mathrm{DBM} = E_\mathrm{L\text{-}RBM} = E_\mathrm{TPQC} > E_\mathrm{S\text{-}RBM}$).

**Key inequality:** Theorem 1 (verbatim, PRR version, p. 6): *"The expressive power of MPQCs with $O(\mathrm{poly}(N))$ single qubit gates and CNOT gates and classical neural networks with $O(\mathrm{poly}(N))$ trainable parameters, where $N$ refers to the number of qubits or the visible units, can be ordered as follows: MPQCs $>$ DBM $>$ long-range RBM $>$ short-range RBM, unless the polynomial hierarchy (PH) collapses."* Lemma 1 (p. 6): MPQCs and TPQCs with $N$ qubits and $O(\mathrm{poly}(N))$ blocks, each containing $O(N)$ trainable parameters and at most $N$ CNOT gates, can efficiently represent MPSs with bond dimensions $D = O(\exp(N))$. Theorem 4 (Appendix C, p. 13) concerns the efficient simulation of IQP circuits by MPQCs; it is not an MPS or ansatz-specific bound.

**Relevance to us:** Our 4-qubit StronglyEntanglingLayers ansatz is an MPQC in the sense of Du et al.: a chain of layered single-qubit rotations plus CNOT entangling gates, linear in layer count. Du's Theorem 1 says this family has strictly larger expressive reach than a matched-parameter classical net *for generative/distribution tasks*. But our task is **discriminative 10-class routing**, where expressive reach is gated by the classical measurement record $M \in [-1,1]^4$. The take-away: expressivity of the pre-measurement state space is large, but the bottleneck is the measurement-induced projection — precisely the quantity Holevo bounds at 4 bits.

**Caveats:** The expressivity ordering is complexity-theoretic (contingent on $\mathrm{PH}$ not collapsing) and defined for generative sampling, not for classification accuracy. The paper does not provide a *no-free-lunch* theorem in our required sense, and does not mention StronglyEntanglingLayers anywhere in the PRR version; our ansatz maps onto their generic MPQC framework but with a specific rotation/entangler pattern whose expressivity is not separately bounded. Du et al. also assume noiseless execution; on real NISQ hardware or our PennyLane simulator with shot noise, expressivity degrades.

## Holevo 1973 — "Bounds for the quantity of information..."

**Main claim:** Maximum classical information extractable from an $N$-qubit quantum state is bounded by the von Neumann entropy, which is at most $N$ bits.

**Key inequality:**
\[ I(M; Y) \leq S(\rho) \leq N \]
where $S$ is von Neumann entropy and the measurement $M$ is any POVM.

**Modern form:** See Nielsen & Chuang, "Quantum Computation and Quantum Information", §12.1.1. The bound $I(M; Y) \le N$ is tight when the state is a uniform mixture over a basis of size $2^N$.

**Relevance to us:** Our 4-qubit VQC with single-qubit PauliZ measurements produces $M \in [-1, 1]^4$. By Holevo, $I(M; Y) \le 4$ bits regardless of how informative the input embedding is.

**Caveats:** The bound is on mutual information, not accuracy directly — must compose with Fano to get error floor. Also, single-qubit Z measurements (as opposed to full tomography) may not saturate the bound; the achievable MI is typically strictly less. The original 1973 Russian-language publication is not freely available online; we cite it for historical priority and refer to Nielsen & Chuang §12.1.1 for the modern textbook statement and proof.

## Fano 1961 — "Transmission of Information"

**Main claim:** For any estimator $\hat{Y}$ of $Y$ from observation $M$ over $K$ classes, the error probability is bounded below by a function of $H(Y) - I(M;Y)$.

**Key inequality:**
\[ H(Y | \hat{Y}) \leq 1 + P_\mathrm{err} \cdot \log_2(K - 1) \]
Rearranging and using $H(Y | \hat{Y}) \geq H(Y | M) = H(Y) - I(M;Y)$:
\[ P_\mathrm{err} \geq \frac{H(Y) - I(M;Y) - 1}{\log_2(K-1)} \]

**Relevance to us:** Combined with Holevo cap $I \le 4$ bits and $H(Y) = \log_2(10) \approx 3.32$ bits, Fano gives a closed-form error floor for any 4-qubit VQC classifier on our 10-class task.

**Caveats:** Fano is tight only for specific input distributions; for real data with structure it is generally a loose bound (overestimates achievable accuracy). The bound assumes deterministic estimator and classical noise; extension to quantum measurement noise is direct (apply to the classical record $M$ after measurement).

## Combined framework for our paper

Composing Holevo (MI $\le N$ for $N$ qubits) with Fano (error $\ge (H(Y) - I - 1) / \log_2(K-1)$) gives:

\[
\mathrm{acc}_{\max}(N, K, I_{xy}) \;=\; 1 - \max\!\left(0, \; \frac{\log_2 K - \min(I_{xy}, N) - 1}{\log_2(K - 1)} \right)
\]

Where $I_{xy}$ is the mutual information between the projected embedding and class labels. This is the equation that anchors Paper A §3.

## Papers we do NOT cite (out of scope)

- Quantum advantage papers (Google Sycamore, etc.) — our bound is architecture-agnostic, independent of quantum speedup claims.
- PAC-Bayes bounds for classical NNs — we target a specific quantum-bottleneck phenomenon.
- Kernel methods / QSVM — different architecture, different bound.
