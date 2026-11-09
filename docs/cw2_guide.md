# CW2 theory report: planning guide (the report must be your own writing, about 2500 words)

Requirements from the brief: title page (student ID, project title, word count), sections Q1–Q4, labelled
diagrams, notation, consistent referencing, cross-references to numbered CW1 figures/results.
A descriptive catalogue of architectures scores poorly; explain **why** choices work and **where they fail**.

## Q1 Fundamentals and backprop (10 marks, ~450 words + calculations)
- **(a)** Biological analogy and 2 limitations (e.g. spikes and timing vs real-valued activations; no
  biological evidence for exact backpropagated error signals; dendritic computation). Perceptron = one linear
  threshold unit -> linear boundary; draw XOR's four points and show no single line separates them; show a
  2-hidden-unit construction (e.g. OR and NAND units feeding AND) and explain that hidden units re-map inputs
  into a space where the classes are linearly separable.
- **(b)/(c)** Do the calculation by hand, showing every chain-rule step. Then check your numbers with
  `python scripts/check_q1.py`. Distinguish *prediction error* (ŷ − y, signed) from *loss* (½(ŷ − y)², scalar).
  Three factors in a hidden error signal: output error × outgoing weight × activation derivative (ReLU'(z)).
  Not recurrent: gradients flow backwards during *training* computation only; at inference information flows
  forwards, with no state carried between inputs.
- **(d)** One step reduces loss on one example only; convergence needs repeated steps on all data, and
  generalisation needs held-out evaluation. Stopping criterion: validation early stopping (cite CW1 T3c, fig09).

## Q2 Architectures and learning (12 marks, ~750 words)
- **(a)** Matrix: rows MLP, RBF, Kohonen SOM, Elman RNN, SNN; columns representation, connectivity/state,
  learning mechanism, supervision, suitable application, limitation. Key accuracy points the brief checks:
  - Elman: the *temporal state* is the context layer copied from the hidden layer; it is trained by
    backpropagation through time. State and learning rule are different things.
  - SNN: information in spike timing/rates; learning varies (STDP, surrogate-gradient training, ANN-to-SNN
    conversion). Do not imply a single rule.
  - RBF: localised Gaussian units; often centres by clustering, then linear output weights.
  - SOM: competitive learning plus neighbourhood update on a topology-preserving map; unsupervised.
- **(b)** Supervised backprop minimises a loss against labels; evaluated with held-out error. SOM minimises
  quantisation error / preserves topology without labels; evaluated by quantisation and topographic error.
  A SOM cluster index has no class meaning until mapped to labels (and clusters need not match classes).
- **(c)** Static tabular -> MLP (or tree baseline); an RNN is unsuitable because there is no order to exploit.
  Online sensor streams -> recurrent or windowed models; a SOM or static MLP on single readings ignores history.
  **Relate to CW1:** on FD001 the LSTM's mean test RMSE (13.27 ± 0.43) was slightly lower than the window
  MLP's (13.40 ± 0.44), but the 0.13 gap is well inside the seed spread, so an LSTM advantage is not
  established; on validation the three models were indistinguishable (≈12.8), and both networks were beaten
  by the random forest on test (12.13 ± 0.13; notebook 03). Note that the window MLP still sees temporal
  information (the whole 30-cycle window, with position-specific weights); it just has no recurrent state or
  weight sharing across time. So argue carefully when recurrent modelling helps (longer dependencies,
  several conditions) and when a fixed window suffices.

## Q3 Justify the design using your evidence (10 marks, ~700 words)
- **(a)** For both networks: linear output + MSE. The target is *bounded* (RUL/125 lies in [0, 1]), but the
  linear output does not enforce this, so negative predictions are clipped at 0 at inference; a scaled sigmoid
  would enforce the bounds but has small gradients at the ends where many targets lie (31% at the cap), and it
  was not tested, so present the linear output as a simplicity choice, not a proven one. ReLU and He/Kaiming
  init apply to the MLP only (gradient flow); the LSTM uses sigmoid gates and tanh cell activations
  (gating eases vanishing gradients over 30 steps). Depth 2, width (tuning: wider MLP worse), Adam, weight decay,
  dropout. **Use evidence:** fig09 learning curves (no train/val divergence), ablation (dropout did not help:
  12.81 -> 12.67), stress test (networks more robust to noise than the forest, fig13), T2 input-scaling result.
- **(b)** Validity: engine-level split; scaler fitted on train only; tuning plan fixed in advance; validation-only
  selection; three seeds; test metrics computed once (freeze manifest; loader correction noted). Threats to discuss (at least two):
  1. Only 20 validation engines: models were indistinguishable on validation but differed on test.
  2. Single-seed tuning: candidate differences smaller than seed variability.
  3. Capped targets and the cap value assumed, not tuned; uncapped RMSE is higher.
  4. Unequal effective budgets: forest has no early stopping; different input representations.
- **(c)** Framework: autograd removes manual gradients (contrast with your T2 gradient check), but hides
  numerical issues; nondeterminism (thread scheduling; cuDNN on GPU, which is why runs used the CPU);
  deployment constraints: forest 373k nodes vs MLP 62k params; timing 55.7 vs 0.21 ms per engine.

## Q4 Developments (8 marks, ~600 words)
Need at least 4 peer-reviewed sources, including 2 published in the last 5 years (2021 onwards for a 2026
submission). At least one development must be hybrid/modular.

**Starting points: verify every detail (authors, year, venue, results) yourself before citing.
Do not cite anything you have not opened.**
- CNN for RUL on C-MAPSS: Babu, Zhao and Li (2016), DASFAA.
- LSTM for RUL: Zheng, Ristovski, Farahat and Gupta (2017), IEEE ICPHM.
- Deep 1D CNN for RUL: Li, Ding and Sun (2018), *Reliability Engineering & System Safety*.
- Mixture of experts: Shazeer et al. (2017), ICLR (sparsely gated MoE).
- Attention/Transformer: Vaswani et al. (2017), NeurIPS.
- **Recent (2021+), find yourself:** search Google Scholar for "CNN-LSTM remaining useful life C-MAPSS",
  "transformer remaining useful life turbofan", and "mixture of experts prognostics", filtered to 2021 onwards,
  and pick peer-reviewed journal or conference papers.

- **(b) Extension idea:** a CNN-LSTM hybrid, where a 1D CNN extracts features from each cycle and an LSTM
  aggregates them over time, or a mixture of experts routed by operating condition for FD002/FD004.
  State how representations pass between components, how it is trained (end-to-end, MSE), a falsifiable
  hypothesis (e.g. "lower mean test RMSE than the window MLP over 3 seeds on FD004"), a fair comparator
  (same split, tuning budget and seeds) and an outcome that would show it does not help.
- **(c)** Trade-off, e.g. interpretability or efficiency vs accuracy. Separate published results from
  your own proposed benefits.
