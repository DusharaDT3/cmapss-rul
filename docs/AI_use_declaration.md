# Declaration of generative-AI use (NB6057CEM CW1 and CW2)

**Student:** K A D Dushara Kalubowila  **Student ID:** KIC-BSCCSAI-251P-052
**Tools:** Claude (Anthropic), used through Claude Code, and OpenAI Codex, both on my own laptop  **Period:** October–November 2026

*Check this against your institution's required wording and form before signing. If you change the
drafts or code yourself after this date, add that to "My contributions" so the declaration stays accurate.*

## Extent of AI assistance

**Substantial.** AI tools produced most of the technical work and the first drafts of both reports.

- **Topic research:** the tool summarised the brief and proposed candidate datasets and topics.
- **CW1 code:** the tool wrote the project code, including the data pipeline and engine-level split,
  the NumPy MLP and its gradient check, the PyTorch MLP and LSTM, the random-forest baseline, the
  pre-registered tuning and evaluation protocol, the inference CLI, the smoke test and the packaging and
  document-build scripts. It also wrote the analysis notebooks and their interpretation text.
- **Experiments:** the tool ran every experiment on my laptop (CPU only) using that code. All reported
  numbers come from those runs and are stored in `results/`; none were typed in by hand or invented. The
  tool identified and corrected several of its own errors during the work, including an inverted plot
  axis, unscaled NumPy-MLP inputs, deleted tuning checkpoints (regenerated and shown to reproduce exactly)
  and a loader that read test files during training (separated, with identical training data).
- **Written work:** the tool drafted the CW1 technical record, the CW2 report, the Q1 diagrams and
  calculations, and the demonstration plan, and revised them after a review.
- **Demonstration materials:** the tool wrote the browser demo app (`scripts/demo_app.py`) and the slide
  deck with speaker notes used in the demonstration video. I presented and narrated the video myself.
- **Literature:** the tool suggested the cited sources and checked their bibliographic details online.
  Liu et al. (2025) was read in full; the claims cited from Ensarioğlu et al. (2023) and Mo et al. (2021)
  were checked against their abstracts only; the textbook references were not checked online.
- **Exploratory follow-up (OpenAI Codex):** Codex wrote and ran the two follow-up experiments reported
  in the technical record (the v1 neural upgrade and the v2 history MLP, in `experiments/`), with their
  scripts, unit tests, notebooks 04–05 and README sections. They were designed after the original test
  results were known and are reported as descriptive only. Claude later checked these results against
  the saved files and added the short follow-up section to the technical record.

## My contributions

- Chose the project topic (NASA C-MAPSS remaining-useful-life prediction) from the options proposed.
- Decided to run everything on the CPU rather than the GPU or Colab.
- Directed the scope of the work and requested the review corrections.
- Recorded the demonstration video (to be completed before submission).

## Responsibility

I understand that I remain responsible for the submitted code, calculations and claims, and that I must
be able to explain them.

Signed: ____________________  Date: __________
