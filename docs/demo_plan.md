# T6b demonstration plan (5–7 minutes, recorded or live, by you)

The brief requires: inference, tracing one result from configuration to output, one design choice, one failure.

**Recording setup (current plan):** present the slide deck and switch to two live tools: the browser demo
app (`python scripts/demo_app.py`, http://localhost:8050) for inference, and a terminal for the trace.

| Time | Slides / tool | Content |
|---|---|---|
| 0:00–1:30 | Slides 1–4 | Problem, user, FD001 data, the four models and the fixed protocol |
| 1:30–2:45 | **Demo app** | engine_normal (11/12/11 cycles; history MLP 7; true 8), engine_too_short (invalid), FD002 (out of scope) |
| 2:45–3:35 | Slide 6 | **Design choice:** engine-level split to prevent leakage |
| 3:35–4:45 | Slide 7 + **terminal** | **Trace:** `cat configs/final_lstm.json`; `ls -l models/lstm_seed2.pt`; `grep "^lstm,configs/final_lstm.json,2," results/results_table.csv` (12.80); `python scripts/evaluate.py --verify-only` |
| 4:45–5:45 | Slides 8–9 | Results (forest best in the original protocol) and the exploratory history MLP (9.75, reused test set) |
| 5:45–6:25 | Slide 10 | **Observed failure:** engines 67 and 50 about 30 cycles late |
| 6:25–6:55 | Slides 11–12 | Recommendation (incl. noise result), limitations, thanks |

The older table below is the original screen-only plan, kept for reference.

| Time | Show | Say (in your own words) |
|---|---|---|
| 0:00–0:45 | README | Problem, user, dataset, FD001 scope |
| 0:45–2:00 | `python scripts/predict.py examples/engine_normal.csv`, then the too-short and FD002 files | Normal prediction; input validation; why FD002 is out of scope |
| 2:00–3:30 | **Trace one result:** `configs/final_lstm.json` -> `scripts/train_torch_model.py configs/final_lstm.json --seed 2` -> `models/lstm_seed2.pt` -> `results/metrics/lstm_seed2.json` -> row `lstm, seed 2` in `results/results_table.csv` (test RMSE 12.80) | Each reported number has a config, seed and checkpoint; hashes in `results/freeze_manifest.json` |
| 3:30–4:45 | Notebook 01 leakage table + `data/splits/fd001_split.json` | **Design choice:** engine-level split, and why a row-level split would leak |
| 4:45–6:00 | fig12 (worst engines 67 and 50) or fig13 (noise) | **Observed failure:** late predictions for short histories, or the forest's collapse under noise |
| 6:00–6:45 | Notebook 03 recommendation | Why the simpler model wins on clean data, and the limitations |

Before recording: run `python scripts/smoke_test.py`, and note that test engine 31's true RUL is 8 cycles
(`sed -n '31p' data/raw/RUL_FD001.txt`); the forest predicts 11, i.e. 3 cycles late, which is worth saying.

## Draft narration (about 6 minutes, put it in your own words before recording)

1. **Problem (45 s).** "This project predicts how many flight cycles a turbofan engine has left, to help a
   maintenance planner act before failure. I use NASA's C-MAPSS FD001: 100 simulated engines run to
   failure for training, and 100 engines cut off early for testing."
2. **Inference (75 s).** Run `python scripts/predict.py examples/engine_normal.csv`. "Test engine 31 is
   predicted to have 11 cycles left; its true value is 8, so the model is three cycles late, which is
   the direction that matters for safety." Run the too-short file: "Fewer than 30 cycles is rejected
   with exit code 2." Run the FD002 file: "Different operating conditions push the sensors outside the
   training range, so the tool refuses it rather than guessing."
3. **Trace one result (90 s).** Open `configs/final_lstm.json`, then show the command
   `python scripts/train_torch_model.py configs/final_lstm.json --seed 2`, the checkpoint
   `models/lstm_seed2.pt`, `results/metrics/lstm_seed2.json`, and the `lstm, 2` row in
   `results/results_table.csv` with test RMSE 12.80. "The freeze manifest stores a hash of this
   checkpoint, written before the one test evaluation, and the evaluation script refuses to run if
   any frozen file changes." (Optional: run `python scripts/evaluate.py --verify-only`.)
4. **Design choice (75 s).** Show the leakage table in notebook 01. "Windows from one engine overlap, so
   splitting rows would put near-copies of training windows into validation. I split by engine instead,
   and fit the scaler on training engines only."
5. **Observed failure (75 s).** Show Fig. 12. "Engines 67 and 50 had only about 70 observed cycles. The
   model saw no degradation yet and predicted about 108 cycles left when the truth was about 78: a late
   error of about 30 cycles, the dangerous kind." Optionally show Fig. 13: "The random forest is best on
   clean data but degrades fastest under sensor noise."
6. **Conclusion (40 s).** "On clean FD001 data the random forest is most accurate, at 12.1 cycles RMSE.
   The neural networks are more robust to noise; the LSTM was slightly better than the MLP, but well within
   seed variation, so no advantage is established. A later exploratory network that uses each engine's whole
   history reached 9.75, but it reused the test set, so it is not a fresh result. This is one simulated
   benchmark, so it does not show the model is ready for deployment." (Optional: run
   `python scripts/predict_neural_history.py examples/engine_normal.csv`, which predicts about 7 cycles.)
