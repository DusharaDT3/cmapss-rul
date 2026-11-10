# CW1 demo speaker notes — English

Target: approximately 6 minutes (required: 5–7 minutes). Open http://localhost:8050. Rehearse aloud and explain the work in your own words. Predictions and timings are calculated live, so check the values on your screen.

## Before recording

- From the project folder, start the app: `.venv/bin/python scripts/demo_app.py`.
- Open Prediction lab and confirm that the normal example has loaded.
- Use Present for full screen if helpful.
- Rehearse with the timer in Demo guide, then return to Prediction lab.
- Record your screen and microphone as required by your lecturer.

## 00:00–00:45 — Problem and data

**Show:** Prediction lab heading and dataset statistics.

“My project predicts the remaining useful life, or RUL, of a turbofan engine from its sensor history. The output estimates how many operating cycles the engine has left.

The intended user is a maintenance planner. Overestimating remaining life could delay maintenance, while underestimating it could lead to unnecessary replacement.

I used NASA C-MAPSS FD001, a simulated dataset with 100 training engines and 100 official test engines. This is a regression problem, so I report error in cycles using RMSE and MAE, rather than classification accuracy.”

## 00:45–02:00 — Live inference and invalid input

**Click:** Normal engine → Run prediction. Show the four prediction cards, known label and seed ranges. Briefly show the sensor chart.

“This example is test engine 31, with 196 observed cycles. The app loads saved preprocessing and three seed checkpoints for each model, then displays their mean prediction.

Here are the original Random Forest, Window MLP and LSTM, alongside the later History MLP. The History MLP predicts approximately 7.3 cycles remaining. The known test label is eight cycles. The other predictions are shown here. One successful example illustrates the output; it does not establish overall performance.

The seed range shows variation between independently trained models. It is not a confidence interval. The shaded chart region marks the last 30 cycles used by the original models. The History MLP uses the complete history observed so far.”

**Click:** Invalid input. Show the rejection message.

“This file has only 20 cycles. The models need at least 30 consecutive cycles, so the app rejects it and returns no prediction.”

**Optional if time permits — Click:** Outside scope.

“This FD002 example exceeds the FD001 sensor-range check and is also rejected. This check does not guarantee detection of every unfamiliar operating condition.”

**Before leaving:** Click Normal engine again, so a valid live result is available for the trace.

## 02:00–03:00 — Architecture and one design choice

**Click:** Model architecture → Window MLP, then LSTM. Show the diagrams and parameter counts.

“The Window MLP receives 30 cycles of 14 sensors. Flattening this produces 420 inputs. Its hidden layers contain 128 and 64 units, with ReLU activations and dropout during training. A linear output predicts scaled RUL. It has 62,209 trainable parameters.

The LSTM reads the same window in time order. It has two recurrent layers with 64 hidden units each. The final time-step representation feeds a linear head. It has 53,825 parameters.”

**Scroll:** Split by whole engine panel.

“One important design choice was splitting whole engines instead of individual rows. Consecutive windows overlap by 29 of 30 cycles, so a row split could leak near-identical examples into validation. I used 80 training engines and 20 validation engines. Preprocessing was fitted on training engines only.”

## 03:00–04:15 — Results and configuration-to-output trace

**Click:** Results & trace. Briefly show the original comparison. Under Trace a saved result, select LSTM and Seed 2.

“In the original fixed comparison, the Random Forest achieved the lowest mean test RMSE: approximately 12.13 cycles, compared with 13.40 for the Window MLP and 13.27 for the LSTM.

I will trace one result: LSTM seed two. This configuration specifies the hidden size, layers, dropout and learning rate. Its trained weights are saved in `models/lstm_seed2.pt`. The fingerprint identifies that checkpoint file.

The saved test RMSE is approximately 12.80 cycles, recorded in `results/results_table.csv`. The live inference panel shows this seed’s prediction for engine 31 and the three-seed mean. These are different quantities: test RMSE measures errors across test engines, while the live output estimates remaining life for one engine.”

**Show:** Expand the configuration or fingerprint disclosure, then point to the terminal command.

“The command below reproduces inference using the saved model. This connects the configuration, checkpoint, evaluation record and prediction output without retraining during the demonstration.”

## 04:15–05:15 — An observed failure

**Click:** Error analysis → Random forest. Choose an overestimate, such as engine 67 if listed. Read the displayed values, then click Replay engine.

“A good average score can hide large individual errors. Positive error means the model predicted more remaining life than the engine actually had.

For engine 67, the original forest ensemble predicts approximately 107.9 cycles, while the true RUL is 77. That is an overestimate of about 30.9 cycles. Relying on that estimate alone could delay maintenance.

A short observed history may contain little visible degradation, which is a possible explanation. These examples do not prove the cause. Replay engine runs the saved models on this engine again, so we can inspect their live predictions.”

**If you show a different engine:** Replace the engine number and values with those on screen. Error is prediction minus capped true RUL.

## 05:15–06:00 — Follow-up, limitations and next step

**Click:** Results & trace → Exploratory follow-up.

“The later History MLP uses more observed history and achieved a mean test RMSE of 9.75 cycles, compared with 10.93 for a new history-based forest.

However, this follow-up was developed after the original test results were known and reused that test set. Feature representations and tuning budgets also differ. It is therefore an exploratory result that needs independent confirmation.

FD001 is simulated, covers one operating condition and one fault mode, and uses targets capped at 125 cycles. This benchmark does not establish readiness for real maintenance decisions.

The next step is to freeze the history model and evaluate it on independent data and broader conditions against a matched baseline. Thank you.”

## Key numbers for rehearsal

- Original input: 30 cycles × 14 sensors.
- Window MLP: 420 → 128 → 64 → 1; ReLU; 62,209 parameters.
- LSTM: two layers × 64 hidden units; 53,825 parameters.
- History MLP: 113 → 128 → 64 → 1; SiLU; 22,913 parameters; exploratory follow-up.
- Split: 80 training / 20 validation engines, split seed 42; 100 official test engines.
- Original mean test RMSE: RF 12.13; MLP 13.40; LSTM 13.27 cycles.
- Follow-up mean test RMSE: History MLP 9.75; history RF 10.93 cycles.
- LSTM seed 2 test RMSE: 12.80 cycles.
- Test engine 31: true RUL 8; History MLP ensemble prediction about 7.3 cycles.
- Test engine 67: true RUL 77; original forest ensemble prediction about 107.9 cycles.

Rehearse aloud and adjust the length to stay within 5–7 minutes. Live timing values vary; they are not the controlled latency benchmark results. If your lecturer requires a live demonstration, use these notes for that presentation rather than assuming a recording is accepted.
