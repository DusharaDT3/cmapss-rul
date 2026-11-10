# CW1 demo — short script (about 6 minutes)

About 620 words. At a calm pace (about 100 words per minute) this is about 6:15. Record each part as a separate file, in order. Say it in your own words if you prefer, but keep each part close to its target time.

---

## Part 1 — Introduction (target 0:45, about 75 words)

**Screen:** Prediction lab, top of the page.

"Hello, I am Dushara Kalubowila, index number 052. My project predicts the remaining useful life of a turbofan engine: how many operating cycles it has left, from its sensor history.

The user is a maintenance planner. If we overestimate remaining life, maintenance could be delayed; if we underestimate it, parts are replaced too early.

I used NASA C-MAPSS FD001: simulated data with 100 training engines and 100 test engines. This is regression, so I report error in cycles using RMSE and MAE."

## Part 2 — Live prediction and invalid input (target 1:30, about 150 words)

**Screen:** Click Normal engine → Run prediction. Point at the four cards, then the chart. Then click Invalid input.

"Here I run test engine 31, which has 196 recorded cycles. The app loads the saved preprocessing and three trained seeds for each model, and shows their average.

There are four models: random forest, window MLP and LSTM from my original comparison, and a later history MLP. The true remaining life is 8 cycles. The original models say about 11, which is a few cycles late. The history MLP says 7.3, the closest. But one good prediction does not prove a model is good overall.

The seed range shows how much the three trained versions disagree; it is not a confidence interval.

The chart shows the sensor history. The shaded part is the last 30 cycles that the original models use; the history MLP uses the whole history.

Now an invalid file. It has only 20 cycles, but the models need 30, so the app rejects it instead of guessing."

## Part 3 — Architecture and one design choice (target 1:10, about 115 words)

**Screen:** Model architecture → Window MLP, then LSTM. Then scroll to Split by whole engine.

"The window MLP takes 30 cycles of 14 sensors and flattens them into 420 inputs. It has two hidden layers of 128 and 64 units with ReLU and dropout, and a linear output. That is 62,209 parameters.

The LSTM reads the same 30 cycles in time order, with two layers of 64 units and a linear head: 53,825 parameters.

My key design choice was to split the data by whole engine, not by row. Neighbouring windows share 29 of their 30 cycles, so a row split would put near-copies into both training and validation. I used 80 engines for training and 20 for validation, and fitted the scaling on training engines only."

## Part 4 — Results and tracing one result (target 1:15, about 125 words)

**Screen:** Results & trace. Point at the table, then choose LSTM and Seed 2 in the trace panel.

"In the original comparison, the random forest had the lowest test RMSE, 12.13 cycles, against 13.40 for the window MLP and 13.27 for the LSTM.

Now I trace one result: LSTM, seed 2. This is its configuration: hidden size, layers, dropout and learning rate. Its trained weights are in models/lstm_seed2.pt, and this fingerprint identifies that exact file. Its saved test RMSE is 12.80, recorded in the results table. And this is the same model's live prediction for engine 31.

Note the difference: test RMSE is the error over all test engines, while the live output is one engine's prediction. This command reproduces the inference from the saved model, without retraining."

## Part 5 — Failure, follow-up and conclusion (target 1:35, about 155 words)

**Screen:** Error analysis → Random forest → engine 67, click Replay engine. Then Results & trace → Exploratory follow-up.

"Good averages can hide large errors. For engine 67, the true remaining life is 77 cycles, but the random forest predicts about 108: an overestimate of 31 cycles. In practice that could delay maintenance. One possible reason is that this engine has only 71 cycles of history, with little visible wear yet, but I cannot prove that from this example alone.

Later I tried a history MLP that uses the whole engine history. It reached 9.75 cycles RMSE, against 10.93 for a history-based random forest. But I built it after seeing the original test results and reused the same test set, so it is exploratory and needs confirming on new data.

Limitations: FD001 is simulated, with one operating condition and one fault mode, and the targets are capped at 125 cycles. So this is not ready for real maintenance decisions. Next, I would test the history MLP on independent data against a matched baseline. Thank you."
