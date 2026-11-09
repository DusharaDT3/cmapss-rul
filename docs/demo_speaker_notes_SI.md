# CW1 demo speaker notes — Sinhala

Target: 6 minutes (allowed: 5–7 minutes). Open http://localhost:8050. These are rehearsal notes: understand each point and explain it in your own words. The displayed predictions and timings are produced live, so read the values on your screen.

## Before recording

- Start the app with `.venv/bin/python scripts/demo_app.py` from the project folder.
- Open Prediction lab and check that the normal example has loaded.
- Use the Present button if you want full screen.
- Use the Demo guide timer for rehearsal, then return to Prediction lab.
- Record your screen and microphone as required by the lecturer.

## 00:00–00:45 — Problem and data

**Show:** Prediction lab heading and dataset statistics.

“මගේ project එකෙන් කරන්නේ turbofan engine එකක sensor history එක භාවිතා කරලා, ඒ engine එකට තව වැඩ කරන්න ඉතිරි operating cycles ගණන අනුමාන කරන එකයි. මේක Remaining Useful Life, නැත්නම් RUL prediction කියලා හඳුන්වනවා.

මේ prediction එක maintenance planner කෙනෙකුට තීරණ ගැනීමට උපකාර විය හැකියි. මම භාවිතා කළේ NASA C-MAPSS FD001 simulated dataset එක. ඒකේ training engines 100ක් සහ official test engines 100ක් තියෙනවා. මෙය regression task එකක් නිසා performance මනින්නේ accuracy percentage එකකින් නොව, cycles වලින් ලැබෙන RMSE සහ MAE වැනි errors වලින්.”

## 00:45–02:00 — Live inference and invalid input

**Click:** Normal engine → Run prediction. Show the four result cards, known label and seed ranges. Scroll to the sensor chart if needed.

“දැන් මම engine 31 සඳහා saved models භාවිතයෙන් prediction එකක් run කරනවා. මේ input එකේ observed cycles 196ක් තියෙනවා. App එක saved preprocessing සහ එක් එක් model එකේ seeds තුනේ checkpoints load කරලා predictions වල mean එක පෙන්වනවා.

මෙතැන Random Forest, Window MLP, LSTM සහ පසුව සංවර්ධනය කළ History MLP එකේ results තියෙනවා. History MLP එකට ආසන්න වශයෙන් cycles 7.3ක් ලැබෙනවා; මේ දන්නා test example එකේ සැබෑ label එක cycles 8යි. අනෙක් models වල අගයන්ත් screen එකේ පෙන්වනවා. එක example එකක් හොඳ වුණා කියලා සමස්ත model එක හොඳයි කියන්න බැහැ.

Seed range එකෙන් පෙන්වන්නේ වෙනස් random seeds වලින් train කළ models අතර වෙනසයි. ඒක confidence interval එකක් නොවේ. Sensor chart එකේ shaded කොටස original models භාවිතා කරන අවසන් cycles 30යි. History MLP එක ඒ වන විට නිරීක්ෂණය කර ඇති සම්පූර්ණ history එක භාවිතා කරනවා.”

**Click:** Invalid input. Show the rejection message.

“මේ file එකේ cycles 20ක් පමණයි. Model එකට අවම වශයෙන් consecutive cycles 30ක් අවශ්‍ය නිසා app එක input එක reject කරනවා. Prediction එකක් නිකුත් කරන්නේ නැහැ.”

**Optional if time permits:** Click Outside scope.

“FD002 example එකේ sensor readings, FD001 සඳහා නියම කර ඇති range check එකෙන් පිටත නිසා එයත් reject කරනවා. මේක scope check එකක්; සියලු වෙනස් operating conditions හඳුනාගන්නා සම්පූර්ණ detector එකක් බව එයින් තහවුරු වෙන්නේ නැහැ.”

**Before leaving:** Click Normal engine again so a valid live result is available for the trace.

## 02:00–03:00 — Architecture and one design choice

**Click:** Model architecture → Window MLP, then LSTM. Show the layer diagram and parameter count.

“Window MLP එකට cycles 30ක sensors 14ක් ලැබෙනවා. ඒවා flatten කළාම inputs 420ක් වෙනවා. Hidden layers දෙකේ units 128ක් සහ 64ක් තියෙනවා; ReLU activations සහ training dropout භාවිතා කරනවා. අවසානයේ linear output එකෙන් RUL predict කරනවා. මේ model එකේ trainable parameters 62,209ක් තියෙනවා.

LSTM එක ඒ sensor readings sequence එකක් විදිහට කියවනවා. ඒකේ layers දෙකක් සහ එක් layer එකකට hidden units 64ක් තියෙනවා. Final time step එකේ hidden state එක linear head එකකට යවනවා. Parameters 53,825ක් තියෙනවා.”

**Scroll:** Split by whole engine panel.

“මගේ වැදගත් design choice එකක් තමයි rows වෙනුවට whole engines අනුව split කිරීම. Consecutive windows දෙකක් cycles 29ක් share කරන නිසා random row split එකකින් ඉතා සමාන data training සහ validation දෙකටම යන්න පුළුවන්. ඒ නිසා engines 80ක් training සඳහාත්, engines 20ක් validation සඳහාත් තබලා තියෙනවා. Preprocessing fit කරලා තියෙන්නේ training engines වලට පමණයි.”

## 03:00–04:15 — Results and trace from configuration to output

**Click:** Results & trace. Briefly show original comparison, then choose LSTM and Seed 2 under Trace a saved result.

“Original fixed comparison එකේ Random Forest එකේ mean test RMSE එක ආසන්න වශයෙන් cycles 12.13යි. Window MLP එකේ 13.40යි; LSTM එකේ 13.27යි. ඒ නිසා මේ original experiment එකේ අඩුම mean error එක ලැබුණේ Random Forest එකටයි.

දැන් මම එක result එකක් trace කරනවා. මේ LSTM seed 2 configuration එකේ layers, hidden units, dropout සහ learning rate වැනි settings තියෙනවා. ඒ configuration එකෙන් train කර saved කරපු checkpoint එක `models/lstm_seed2.pt` යන file එකයි. මෙතැන fingerprint එකෙන් එම checkpoint file එක හඳුනාගන්න පුළුවන්.

එයට අදාළ saved test RMSE එක cycles 12.80ක්. ඒ metric එක `results/results_table.csv` තුළ තියෙනවා. මෙහි live inference කොටසෙන් ඒ seed එක engine 31 සඳහා දෙන prediction එකත් seeds තුනේ mean එකත් වෙන වෙනම පෙන්වනවා. Test RMSE එක test engines සමූහයේ error එකයි; live prediction එක මේ එක් engine එකකට ඉතිරි cycles ගණනයි.”

**Show:** Configuration or fingerprint disclosure briefly; point to the reproduce command.

“මේ command එක භාවිතයෙන් එම inference එක terminal එකෙන් නැවත run කරන්නත් පුළුවන්. ඒ නිසා configuration එක, checkpoint එක, saved evaluation එක සහ inference output එක අතර සම්බන්ධය පරීක්ෂා කරන්න පුළුවන්.”

## 04:15–05:15 — One observed failure

**Click:** Error analysis → Random forest. Select an overestimate, for example engine 67 if it appears in the five listed cases. Read the actual displayed values. Click Replay engine.

“Average error එක හොඳ වුණත් තනි engines වල ලොකු වැරදි තියෙන්න පුළුවන්. මේ example එකේ true RUL එකට වඩා model එක වැඩි remaining life එකක් predict කරලා තියෙනවා. Positive error එකෙන් අදහස් වෙන්නේ overestimate එකක් බවයි.

උදාහරණයක් ලෙස engine 67 සඳහා Random Forest එක cycles 107.9ක් පමණ predict කරන නමුත් true RUL එක cycles 77යි. ඒක cycles 30.9ක පමණ overestimate එකක්. මේ prediction එක පමණක් විශ්වාස කළොත් maintenance ප්‍රමාද විය හැකියි.

මේ engine එකේ observed history එක කෙටි නිසා degradation signal එක පැහැදිලි නැති වීම possible explanation එකක්. නමුත් මේ examples වලින් ඒ හේතුව තහවුරු කරන්න බැහැ. Replay කිරීමෙන් ඒ engine එක saved models හරහා නැවත run කරලා output එක පෙන්වන්න පුළුවන්.”

**If a different engine is displayed:** Replace the engine number and values above with the actual values shown. Error = prediction − capped true RUL.

## 05:15–06:00 — Follow-up, limitations and next step

**Click:** Results & trace → Exploratory follow-up.

“පසුව observed history පිළිබඳ වැඩි information භාවිතා කරන History MLP එකක් අත්හදා බැලුවා. එහි mean test RMSE එක cycles 9.75ක්; history භාවිතා කරන නව Random Forest එකේ 10.93ක්. හැබැයි මේ follow-up එක කළේ මුල් test results දැනගත් පසුවයි, ඒ test set එකම නැවත භාවිතා කරලා. Feature representations සහ tuning budgets වලත් වෙනස්කම් තියෙනවා. ඒ නිසා මෙය තහවුරු කළ independent improvement එකක් කියලා කියන්න බැහැ.

මෙම dataset එක simulated data එකක් සහ එක operating condition එකකට සීමා වෙනවා. RUL targets cycles 125කින් cap කරලා තියෙනවා. ඒ නිසා මේ benchmark එකෙන් real engines සඳහා deployment readiness තහවුරු වෙන්නේ නැහැ.

මීළඟ පියවර තමයි History MLP configuration එක freeze කරලා, independent data සහ වැඩි operating conditions යටතේ matched baseline එකක් සමඟ evaluate කිරීම. මේ project එකෙන් models වල ශක්තීන් සහ සීමා දෙකම evidence සමඟ පෙන්වනවා. ස්තුතියි.”

## Key numbers to remember

- Original inputs: 30 cycles × 14 sensors; Window MLP: 420 → 128 → 64 → 1, ReLU, 62,209 parameters.
- LSTM: 2 layers × 64 hidden units; 53,825 parameters.
- History MLP: 113 → 128 → 64 → 1, SiLU, 22,913 parameters; exploratory follow-up.
- Split: 80 train engines / 20 validation engines, split seed 42; 100 official test engines.
- Original mean test RMSE: RF 12.13; MLP 13.40; LSTM 13.27 cycles.
- Follow-up mean test RMSE: History MLP 9.75; history RF 10.93 cycles.
- LSTM seed 2 test RMSE: 12.80 cycles.
- Normal test engine 31: known true RUL 8; history MLP mean about 7.3 cycles.
- Failure engine 67: true RUL 77; original forest ensemble about 107.9 cycles.

Do not memorise the text without understanding it. Rehearse aloud and shorten it if it exceeds seven minutes. Live timing numbers vary; do not read them as the controlled benchmark results.
