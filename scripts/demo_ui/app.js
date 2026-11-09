'use strict';
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = (n,d=1) => Number(n).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const names = {predict:'Prediction lab',architecture:'Model architecture',evidence:'Results & trace',failures:'Error analysis',guide:'Demo guide'};
const colors = ['#3564d5','#d88b4d','#269286','#8f6abb','#c45c6b','#597ea8','#b99c46','#34756c','#91664e','#b75e9d','#7a8c4d','#549cba','#726cac','#a47770'];
let evidence, lastResult, chart, architecture='mlp', busy=false;
let visibleSensors = new Set(['s2','s7','s11']);
function navigate(){
  const view = location.hash.slice(1) in names ? location.hash.slice(1) : 'predict';
  document.querySelectorAll('.view').forEach(el => el.hidden = el.id !== `view-${view}`);
  document.querySelectorAll('[data-view]').forEach(el => {el.classList.toggle('active',el.dataset.view===view); if(el.dataset.view===view)el.setAttribute('aria-current','page');else el.removeAttribute('aria-current');});
  $('breadcrumb').textContent = names[view];
  window.scrollTo(0,0);
  if(view==='predict' && chart) drawChart();
}
window.addEventListener('hashchange',navigate); navigate();
document.querySelectorAll('[data-goto]').forEach(b=>b.addEventListener('click',()=>{location.hash=b.dataset.goto;window.scrollTo({top:0,behavior:'smooth'});}));
$('fullscreen').addEventListener('click',async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen();}catch{showError('Your browser did not allow full screen. You can use its full screen shortcut instead.');}});
function showError(message){$('app-error').textContent=message;$('app-error').hidden=false;}
async function jsonFetch(url,options){const r=await fetch(url,options);let body;try{body=await r.json();}catch{throw new Error('The server did not return valid data. Check that the demo is running.');}if(!r.ok)throw new Error(body.error||`Request failed (${r.status})`);return body;}
function selectedExample(name){document.querySelectorAll('[data-example]').forEach(b=>b.classList.toggle('selected',b.dataset.example===name));}
document.querySelectorAll('[data-example]').forEach(b=>b.addEventListener('click',()=>{$('example').value=b.dataset.example;selectedExample(b.dataset.example);predict({example:b.dataset.example});}));
$('example').addEventListener('change',()=>selectedExample($('example').value));
$('predict').addEventListener('click',()=>predict({example:$('example').value}));
$('upload').addEventListener('change',async e=>{const file=e.target.files[0];if(!file)return;if(file.size>5*1024*1024){showError('Choose a CSV smaller than 5 MB.');e.target.value='';return;}selectedExample('');try{await predict({text:await file.text(),name:file.name});}catch(err){showError(err.message);}e.target.value='';});
async function predict(body){
  if(busy)return;busy=true;$('app-error').hidden=true;
  document.querySelectorAll('#predict, [data-example], #example, #upload').forEach(el=>el.disabled=true);
  $('predict').textContent='Running saved models…';
  $('prediction-result').setAttribute('aria-busy','true');
  $('prediction-result').innerHTML='<div class="empty-state">Validating input and running the saved checkpoints…</div>';
  $('sensor-panel').hidden=true;
  lastResult=null;
  if(evidence)renderTrace();
  try{
    const r=await jsonFetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    lastResult=r;renderPrediction(r);if(evidence)renderTrace();
  }catch(err){$('prediction-result').innerHTML=`<div class="notice danger"><strong>Prediction could not run</strong><p>${esc(err.message)}</p></div>`;}
  finally{busy=false;document.querySelectorAll('#predict, [data-example], #example, #upload').forEach(el=>el.disabled=false);$('predict').innerHTML='Run prediction <span>↗</span>';$('prediction-result').removeAttribute('aria-busy');}
}
function renderPrediction(r){
  chart=r.chart||null;$('sensor-panel').hidden=!chart;
  if(!r.ok){
    $('prediction-result').innerHTML=`<div class="notice danger"><strong>Input rejected · ${esc(r.kind)}</strong><p>${esc(r.message)}</p><p>Source: ${esc(r.source)} · CLI exit code ${esc(r.code)}. No prediction returned.</p></div>`;
  }else{
    const truth=r.truth!=null?`<span class="truth-pill">Known test label: ${fmt(r.truth,0)} cycles</span>`:'';
    $('prediction-result').innerHTML=`<div class="result-heading"><div><h2>Engine ${esc(r.unit)} <span class="muted">/ prediction results</span></h2><p>${esc(r.n_cycles)} observed cycles · ${esc(r.source)}</p></div>${truth}</div><div class="prediction-grid">${r.predictions.map(p=>`<article class="prediction-card ${p.id==='history'?'history-card':''}"><div class="model-title"><h3>${esc(p.model)}</h3><span class="model-indicator"></span></div><div class="phase-label">${p.phase==='exploratory'?'Exploratory follow-up':'Original comparison'}</div>${p.unavailable?`<p class="unavailable">${esc(p.unavailable)}</p>`:`<div class="prediction-value">${fmt(p.rul)}<small>cycles</small></div><div class="seed-bar"><i style="width:${Math.max(2,Math.min(100,p.rul/125*100))}%"></i></div><div class="seed-range">Seed range ${fmt(p.lo)}–${fmt(p.hi)} cycles</div>`}<div class="meta-line"><span>${p.ms!=null?fmt(p.ms)+' ms':'Unavailable'}</span><button data-trace="${esc(p.id)}">Trace result ↗</button></div></article>`).join('')}</div><div class="prediction-foot">Predictions average three seed models. Seed ranges are not confidence intervals. Live timings cover prediction calls on this machine; they are not the controlled latency benchmark.${r.predictions.some(p=>p.rul>=115)?'<br>Targets were capped at 125 cycles. High estimates do not give an exact failure time.':''}</div><div class="result-heading"><span class="caption">History MLP uses the full observed prefix and was developed after earlier test results were known.</span><button class="button secondary" id="export-result">↓ Export result</button></div>`;
    document.querySelectorAll('[data-trace]').forEach(b=>b.addEventListener('click',()=>{$('trace-model').value=b.dataset.trace;populateSeeds();renderTrace();location.hash='evidence';setTimeout(()=>$('trace-content').scrollIntoView({behavior:'smooth',block:'center'}),50);}));
    $('export-result').addEventListener('click',()=>{const blob=new Blob([JSON.stringify(r,null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=`engine_${r.unit}_prediction.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
  }
  if(chart){$('sensor-legend').innerHTML=Object.keys(chart.sensors).map((s,i)=>`<button data-sensor="${esc(s)}" aria-pressed="${visibleSensors.has(s)}" class="${visibleSensors.has(s)?'active':''}"><i style="background:${colors[i%colors.length]}"></i>${esc(s)}</button>`).join('');document.querySelectorAll('[data-sensor]').forEach(b=>b.addEventListener('click',()=>{if(visibleSensors.has(b.dataset.sensor))visibleSensors.delete(b.dataset.sensor);else visibleSensors.add(b.dataset.sensor);b.classList.toggle('active',visibleSensors.has(b.dataset.sensor));b.setAttribute('aria-pressed',String(visibleSensors.has(b.dataset.sensor)));drawChart();}));drawChart();}
}
function drawChart(){
  if(!chart)return;
  const W=1000,H=255,L=50,R=20,T=24,B=34;
  const entries=Object.entries(chart.sensors), values=entries.filter(([s])=>visibleSensors.has(s)).flatMap(([,v])=>v);
  const min=Math.max(-1,Math.min(-.35,...values)),max=Math.min(2,Math.max(1.35,...values));
  const clamp=v=>Math.min(max,Math.max(min,v));
  const x=i=>L+i/(chart.cycles.length-1)*(W-L-R),y=v=>T+(max-v)/(max-min)*(H-T-B);
  let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Scaled sensor readings across observed engine cycles"><text x="${L}" y="12" fill="#8393a8" font-size="10">Scaled sensor value</text><rect x="${x(Math.max(0,chart.cycles.length-chart.window))}" y="${T}" width="${W-R-x(Math.max(0,chart.cycles.length-chart.window))}" height="${H-T-B}" fill="#edf2fd"/>`;
  for(let i=0;i<=4;i++){const val=min+i*(max-min)/4, yy=y(val);svg+=`<line x1="${L}" x2="${W-R}" y1="${yy}" y2="${yy}" stroke="#e8edf5"/><text x="${L-9}" y="${yy+4}" text-anchor="end" fill="#8594a9" font-size="10">${val.toFixed(2)}</text>`;}
  for(const v of chart.limits)svg+=`<line x1="${L}" x2="${W-R}" y1="${y(v)}" y2="${y(v)}" stroke="#cfb49f" stroke-dasharray="5 5"/>`;
  entries.forEach(([s,vs],j)=>{if(!visibleSensors.has(s))return;const points=vs.map((v,i)=>`${x(i).toFixed(1)},${y(clamp(v)).toFixed(1)}`).join(' ');svg+=`<polyline points="${points}" fill="none" stroke="${colors[j%colors.length]}" stroke-width="1.7" stroke-linejoin="round" opacity=".85"><title>${esc(s)}</title></polyline>`;});
  for(let i=0;i<=4;i++){const k=Math.round(i*(chart.cycles.length-1)/4);svg+=`<text x="${x(k)}" y="${H-10}" text-anchor="middle" fill="#8594a9" font-size="10">${chart.cycles[k]}</text>`;}
  svg+=`<text x="${W-R}" y="12" text-anchor="end" fill="#8393a8" font-size="10">Observed cycle →</text></svg>`;
  if(values.some(v=>v<min||v>max))svg+='<p class="caption">Extreme values are clipped to the chart boundary; the original values are still used by the input checks.</p>';
  $('sensor-chart').innerHTML=svg;
}
function renderArchitecture(){
  const m=evidence.models[architecture];
  document.querySelectorAll('[data-arch]').forEach(b=>{b.classList.toggle('active',b.dataset.arch===architecture);b.setAttribute('aria-pressed',String(b.dataset.arch===architecture));});
  $('architecture-content').innerHTML=`<div class="panel"><div class="panel-heading"><div><span class="eyebrow">${m.phase==='exploratory'?'EXPLORATORY FOLLOW-UP':'ORIGINAL COMPARISON'}</span><h2>${esc(m.name)}</h2></div><span class="chip ${m.phase==='exploratory'?'amber':'blue'}">${m.parameters!=null?fmt(m.parameters,0)+' trainable parameters':'Non-neural baseline'}</span></div><p class="architecture-description">${esc(m.description)}</p><div class="architecture-flow" aria-label="Model layer flow">${m.blocks.map(([value,name])=>`<div class="layer-node"><div class="layer-dots"><i></i><i></i><i></i><i></i><i></i></div><div class="node-value">${esc(value)}</div><div class="node-name">${esc(name)}</div></div>`).join('')}</div><p class="caption">Layer-flow schematic. Nodes illustrate stages, not individual learned weights. ${esc(m.postprocessing)}</p><div class="arch-stats"><div><strong>${m.parameters!=null?fmt(m.parameters,0):fmt(m.tree_nodes,0)}</strong><span>${m.parameters!=null?'parameters per seed model':'tree nodes in seed 0 forest'}</span></div><div><strong>${m.config.dropout!=null?m.config.dropout:'300 trees'}</strong><span>${m.config.dropout!=null?'dropout during training':'per forest'}</span></div><div><strong>${architecture==='history'?'AdamW':architecture==='random_forest'?'Tree regression':'Adam'}</strong><span>training method</span></div><div><strong>${architecture==='random_forest'?'Squared error':'MSE on RUL / 125'}</strong><span>training objective</span></div></div></div><div class="panel"><div class="panel-heading"><div><span class="eyebrow">LAYER DETAILS</span><h2>Shapes & parameter counts</h2></div><span class="chip">Batch size = 1</span></div><div class="table-wrap"><table><thead><tr><th>Layer / stage</th><th>Output shape</th><th>Trainable parameters</th></tr></thead><tbody>${m.layers.map(([name,shape,params])=>`<tr><td>${esc(name)}</td><td><code>${esc(shape.join(' × '))}</code></td><td>${params===null?'Not applicable':fmt(params,0)}</td></tr>`).join('')}</tbody></table></div><details><summary>Inspect the saved configuration</summary><pre>${esc(JSON.stringify(m.config,null,2))}</pre></details></div>`;
}
function resultsTable(phase){return `<div class="table-wrap"><table><thead><tr><th>Model</th><th>Validation RMSE</th><th>Test RMSE ± SD</th><th>Test MAE</th><th>Seeds</th></tr></thead><tbody>${evidence.results.filter(r=>r.phase===phase).map(r=>`<tr><td><strong>${esc(r.model)}</strong></td><td>${fmt(r.val_rmse,2)}</td><td class="rmse-cell"><span class="metric-number">${fmt(r.test_rmse,2)}</span><span class="metric-std"> ± ${fmt(r.std,2)}</span><div class="mini-bar"><span style="width:${r.test_rmse/16*100}%"></span></div></td><td>${fmt(r.test_mae,2)}</td><td>${r.seeds.join(', ')}</td></tr>`).join('')}</tbody></table></div>`;}
function populateSeeds(){const records=evidence.traces[$('trace-model').value].records;$('trace-seed').innerHTML=records.map(r=>`<option value="${r.seed}">Seed ${r.seed}</option>`).join('');$('trace-seed').value=String(records.at(-1).seed);}
function renderTrace(){
  const key=$('trace-model').value,t=evidence.traces[key],r=t.records.find(r=>String(r.seed)===$('trace-seed').value);if(!r)return;
  const live=lastResult?.ok?lastResult.predictions.find(p=>p.id===key):null, single=live?.seed_values?.find(s=>s.seed===r.seed);
  $('trace-content').innerHTML=`<div class="trace-flow"><div class="trace-item"><span class="eyebrow">01 / CONFIGURATION</span><strong>${esc(evidence.models[key].name)}</strong><code>${esc(t.config_path)}</code></div><div class="trace-item"><span class="eyebrow">02 / CHECKPOINT</span><strong>Seed ${r.seed}</strong><code>${esc(r.checkpoint.path)}</code></div><div class="trace-item"><span class="eyebrow">03 / SAVED TEST METRIC</span><div class="trace-metric">${fmt(r.metrics.test_rmse,2)}</div><strong>RMSE · cycles</strong><code>${esc(t.metrics_path)}</code></div><div class="trace-item"><span class="eyebrow">04 / LIVE INFERENCE</span>${single?`<div class="trace-metric">${fmt(single.rul)}</div><strong>cycles · this seed</strong><code>Engine ${esc(lastResult.unit)} · ensemble ${fmt(live.rul)} cycles</code>`:'<strong>Run an engine first</strong><code>Open Prediction lab and run a valid example.</code>'}</div></div><p class="caption">The test RMSE summarises held-out engine errors for one seed. The live output is one engine’s prediction; the demo displays the three-seed mean. ${single?'Current source: '+esc(lastResult.source)+'.':''}</p><div class="trace-details"><details><summary>Configuration used by the loaded model</summary><pre>${esc(JSON.stringify(t.config,null,2))}</pre></details><details><summary>Checkpoint fingerprint & saved metrics</summary><pre>${esc(JSON.stringify({checkpoint:r.checkpoint,preprocessing:t.preprocessing,metrics:r.metrics},null,2))}</pre></details></div><div class="terminal-command"><span class="muted">Reproduce the normal example in a terminal</span><br><code>$ ${esc(t.command)}</code></div><p class="caption">SHA-256 fingerprints identify the files loaded for this session. They are not a claim that the freeze manifest has been verified here. All evidence is read from the saved project artifacts.</p>`;
}
function renderFailures(){
  const key=$('failure-model').value,rows=evidence.failures.filter(r=>r.model===key);
  const max=Math.max(125,...rows.map(r=>Math.max(r.true_rul,r.predicted)));
  $('failure-content').innerHTML=rows.map(r=>`<article class="failure-card"><div><strong>Engine ${r.unit}</strong><small>${r.observed} observed cycles</small></div><div class="paired-bars"><div class="paired-row"><span>True RUL</span><div class="bar-track"><i style="width:${r.true_rul/max*100}%"></i></div><b>${fmt(r.true_rul,0)}</b></div><div class="paired-row predicted"><span>Predicted</span><div class="bar-track"><i style="width:${r.predicted/max*100}%"></i></div><b>${fmt(r.predicted)}</b></div></div><div class="failure-actions"><span class="error-value ${r.error<0?'negative':''}">${r.error>0?'+':''}${fmt(r.error)}</span><small> cycles</small><br><button class="button secondary" data-replay="${r.unit}">Replay engine ↗</button></div></article>`).join('')+`<p class="caption">Prediction − capped true RUL. Three-seed ensemble values from ${key==='history'?'experiments/neural_history_v2/test_predictions.csv':'results/test_predictions.csv'}. Examples are selected by largest absolute error for inspection.</p>`;
  document.querySelectorAll('[data-replay]').forEach(b=>b.addEventListener('click',()=>{if(busy)return;location.hash='predict';selectedExample('');window.scrollTo({top:0,behavior:'smooth'});predict({case_unit:Number(b.dataset.replay)});}));
}
$('trace-model').addEventListener('change',()=>{populateSeeds();renderTrace();});$('trace-seed').addEventListener('change',renderTrace);
$('failure-model').addEventListener('change',renderFailures);
$('learning-select').addEventListener('change',()=>{$('learning-image').src=`/figures/${$('learning-select').value}.png`;$('learning-image').alt=$('learning-select').value==='history'?'Saved history MLP learning curves':'Saved original MLP and LSTM learning curves';});
let elapsed=0,started=null;
function updateTimer(){const t=Math.floor((elapsed+(started?Date.now()-started:0))/1000);$('timer').textContent=`${String(Math.floor(t/60)).padStart(2,'0')}:${String(t%60).padStart(2,'0')}`;$('timer').classList.toggle('good',t>=300&&t<=420);$('timer').classList.toggle('over',t>420);$('timer-note').textContent=t>420?'Over 7 minutes — shorten your explanation.':t>=300?'Within the required 5–7 minute range.':'Target: 5–7 minutes · timer continues across sections';}
$('timer-toggle').addEventListener('click',()=>{if(started){elapsed+=Date.now()-started;started=null;$('timer-toggle').textContent='Resume timer';}else{started=Date.now();$('timer-toggle').textContent='Pause timer';}updateTimer();});
$('timer-reset').addEventListener('click',()=>{elapsed=0;started=null;$('timer-toggle').textContent='Start timer';updateTimer();});setInterval(updateTimer,500);
async function init(){
  try{
    const [ev,examples]=await Promise.all([jsonFetch('/api/evidence'),jsonFetch('/api/examples')]);evidence=ev;
    $('example').innerHTML=examples.map(name=>`<option value="${esc(name)}">${esc(name)}</option>`).join('');
    const requested=new URLSearchParams(location.search).get('run');$('example').value=examples.includes(requested)?requested:'engine_normal.csv';selectedExample($('example').value);
    const order=['mlp','lstm','history','random_forest'].filter(k=>ev.models[k]);
    $('architecture-tabs').innerHTML=order.map(k=>`<button data-arch="${k}" aria-pressed="${k===architecture}">${esc(ev.models[k].name)}</button>`).join('');
    document.querySelectorAll('[data-arch]').forEach(b=>b.addEventListener('click',()=>{architecture=b.dataset.arch;renderArchitecture();}));renderArchitecture();
    $('original-results').innerHTML=resultsTable('original');$('followup-results').innerHTML=resultsTable('exploratory');
    const options=order.map(k=>`<option value="${k}">${esc(ev.models[k].name)}</option>`).join('');$('trace-model').innerHTML=options;$('trace-model').value='lstm';populateSeeds();renderTrace();
    $('failure-model').innerHTML=options;$('failure-model').value='random_forest';renderFailures();
    $('connection').textContent=`${Object.keys(ev.models).length} models ready · CPU`;
    await predict({example:$('example').value});
  }catch(err){$('connection').textContent='Connection unavailable';showError(err.message);$('prediction-result').innerHTML='<div class="empty-state">Start the local server, then reload this page.</div>';}
}
init();
