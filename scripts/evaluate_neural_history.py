"""Once-only descriptive evaluation of v2; selection must be complete first."""
import json
import sys
import time
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src import data
from src.neural_history import HistoryRun,frame_features
from src.metrics import all_metrics
from src.train_torch import environment
from scripts.train_neural_history import OUT,digest


def main():
    if (OUT/'evaluation_manifest.json').exists():
        raise SystemExit('Evaluation already started/completed. Read results; no further tuning in this experiment.')
    selected=data.load_config(OUT/'selection.json');train=data.load_config(OUT/'training_manifest.json')
    for rel,h in train['files'].items():
        if digest(data.ROOT/rel)!=h:raise SystemExit(f'Changed training source/input: {rel}')
    seeds=selected['seeds']
    cfg=data.load_config(data.ROOT/train['plan']['data_config'])
    paths=[Path(__file__),OUT/'selection.json',data.ROOT/'src/metrics.py',
           data.ROOT/'scripts/predict_neural_history.py',data.ROOT/'scripts/predict.py',
           data.RAW_DIR/f"test_{cfg['subset']}.txt",data.RAW_DIR/f"RUL_{cfg['subset']}.txt"]
    paths += [OUT/f'selected_{name}_seed{s}.pt' for name in selected['models'] for s in seeds]
    files={**train['files'],**{str(p.relative_to(data.ROOT)):digest(p) for p in paths}}
    (OUT/'evaluation_manifest.json').write_text(json.dumps({'frozen_at':datetime.now(timezone.utc).isoformat(),'files':files,'test_already_seen':True,'phase':'second exploratory follow-up'},indent=2))
    torch.set_num_threads(train['plan']['threads'])
    raw=data.load_raw(cfg['subset'],'test');truth=data.load_test_rul(cfg['subset']).set_index('unit').rul
    all_preds={};rows=[];summaries={};units=None
    for kind in selected['models']:
        runs=[HistoryRun(OUT/f'selected_{kind}_seed{s}.pt') for s in seeds]
        X,_,u,_=frame_features(raw,runs[0].sensor_scaler,runs[0].config['representation'],runs[0].window,True)
        if units is None:units=u
        else:np.testing.assert_array_equal(units,u)
        yr=truth.loc[u].to_numpy();y=np.minimum(yr,cfg['rul_cap'])
        predictions=[]
        for run in runs:
            p=run.predict_features(X);predictions.append(p);m=all_metrics(y,p)
            val=next(r for r in selected['models'][kind]['runs'] if r['seed']==run.seed)
            rows.append({'model':kind,'seed':run.seed,'val_rmse':val['val_rmse'],'test_rmse':m['rmse'],'test_mae':m['mae'],'test_nasa_score':m['nasa_score']})
            all_preds[f'{kind}_seed{run.seed}']=p
        ensemble=np.mean(predictions,axis=0);all_preds[f'{kind}_ensemble']=ensemble
        own=[r for r in rows if r['model']==kind]
        summary={key:{'mean':float(np.mean([r[key] for r in own])),'std':float(np.std([r[key] for r in own],ddof=1))} for key in ('val_rmse','test_rmse','test_mae','test_nasa_score')}
        summary.update(ensemble_test=all_metrics(y,ensemble),ensemble_uncapped_test=all_metrics(yr,ensemble))
        for _ in range(3):
            for r in runs:r.predict_features(X[:1])
        times=[]
        for _ in range(10):
            start=time.perf_counter();np.mean([r.predict_features(X[:1]) for r in runs],axis=0);times.append((time.perf_counter()-start)*1000)
        summary['ensemble_model_ms_batch1']=float(np.median(times));summaries[kind]=summary
        print(kind,'test RMSE',summary['test_rmse'],'ensemble RMSE',summary['ensemble_test']['rmse'],flush=True)
    result={'test_already_seen':True,'n_test_engines':int(len(units)),'rul_cap':cfg['rul_cap'],
            'seeds':seeds,'models':summaries,'selection':selected,'environment':environment(),
            'timing':'3 warmups, median 10 repeats, batch 1, model+standardisation only; history feature extraction excluded',
            'limitations':'Exploratory reused test set. New history information and additional tuning; compare directly to the new history RF. No fresh generalisation claim.'}
    pd.DataFrame(rows).to_csv(OUT/'comparison.csv',index=False)
    pd.DataFrame({'unit':units,'true_rul':truth.loc[units].to_numpy(),'capped_rul':np.minimum(truth.loc[units].to_numpy(),cfg['rul_cap']),**all_preds}).to_csv(OUT/'test_predictions.csv',index=False)
    (OUT/'test_results.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
