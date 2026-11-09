"""Predict RUL with the selected v2 history neural ensemble.

Usage: python scripts/predict_neural_history.py examples/engine_normal.csv
Requires the COMPLETE observed history starting at cycle 1, with at least 30 cycles.
This model uses no forest at inference. It averages the three selected neural seeds.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src import data
from src.neural_history import HistoryRun
from scripts.predict import read_engine,validate,InputError,OutOfScope,SCALED_LIMITS


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('file')
    args=ap.parse_args()
    torch.set_num_threads(2)
    out=data.ROOT/'experiments/neural_history_v2'
    selected=data.load_config(out/'selection.json')
    runs=[HistoryRun(out/f'selected_neural_seed{s}.pt') for s in selected['seeds']]
    sc=runs[0].sensor_scaler
    try:
        engine=validate(read_engine(args.file),sc['features'],runs[0].window)
        if engine.cycle.iloc[0]!=1:
            raise InputError('history model requires the complete observed history beginning at cycle 1')
        if any(r.sensor_scaler!=sc for r in runs):
            raise InputError('selected checkpoints have inconsistent sensor preprocessing')
        scaled=data.apply_scaler(engine,sc)[sc['features']].to_numpy()
        lo,hi=SCALED_LIMITS
        outside=[f for f,v in zip(sc['features'],scaled.T) if v.min()<lo or v.max()>hi]
        if outside:raise OutOfScope(f'sensor history outside the FD001 training range: {outside}')
    except (InputError,OutOfScope) as e:
        print(f'REJECTED: {e}',file=sys.stderr);sys.exit(e.code)
    predictions=np.array([r.predict_frame(engine)[0][0] for r in runs])
    print(f"Engine {int(engine.unit.iloc[0])}: {len(engine)} observed cycles")
    print(f"Estimated remaining useful life: {predictions.mean():.1f} cycles (three-neural-model range {predictions.min():.1f}–{predictions.max():.1f})")
    print('Model: history MLP, initial sensor baseline + observed trends; no Random Forest component.')
    print('Exploratory FD001 simulation model. Reused-benchmark performance is not independent generalisation evidence.')
    print('The prediction estimates a target capped at 125 cycles; it is not a guaranteed time to failure.')


if __name__=='__main__':main()
