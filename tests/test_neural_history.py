import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src import data
from src.neural_history import engine_features, frame_features, training_arrays, fit_scaler, transform, HistoryMLP, HistoryRun


class HistoryTests(unittest.TestCase):
    def test_future_observations_cannot_change_features(self):
        rng=np.random.default_rng(12)
        X=rng.normal(size=(90,14)); cycle=np.arange(1,91)
        for rep in ("history","pooled_history"):
            expected=engine_features(X[:40],cycle[:40],[40],rep)
            changed=X.copy();changed[40:]=1e9
            actual=engine_features(changed,cycle,[40],rep)
            np.testing.assert_array_equal(actual,expected)

    def test_target_and_unit_identity_do_not_enter_features(self):
        cols=[f's{i}' for i in range(14)]
        rng=np.random.default_rng(12)
        df=pd.DataFrame(rng.normal(size=(45,14)),columns=cols)
        df['cycle']=np.arange(1,46);df['unit']=1;df['rul']=np.arange(45)
        sc={'features':cols,'min':[0.0]*14,'max':[1.0]*14}
        a=frame_features(df,sc,'history')[0]
        df['rul']=99999;df['unit']=500
        b=frame_features(df,sc,'history')[0]
        np.testing.assert_array_equal(a,b)

    def test_training_does_not_load_test_files(self):
        original=data.load_raw
        def restricted(subset,part):
            self.assertEqual(part,'train')
            return original(subset,part)
        with patch.object(data,'load_raw',side_effect=restricted),patch.object(data,'load_test_rul',side_effect=AssertionError('test labels accessed')):
            tr,va,_=training_arrays(data.load_config(data.ROOT/'configs/data_fd001.json'),'history')
        self.assertFalse(set(tr[2])&set(va[2]))
        self.assertEqual(tr[0].shape[1],113)

    def test_checkpoint_roundtrip_both_outputs(self):
        X=np.random.default_rng(0).normal(size=(8,113)).astype(np.float32)
        sc=fit_scaler(X)
        for output in ('sigmoid','linear'):
            c={'hidden':[16,8],'dropout':0.1,'output':output,'representation':'history'}
            m=HistoryMLP(113,c).eval()
            with torch.no_grad():p=m(torch.from_numpy(transform(X,sc))).numpy().clip(0,1)*125
            with tempfile.TemporaryDirectory() as td:
                path=Path(td)/'checkpoint.pt'
                torch.save({'kind':'neural','config':c,'seed':0,'scaler':sc,'sensor_scaler':{},'cap':125,'window':30,'input_dim':113,'state_dict':m.state_dict()},path)
                q=HistoryRun(path).predict_features(X)
            np.testing.assert_allclose(p,q)


if __name__=='__main__':unittest.main()
