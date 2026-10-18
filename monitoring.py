"""Lightweight model performance and input-drift monitoring."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_predictions(actual,predicted,threshold=None):
    actual=np.asarray(actual,dtype=float); predicted=np.asarray(predicted,dtype=float)
    if len(actual)==0 or len(actual)!=len(predicted): raise ValueError('actual and predicted must have equal, non-zero lengths.')
    rmse=float(np.sqrt(mean_squared_error(actual,predicted))); mae=float(mean_absolute_error(actual,predicted))
    return {'rmse':rmse,'mae':mae,'threshold':threshold,'threshold_exceeded':bool(threshold is not None and rmse>threshold),'n_observations':int(len(actual))}

def detect_input_drift(reference,current,z_threshold=3.0):
    """Simple numeric mean-shift check using pooled scale; not a formal test."""
    ref=pd.to_numeric(pd.Series(reference),errors='coerce').dropna().to_numpy(dtype=float)
    cur=pd.to_numeric(pd.Series(current),errors='coerce').dropna().to_numpy(dtype=float)
    if len(ref)<2 or len(cur)<1: raise ValueError('Need >=2 reference values and >=1 current value.')
    scale=float(np.std(ref,ddof=1)); shift=float(abs(np.mean(cur)-np.mean(ref)))
    if scale==0: score=0.0 if shift==0 else float('inf')
    else: score=shift/scale
    return {'reference_mean':float(np.mean(ref)),'current_mean':float(np.mean(cur)),'standardized_mean_shift':score,'threshold':float(z_threshold),'drift_detected':bool(score>z_threshold),'reference_n':len(ref),'current_n':len(cur)}

def log_monitoring(result,log_dir='logs/production'):
    folder=Path(log_dir); folder.mkdir(parents=True,exist_ok=True); path=folder/'monitoring.csv'
    row={'timestamp':pd.Timestamp.now(tz='UTC').isoformat(),'kind':result.get('kind','performance'),**result}
    pd.DataFrame([{'timestamp':row['timestamp'],'kind':row['kind'],'result_json':json.dumps(result,sort_keys=True,allow_nan=False)}]).to_csv(path,mode='a',header=not path.exists(),index=False)
    return str(path)

def monitor_artifact(artifact,new_actual=None,new_predicted=None,threshold=None,log_dir='logs/production'):
    result={'kind':'performance','country':artifact.get('country'),'version':artifact.get('version')}
    if new_actual is not None and new_predicted is not None: result.update(evaluate_predictions(new_actual,new_predicted,threshold))
    else:
        holdout=artifact.get('holdout',{}); actual=holdout.get('actual',[]); predicted=holdout.get('predictions',{}).get(artifact.get('model_name'),[])
        if actual and predicted: result.update(evaluate_predictions(actual,predicted,threshold))
        else: result.update({'status':'no_holdout_metrics'})
    log_monitoring(result,log_dir)
    return result
