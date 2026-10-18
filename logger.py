"""CSV-based audit logging for training and predictions."""
from __future__ import annotations
import csv, json, os, threading
from datetime import datetime, timezone
from pathlib import Path
TRAIN_FIELDS=['timestamp','country','parameters','runtime_seconds','version','status','metrics','error']
PREDICTION_FIELDS=['timestamp','country','target_date','parameters','runtime_seconds','version','predicted_value','status','error']
class CSVLogger:
    def __init__(self,log_dir='logs/production'):
        self.log_dir=Path(log_dir); self.log_dir.mkdir(parents=True,exist_ok=True); self._lock=threading.Lock()
        self.training_path=self.log_dir/'training.csv'; self.predictions_path=self.log_dir/'predictions.csv'
    @staticmethod
    def _now(): return datetime.now(timezone.utc).isoformat()
    def _append(self,path,fields,row):
        with self._lock:
            exists=path.exists() and path.stat().st_size>0
            with path.open('a',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
                if not exists: writer.writeheader()
                writer.writerow({key:row.get(key,'') for key in fields})
    def log_training(self,country,parameters=None,runtime_seconds=0.0,version='',status='success',metrics=None,error=''):
        self._append(self.training_path,TRAIN_FIELDS,{'timestamp':self._now(),'country':country,'parameters':json.dumps(parameters or {},sort_keys=True),'runtime_seconds':round(float(runtime_seconds),6),'version':version,'status':status,'metrics':json.dumps(metrics or {},sort_keys=True),'error':error})
    def log_prediction(self,country,target_date,parameters=None,runtime_seconds=0.0,version='',predicted_value=None,status='success',error=''):
        self._append(self.predictions_path,PREDICTION_FIELDS,{'timestamp':self._now(),'country':country,'target_date':str(target_date),'parameters':json.dumps(parameters or {},sort_keys=True),'runtime_seconds':round(float(runtime_seconds),6),'version':version,'predicted_value':predicted_value,'status':status,'error':error})
    def read_logs(self,kind='all',limit=500):
        import pandas as pd
        if limit<1: raise ValueError('limit must be positive')
        paths={'training':self.training_path,'predictions':self.predictions_path}
        if kind not in {'all',*paths}: raise ValueError("kind must be 'all', 'training', or 'predictions'")
        result={}
        for name,path in paths.items():
            if kind not in ('all',name): continue
            result[name]=pd.read_csv(path).tail(limit).fillna('').to_dict(orient='records') if path.exists() else []
        return result
