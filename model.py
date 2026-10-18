"""Time-series training, versioning, loading and recursive forecasting."""
from __future__ import annotations
import json, math, time
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

FEATURES=['time_idx','lag_1','lag_2','lag_3','rolling_mean_3','month_sin','month_cos']

def _features(values, dates):
    vals=list(map(float,values)); rows=[]
    for i,date in enumerate(dates):
        lag=lambda k: vals[i-k] if i>=k else (vals[0] if vals else 0.0)
        recent=vals[max(0,i-3):i] or [vals[0] if vals else 0.0]
        month=pd.Timestamp(date).month
        rows.append([i,lag(1),lag(2),lag(3),float(np.mean(recent)),math.sin(2*math.pi*month/12),math.cos(2*math.pi*month/12)])
    return pd.DataFrame(rows,columns=FEATURES)

def _metrics(actual,pred):
    return {'rmse':float(np.sqrt(mean_squared_error(actual,pred))),'mae':float(mean_absolute_error(actual,pred))}

def _make_estimators(random_state=42):
    return {
      'random_forest':RandomForestRegressor(n_estimators=150,min_samples_leaf=2,random_state=random_state,n_jobs=1),
      'gradient_boosting':GradientBoostingRegressor(n_estimators=100,learning_rate=0.04,max_depth=2,loss='huber',random_state=random_state),
      'linear_regression':make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LinearRegression()),
    }

def _series_for_country(data,country):
    frame=data[data.country.astype(str).str.casefold()==country.casefold()].copy()
    if frame.empty: raise ValueError(f"No data found for country '{country}'.")
    frame['date']=pd.to_datetime(frame.date,errors='raise')
    frame=frame.groupby('date',as_index=False).revenue.sum().sort_values('date')
    return frame

def _prepare_data(data):
    if not {'date','country','revenue'}.issubset(data.columns): raise ValueError("Data must contain 'date', 'country', and 'revenue' columns.")
    frame=data[['date','country','revenue']].copy(); frame['date']=pd.to_datetime(frame.date,errors='coerce'); frame['revenue']=pd.to_numeric(frame.revenue,errors='coerce')
    frame=frame.dropna(subset=['date','country','revenue'])
    if frame.empty: raise ValueError('No valid time-series rows supplied.')
    if (frame.country.astype(str).str.casefold()=='all').any():
        # An existing aggregate is used as-is; if absent, construct it from countries.
        pass
    else:
        all_rows=frame.groupby('date',as_index=False).revenue.sum(); all_rows['country']='all'; frame=pd.concat([frame,all_rows],ignore_index=True)
    return frame

def train_one_series(frame,random_state=42,test_size=0.2):
    frame=frame.sort_values('date').drop_duplicates('date',keep='last').reset_index(drop=True)
    y=frame.revenue.astype(float).to_numpy(); dates=pd.DatetimeIndex(frame.date)
    if len(y)<5: raise ValueError(f'At least 5 time points are required; got {len(y)}.')
    X=_features(y,dates)
    split=max(3,int(len(y)*(1-test_size)))
    split=min(split,len(y)-1)
    # Features at time t only use observations before t; validation is chronological.
    train_idx=np.arange(1,split); test_idx=np.arange(split,len(y))
    if len(train_idx)<2 or len(test_idx)<1: raise ValueError('Not enough observations for a chronological holdout.')
    actual=y[test_idx]
    baseline=np.array([y[i-1] for i in test_idx])
    results={'naive_baseline':{'metrics':_metrics(actual,baseline),'predictions':baseline.tolist()}}
    estimators=_make_estimators(random_state)
    fitted={}
    for name,estimator in estimators.items():
        estimator.fit(X.iloc[train_idx],y[train_idx]); pred=estimator.predict(X.iloc[test_idx])
        results[name]={'metrics':_metrics(actual,pred),'predictions':pred.astype(float).tolist()}; fitted[name]=estimator
    best_name=min(fitted,key=lambda name:(results[name]['metrics']['rmse'],results[name]['metrics']['mae']))
    # Train selected model on all available feature-target pairs for production forecasting.
    best_model=_make_estimators(random_state)[best_name]
    best_model.fit(X.iloc[1:],y[1:])
    return {'model':best_model,'model_name':best_name,'metrics':{name:item['metrics'] for name,item in results.items()},'holdout':{'dates':[d.strftime('%Y-%m-%d') for d in dates[test_idx]],'actual':actual.astype(float).tolist(),'predictions':{name:item['predictions'] for name,item in results.items()}},'history':{'dates':[d.strftime('%Y-%m-%d') for d in dates],'values':y.tolist()},'feature_names':FEATURES}

def _version(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')

def train_models(data,model_dir='models/production',random_state=42,test_size=0.2,figure_dir='eda/figures'):
    frame=_prepare_data(data); out=Path(model_dir); version=_version(); results={}
    for country in sorted(frame.country.astype(str).unique()):
        series=_series_for_country(frame,country)
        trained=train_one_series(series,random_state,test_size)
        artifact={'country':country,'version':version,'created_at':datetime.now(timezone.utc).isoformat(),'frequency':_infer_frequency(series.date),'model_name':trained['model_name'],'model':trained['model'],'metrics':trained['metrics'],'holdout':trained['holdout'],'history':trained['history'],'feature_names':FEATURES}
        country_dir=out/_country_slug(country); country_dir.mkdir(parents=True,exist_ok=True)
        artifact_path=country_dir/f'{version}.joblib'; joblib.dump(artifact,artifact_path)
        latest_path=country_dir/'latest.json'; latest_path.write_text(json.dumps({'version':version,'artifact':artifact_path.name},indent=2),encoding='utf-8')
        results[country]={'version':version,'model_name':trained['model_name'],'metrics':trained['metrics'],'artifact':str(artifact_path),'selected_metrics':trained['metrics'][trained['model_name']]}
    if figure_dir:
        _save_baseline_comparison(results,figure_dir)
    return {'version':version,'models':results,'model_dir':str(out)}


def _save_baseline_comparison(results,figure_dir):
    """Save a model-vs-naive holdout RMSE comparison chart."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=Path(figure_dir); folder.mkdir(parents=True,exist_ok=True)
    rows=[]
    for country,info in results.items():
        metrics=info['metrics']
        baseline=metrics.get('naive_baseline',{}).get('rmse')
        selected=metrics.get(info['model_name'],{}).get('rmse')
        if baseline is not None and selected is not None: rows.append((country,baseline,selected))
    if not rows: return
    labels=[r[0] for r in rows]; baseline=[r[1] for r in rows]; selected=[r[2] for r in rows]
    x=np.arange(len(labels)); width=0.38
    fig,ax=plt.subplots(figsize=(max(8,len(labels)*1.1),5))
    ax.bar(x-width/2,baseline,width,label='Naive baseline RMSE')
    ax.bar(x+width/2,selected,width,label='Selected model RMSE')
    ax.set_xticks(x,labels,rotation=35,ha='right'); ax.set_ylabel('RMSE (lower is better)')
    ax.set_title('Holdout forecast error: selected model vs naive baseline'); ax.legend(); fig.tight_layout()
    fig.savefig(folder/'baseline_vs_selected.png',dpi=140); plt.close(fig)

def _infer_frequency(dates):
    dates=pd.to_datetime(pd.Series(dates)).sort_values().drop_duplicates()
    if len(dates)>1 and dates.dt.day.eq(1).all() and dates.dt.to_period('M').astype('int64').diff().dropna().eq(1).all(): return 'monthly'
    return 'daily'

def _country_slug(country):
    import re
    return re.sub(r'[^a-z0-9]+','_',str(country).casefold()).strip('_') or 'unknown'

def load_model(country,model_dir='models/production',version=None):
    folder=Path(model_dir)/_country_slug(country)
    if version:
        path=folder/f'{version}.joblib'
    else:
        pointer=folder/'latest.json'
        if not pointer.exists(): raise FileNotFoundError(f"No trained model found for '{country}'. Train a model first.")
        info=json.loads(pointer.read_text(encoding='utf-8')); path=folder/info['artifact']
    if not path.exists(): raise FileNotFoundError(f'Model artifact not found: {path}')
    return joblib.load(path)

def predict(country,target_date,model_dir='models/production'):
    artifact=load_model(country,model_dir)
    target=pd.Timestamp(target_date)
    if pd.isna(target): raise ValueError('target_date must be a valid date.')
    target=target.normalize(); dates=pd.to_datetime(artifact['history']['dates']); values=list(map(float,artifact['history']['values']))
    if target<=dates[-1]: raise ValueError(f"target_date must be after the last training date ({dates[-1].date()}).")
    freq=artifact.get('frequency','monthly')
    current=dates[-1]
    # Forecast along the series calendar so lag features are constructed consistently.
    if freq=='monthly': current=current.to_period('M').to_timestamp(); target=target.to_period('M').to_timestamp(); offset=pd.offsets.MonthBegin(1)
    else: offset=pd.offsets.Day(1)
    while current<target:
        current=current+offset
        Xnext=_features(values+[0.0],[current])
        # _features is indexed relative to its input; set the correct time index and lags.
        n=len(values); month=current.month
        recent=values[-3:] or [0.0]
        row=pd.DataFrame([[n,values[-1],values[-2] if n>=2 else values[-1],values[-3] if n>=3 else values[-1],float(np.mean(recent)),math.sin(2*math.pi*month/12),math.cos(2*math.pi*month/12)]],columns=FEATURES)
        prediction=float(artifact['model'].predict(row)[0]); values.append(prediction)
    return {'country':country,'date':target.strftime('%Y-%m-%d'),'predicted_revenue':float(values[-1]),'model_version':artifact['version'],'model_name':artifact['model_name'],'frequency':freq}

def load_training_data(path):
    data=pd.read_csv(path)
    if not {'date','country','revenue'}.issubset(data.columns): raise ValueError('Training CSV must contain date,country,revenue. Run data_ingestion.py first.')
    return data
