"""CLI data ingestion for subscription revenue forecasting."""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
from typing import Any
import pandas as pd
SUPPORTED_EXTENSIONS={'.csv','.json','.jsonl'}

def normalize_column_name(name: Any) -> str:
    return re.sub(r'[^a-z0-9]+','_',str(name).strip().lower()).strip('_')

def read_input_file(path: Path) -> pd.DataFrame:
    suffix=path.suffix.lower()
    if suffix=='.csv': return pd.read_csv(path)
    if suffix=='.jsonl': return pd.read_json(path,lines=True)
    if suffix=='.json':
        with path.open(encoding='utf-8') as f: payload=json.load(f)
        if isinstance(payload,list): records=payload
        elif isinstance(payload,dict): records=payload.get('data',payload.get('records',payload))
        else: raise ValueError('JSON must contain an object or array of records.')
        if isinstance(records,dict): records=[records]
        if not isinstance(records,list): raise ValueError("JSON 'data' or 'records' must be a list.")
        return pd.DataFrame(records)
    raise ValueError(f'Unsupported file format: {path.suffix}')

def load_raw_data(input_dir: str|Path) -> pd.DataFrame:
    folder=Path(input_dir)
    if not folder.exists(): raise FileNotFoundError(f'Input directory does not exist: {folder}')
    if not folder.is_dir(): raise NotADirectoryError(f'Input path is not a directory: {folder}')
    files=sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not files: raise FileNotFoundError(f'No CSV, JSON, or JSONL files found in {folder}')
    frames=[]
    for path in files:
        try: frame=read_input_file(path)
        except Exception as exc: raise ValueError(f'Failed to read {path.name}: {exc}') from exc
        if frame.empty: print(f'WARNING: skipping empty file {path.name}',file=sys.stderr); continue
        frame.columns=[normalize_column_name(c) for c in frame.columns]
        if len(set(frame.columns))!=len(frame.columns): raise ValueError(f'Duplicate column names after normalization in {path.name}')
        frame['_source_file']=path.name
        frames.append(frame)
    if not frames: raise ValueError('All supported input files were empty.')
    return pd.concat(frames,ignore_index=True,sort=False)

def parse_revenue(values: pd.Series) -> pd.Series:
    text=values.astype('string').str.strip()
    paren=text.str.match(r'^\(.*\)$',na=False)
    text=text.str.replace(r'^\((.*)\)$',r'',regex=True).str.replace(',','',regex=False)
    text=text.str.replace(r'[\s$€£₹]','',regex=True)
    result=pd.to_numeric(text,errors='coerce')
    result.loc[paren]=-result.loc[paren].abs()
    return result

def clean_transactions(raw: pd.DataFrame,date_column='date',country_column='country',revenue_column='price'):
    date_column,country_column,revenue_column=map(normalize_column_name,(date_column,country_column,revenue_column))
    required={date_column,country_column,revenue_column}; missing=required-set(raw.columns)
    if missing: raise ValueError(f'Missing required columns: {sorted(missing)}. Available columns: {list(raw.columns)}')
    frame=raw[[date_column,country_column,revenue_column]].copy(); original=len(frame)
    frame.rename(columns={date_column:'date',country_column:'country',revenue_column:'revenue'},inplace=True)
    frame['date']=pd.to_datetime(frame['date'],errors='coerce',utc=True).dt.tz_localize(None).dt.normalize()
    frame['country']=frame['country'].astype('string').str.strip().str.replace(r'\s+',' ',regex=True)
    frame.loc[frame['country']=='','country']=pd.NA
    frame['revenue']=parse_revenue(frame['revenue'])
    bad_date=frame.date.isna(); bad_country=frame.country.isna(); bad_revenue=frame.revenue.isna()
    invalid=bad_date|bad_country|bad_revenue
    report={'input_rows':original,'invalid_date_rows':int(bad_date.sum()),'invalid_country_rows':int(bad_country.sum()),'invalid_revenue_rows':int(bad_revenue.sum()),'dropped_rows':int(invalid.sum())}
    frame=frame.loc[~invalid].copy()
    if frame.empty: raise ValueError('No valid transactions remain after cleaning.')
    if (frame.country.str.casefold()=='all').any(): raise ValueError("Source country 'all' is reserved for the aggregate.")
    frame['country']=frame.country.astype(str)
    return frame.reset_index(drop=True),report

def build_time_series(transactions: pd.DataFrame,frequency='monthly') -> pd.DataFrame:
    frequency=frequency.lower().strip()
    if frequency not in {'daily','monthly'}: raise ValueError("frequency must be 'daily' or 'monthly'.")
    data=transactions[['date','country','revenue']].copy()
    if frequency=='daily': data['date']=data.date.dt.normalize(); freq='D'
    else: data['date']=data.date.dt.to_period('M').dt.to_timestamp(); freq='MS'
    countries=data.groupby(['country','date'],as_index=False).revenue.sum()
    aggregate=data.groupby('date',as_index=False).revenue.sum().assign(country='all')[['country','date','revenue']]
    grouped=pd.concat([countries,aggregate],ignore_index=True)
    dates=pd.date_range(grouped.date.min(),grouped.date.max(),freq=freq,name='date')
    outputs=[]
    for country in sorted(grouped.country.unique()):
        vals=grouped.loc[grouped.country==country].set_index('date').revenue.reindex(dates,fill_value=0.0)
        outputs.append(pd.DataFrame({'date':dates,'country':country,'revenue':vals.to_numpy(dtype=float)}))
    return pd.concat(outputs,ignore_index=True).sort_values(['country','date']).reset_index(drop=True)

def safe_country_filename(country: str) -> str:
    return re.sub(r'[^a-z0-9]+','_',country.casefold()).strip('_') or 'unknown'

def save_outputs(transactions,time_series,output_dir):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); country_dir=out/'countries'; country_dir.mkdir(exist_ok=True)
    tx=out/'cleaned_transactions.csv'; combined=out/'revenue_timeseries.csv'
    transactions.to_csv(tx,index=False,date_format='%Y-%m-%d'); time_series.to_csv(combined,index=False,date_format='%Y-%m-%d')
    mapping={}
    for country,frame in time_series.groupby('country'):
        dest=country_dir/f'{safe_country_filename(country)}.csv'; frame.to_csv(dest,index=False,date_format='%Y-%m-%d'); mapping[country]=str(dest)
    return {'cleaned_transactions':str(tx),'combined_timeseries':str(combined),'country_directory':str(country_dir),'country_files_written':str(len(mapping))}

def ingest_data(input_dir,output_dir,frequency='monthly',date_column='date',country_column='country',revenue_column='price'):
    raw=load_raw_data(input_dir); cleaned,report=clean_transactions(raw,date_column,country_column,revenue_column)
    series=build_time_series(cleaned,frequency); outputs=save_outputs(cleaned,series,output_dir)
    print('Data ingestion completed successfully.')
    for key,value in report.items(): print(f'{key}: {value}')
    print(f'valid_transactions: {len(cleaned)}'); print(f'time_series_rows: {len(series)}')
    for key,value in outputs.items(): print(f'{key}: {value}')
    return series

def main():
    parser=argparse.ArgumentParser(description='Build country-level revenue time series.')
    parser.add_argument('--input-dir',default='data/raw'); parser.add_argument('--output-dir',default='data/processed')
    parser.add_argument('--frequency',choices=['daily','monthly'],default='monthly')
    parser.add_argument('--date-column',default='date'); parser.add_argument('--country-column',default='country'); parser.add_argument('--revenue-column',default='price')
    args=parser.parse_args()
    try: ingest_data(args.input_dir,args.output_dir,args.frequency,args.date_column,args.country_column,args.revenue_column)
    except (ValueError,FileNotFoundError,NotADirectoryError) as exc: print(f'ERROR: {exc}',file=sys.stderr); raise SystemExit(1)
if __name__=='__main__': main()
