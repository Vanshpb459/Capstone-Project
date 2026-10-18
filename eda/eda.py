"""Exploratory data analysis; saves charts under eda/figures."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_eda(input_path='data/processed/revenue_timeseries.csv',output_dir='eda/figures'):
    data=pd.read_csv(input_path,parse_dates=['date'])
    required={'date','country','revenue'}
    if not required.issubset(data.columns): raise ValueError(f'Input must contain {sorted(required)}')
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    all_data=data[data.country.astype(str).str.casefold()=='all'].sort_values('date')
    if all_data.empty: raise ValueError("No 'all' aggregate series found; run data_ingestion.py first.")
    fig,ax=plt.subplots(figsize=(10,5)); ax.plot(all_data.date,all_data.revenue,marker='o',linewidth=1.5); ax.set(title='Total revenue over time',xlabel='Date',ylabel='Revenue'); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out/'revenue_over_time.png',dpi=140); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5)); ax.hist(all_data.revenue,bins=min(20,max(5,len(all_data)//2)),edgecolor='black'); ax.set(title='Distribution of total revenue',xlabel='Revenue',ylabel='Frequency'); fig.tight_layout(); fig.savefig(out/'revenue_distribution.png',dpi=140); plt.close(fig)
    country_data=data[data.country.astype(str).str.casefold()!='all']
    totals=country_data.groupby('country').revenue.sum().sort_values(ascending=False)
    fig,ax=plt.subplots(figsize=(9,5)); totals.plot(kind='bar',ax=ax); ax.set(title='Revenue by country (sum over observed period)',xlabel='Country',ylabel='Revenue'); fig.tight_layout(); fig.savefig(out/'revenue_by_country.png',dpi=140); plt.close(fig)
    monthly=all_data.set_index('date').revenue.resample('MS').sum()
    fig,ax=plt.subplots(figsize=(10,5)); ax.plot(monthly.index,monthly.rolling(3,min_periods=1).mean(),label='3-period rolling mean'); ax.plot(monthly.index,monthly,alpha=.45,label='Revenue'); ax.set(title='Trend and seasonality overview',xlabel='Date',ylabel='Revenue'); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out/'trend_seasonality.png',dpi=140); plt.close(fig)
    return sorted(str(p) for p in out.glob('*.png'))

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--input',default='data/processed/revenue_timeseries.csv'); parser.add_argument('--output-dir',default='eda/figures'); args=parser.parse_args()
    for path in run_eda(args.input,args.output_dir): print(path)
if __name__=='__main__': main()
