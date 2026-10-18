"""Flask API for subscription revenue forecasting."""
from __future__ import annotations
import os, time
from pathlib import Path
import pandas as pd
from flask import Flask, jsonify, request
from data_ingestion import ingest_data
from logger import CSVLogger
from model import train_models, load_training_data, predict, load_model
from monitoring import monitor_artifact, detect_input_drift, log_monitoring

def create_app(config=None):
    app=Flask(__name__)
    base=Path(__file__).resolve().parent
    app.config.update(DATA_PATH=str(base/'data/processed/revenue_timeseries.csv'),RAW_DIR=str(base/'data/raw'),PROCESSED_DIR=str(base/'data/processed'),MODEL_DIR=str(base/'models/production'),LOG_DIR=str(base/'logs/production'),FIGURE_DIR=str(base/'eda/figures'),TESTING=False)
    if config: app.config.update(config)
    logger=CSVLogger(app.config['LOG_DIR'])
    app.extensions['forecast_logger']=logger
    def error(message,status): return jsonify({'error':message}),status

    @app.get('/health')
    def health(): return jsonify({'status':'ok','service':'subscription-revenue-forecast'})

    @app.route('/train',methods=['POST'])
    def train():
        started=time.perf_counter(); payload=request.get_json(silent=True) or {}
        test_mode=bool(payload.get('test',False))
        model_dir=app.config['MODEL_DIR']; log_dir=app.config['LOG_DIR']
        figure_dir=app.config['FIGURE_DIR']
        if test_mode:
            model_dir=app.config.get('TEST_MODEL_DIR',str(base/'models/test'))
            log_dir=app.config.get('TEST_LOG_DIR',str(base/'logs/test'))
            figure_dir=app.config.get('TEST_FIGURE_DIR',str(base/'eda/test_figures'))
            # Do not reuse the production logger when a test flag is supplied.
            local_logger=CSVLogger(log_dir)
        else: local_logger=logger
        try:
            data_path=payload.get('data_path',app.config['DATA_PATH'])
            if payload.get('ingest',False):
                ingest_data(payload.get('raw_dir',app.config['RAW_DIR']),app.config['PROCESSED_DIR'],payload.get('frequency','monthly'))
                data_path=str(Path(app.config['PROCESSED_DIR'])/'revenue_timeseries.csv')
            data=load_training_data(data_path)
            result=train_models(data,model_dir=model_dir,random_state=int(payload.get('random_state',42)),test_size=float(payload.get('test_size',0.2)),figure_dir=figure_dir)
            elapsed=time.perf_counter()-started
            for country,info in result['models'].items():
                local_logger.log_training(country,{'random_state':int(payload.get('random_state',42)),'test_size':float(payload.get('test_size',0.2)),'test_mode':test_mode},elapsed,info['version'],'success',info['metrics'])
            # Save holdout performance monitoring records from the selected model.
            for country in result['models']:
                artifact=load_model(country,model_dir)
                monitor_artifact(artifact,log_dir=log_dir)
            return jsonify({'status':'trained','test_mode':test_mode,**result}),200
        except (ValueError,FileNotFoundError,KeyError) as exc:
            local_logger.log_training('all',{'test_mode':test_mode},time.perf_counter()-started,'','error',{},str(exc))
            return error(str(exc),400 if isinstance(exc,ValueError) else 404)
        except Exception as exc:
            local_logger.log_training('all',{'test_mode':test_mode},time.perf_counter()-started,'','error',{},str(exc))
            app.logger.exception('Training failed')
            return error('Training failed due to an internal error.',500)

    @app.route('/predict',methods=['GET','POST'])
    def forecast():
        started=time.perf_counter(); payload=request.get_json(silent=True) or {}
        country=request.args.get('country',payload.get('country','all'))
        date=request.args.get('date',payload.get('date'))
        if not isinstance(country,str) or not country.strip(): return error("'country' must be a non-empty string.",400)
        if not date: return error("Missing required 'date' in YYYY-MM-DD format.",400)
        try:
            parsed=pd.to_datetime(date,format='%Y-%m-%d',errors='raise')
            if parsed.strftime('%Y-%m-%d')!=date: return error("'date' must use YYYY-MM-DD format.",400)
        except (ValueError,TypeError): return error("'date' must use YYYY-MM-DD format.",400)
        try:
            result=predict(country.strip(),parsed,app.config['MODEL_DIR'])
            logger.log_prediction(country.strip(),result['date'],{'method':'recursive_forecast'},time.perf_counter()-started,result['model_version'],result['predicted_revenue'])
            return jsonify(result),200
        except ValueError as exc:
            logger.log_prediction(country.strip(),date,{},time.perf_counter()-started,'',None,'error',str(exc))
            return error(str(exc),400)
        except FileNotFoundError as exc:
            logger.log_prediction(country.strip(),date,{},time.perf_counter()-started,'',None,'error',str(exc))
            return error(str(exc),404)
        except Exception:
            app.logger.exception('Prediction failed')
            return error('Prediction failed due to an internal error.',500)

    @app.get('/logs')
    def logs():
        kind=request.args.get('kind','all'); limit=request.args.get('limit','200')
        try: limit=int(limit)
        except ValueError: return error("'limit' must be an integer.",400)
        try: return jsonify(logger.read_logs(kind,limit)),200
        except ValueError as exc: return error(str(exc),400)

    @app.post('/monitor')
    def monitor():
        payload=request.get_json(silent=True) or {}; country=payload.get('country','all')
        try:
            artifact=load_model(country,app.config['MODEL_DIR'])
            if 'actual' in payload and 'predicted' in payload:
                from monitoring import evaluate_predictions
                result={'kind':'performance','country':country,'version':artifact['version'],**evaluate_predictions(payload['actual'],payload['predicted'],payload.get('threshold'))}
                log_monitoring(result,app.config['LOG_DIR'])
            elif 'current_values' in payload and 'reference_values' in payload:
                result={'kind':'drift','country':country,**detect_input_drift(payload['reference_values'],payload['current_values'],payload.get('z_threshold',3.0))}
                log_monitoring(result,app.config['LOG_DIR'])
            else: result=monitor_artifact(artifact,threshold=payload.get('threshold'),log_dir=app.config['LOG_DIR'])
            return jsonify(result),200
        except (ValueError,FileNotFoundError) as exc: return error(str(exc),400 if isinstance(exc,ValueError) else 404)

    return app

app=create_app()
if __name__=='__main__': app.run(host=os.environ.get('HOST','0.0.0.0'),port=int(os.environ.get('PORT','5000')),debug=False)
