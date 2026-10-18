import tempfile, unittest
from pathlib import Path
import numpy as np, pandas as pd
# CI/offline fallback: use a tiny Flask-compatible test shim only when Flask
# is not installed. In a normal install, the real Flask test client is used.
import importlib.util, sys, types, logging
if importlib.util.find_spec('flask') is None:
    fake=types.ModuleType('flask')
    class _Response:
        def __init__(self,data,status=200): self._data=data; self.status_code=status
        def get_json(self): return self._data
    class _Request:
        args={}; _json=None
        def get_json(self,silent=False): return self._json
    _request=_Request()
    class Flask:
        def __init__(self,name): self.config={}; self.extensions={}; self.logger=logging.getLogger(name); self.routes={}
        def route(self,path,methods=None):
            def dec(fn):
                for method in (methods or ['GET']): self.routes[(method,path)]=fn
                return fn
            return dec
        def get(self,path): return self.route(path,['GET'])
        def post(self,path): return self.route(path,['POST'])
        def test_client(self):
            app=self
            class Client:
                def _call(self,method,path,query=None,json=None):
                    from urllib.parse import urlparse, parse_qs
                    parsed=urlparse(path); route_path=parsed.path
                    args=dict(parse_qs(parsed.query)); args.update(query or {})
                    _request.args={k:(v[-1] if isinstance(v,list) else v) for k,v in args.items()}; _request._json=json
                    fn=app.routes.get((method,route_path))
                    if not fn: return _Response({'error':'not found'},404)
                    result=fn()
                    if isinstance(result,tuple): data,status=result
                    else: data,status=result,200
                    return _Response(data,status)
                def get(self,path,query_string=None): return self._call('GET',path,query_string)
                def post(self,path,json=None): return self._call('POST',path,json=json)
            return Client()
    def jsonify(data): return data
    fake.Flask=Flask; fake.request=_request; fake.jsonify=jsonify; sys.modules['flask']=fake
from app import create_app
from model import train_models

class APITests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); base=Path(self.tmp.name)
        self.model_dir=base/'models'/'test'; self.log_dir=base/'logs'/'test'; self.data_path=base/'series.csv'
        dates=pd.date_range('2022-01-01',periods=30,freq='MS'); rows=[]
        for country,mult in [('India',100),('USA',60)]:
            for i,date in enumerate(dates): rows.append({'date':date,'country':country,'revenue':mult+i*2+np.sin(i)})
        data=pd.DataFrame(rows); data.to_csv(self.data_path,index=False)
        train_models(data,self.model_dir,figure_dir=base/'figures')
        self.app=create_app({'TESTING':True,'MODEL_DIR':str(self.model_dir),'LOG_DIR':str(self.log_dir),'DATA_PATH':str(self.data_path),'TEST_MODEL_DIR':str(self.model_dir),'TEST_LOG_DIR':str(self.log_dir),'FIGURE_DIR':str(base/'figures'),'TEST_FIGURE_DIR':str(base/'test_figures')})
        self.client=self.app.test_client()
    def tearDown(self): self.tmp.cleanup()
    def test_health(self): self.assertEqual(self.client.get('/health').status_code,200)
    def test_country_and_all_predictions(self):
        for country in ['India','all']:
            response=self.client.get('/predict',query_string={'country':country,'date':'2025-01-01'})
            self.assertEqual(response.status_code,200,response.get_json()); self.assertIn('predicted_revenue',response.get_json())
    def test_bad_date_and_missing_model(self):
        self.assertEqual(self.client.get('/predict?country=India&date=bad').status_code,400)
        self.assertEqual(self.client.get('/predict?country=Canada&date=2025-01-01').status_code,404)
    def test_logs_endpoint(self):
        self.client.get('/predict?country=India&date=2025-01-01')
        response=self.client.get('/logs?kind=predictions'); self.assertEqual(response.status_code,200); self.assertEqual(len(response.get_json()['predictions']),1)
    def test_train_test_flag_isolated(self):
        response=self.client.post('/train',json={'data_path':str(self.data_path),'test':True})
        self.assertEqual(response.status_code,200,response.get_json()); self.assertTrue((self.log_dir/'training.csv').exists())
if __name__=='__main__': unittest.main()
