import tempfile, unittest
from pathlib import Path
import numpy as np, pandas as pd
from model import train_models, load_model, predict
from monitoring import evaluate_predictions, detect_input_drift

class ModelTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.model_dir=Path(self.tmp.name)/'models'
        dates=pd.date_range('2022-01-01',periods=36,freq='MS'); rows=[]
        for country,mult in [('India',100),('USA',80)]:
            for i,date in enumerate(dates): rows.append({'date':date,'country':country,'revenue':mult+3*i+10*np.sin(2*np.pi*i/12)})
        self.data=pd.DataFrame(rows)
    def tearDown(self): self.tmp.cleanup()
    def test_train_save_load_predict_country_and_all(self):
        result=train_models(self.data,self.model_dir,figure_dir=Path(self.tmp.name)/'figures')
        self.assertIn('India',result['models']); self.assertIn('all',result['models'])
        artifact=load_model('India',self.model_dir); self.assertTrue(artifact['version']); self.assertEqual(artifact['frequency'],'monthly')
        forecast=predict('India','2025-01-01',self.model_dir)
        self.assertEqual(forecast['country'],'India'); self.assertTrue(np.isfinite(forecast['predicted_revenue']))
        self.assertTrue(np.isfinite(predict('all','2025-01-01',self.model_dir)['predicted_revenue']))
    def test_missing_country_raises(self):
        short=pd.DataFrame([{'date':'2024-01-01','country':'India','revenue':1.0}]*4)
        with self.assertRaises(ValueError): train_models(short,self.model_dir,figure_dir=Path(self.tmp.name)/'figures')
    def test_monitoring(self):
        result=evaluate_predictions([1,2,3],[1,2,4],threshold=.5); self.assertAlmostEqual(result['mae'],1/3); self.assertTrue(result['threshold_exceeded'])
        drift=detect_input_drift([1,2,3,4],[100,101]); self.assertTrue(drift['drift_detected'])

if __name__=='__main__': unittest.main()
