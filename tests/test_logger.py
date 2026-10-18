import csv, tempfile, unittest
from pathlib import Path
from logger import CSVLogger

class LoggerTests(unittest.TestCase):
    def test_training_and_prediction_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger=CSVLogger(tmp); logger.log_training('India',{'seed':42},.25,'v1',metrics={'rmse':3.2})
            logger.log_prediction('India','2025-01-01',{},.01,'v1',123.4)
            self.assertTrue(logger.training_path.exists()); self.assertTrue(logger.predictions_path.exists())
            with logger.training_path.open() as f: self.assertEqual(len(list(csv.DictReader(f))),1)
            logs=logger.read_logs('all'); self.assertEqual(len(logs['training']),1); self.assertEqual(logs['predictions'][0]['country'],'India')
    def test_invalid_kind(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError): CSVLogger(tmp).read_logs('bogus')
if __name__=='__main__': unittest.main()
