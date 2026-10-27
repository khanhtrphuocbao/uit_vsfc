import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import run_experiment as base
import run_final_test as runner

class FinalRunnerTests(unittest.TestCase):
    def setUp(self):
        self.plan=json.loads(runner.PLAN_PATH.read_text())
    def test_requires_approval_before_source_loading(self):
        with patch.object(base,'load_source',side_effect=AssertionError('source accessed')):
            with self.assertRaises(SystemExit): runner.main([])
            with self.assertRaises(SystemExit): runner.main(['--approve-final-test'])
    def test_rejects_changed_plan_and_extra_configuration(self):
        plan=copy.deepcopy(self.plan); plan['configurations'].append(plan['configurations'][0])
        with self.assertRaises(ValueError): runner.validate_plan(plan)
        plan=copy.deepcopy(self.plan); plan['configurations'][0]['feature']='char'
        with self.assertRaises(ValueError): runner.validate_plan(plan)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'changed.json'; path.write_text(json.dumps(plan))
            with self.assertRaises(ValueError): runner.read_plan(path)
    def test_saved_indices_match_baseline_and_tampering_is_rejected(self):
        indices=runner.load_indices(self.plan)
        self.assertEqual(len(indices[0.1]),1142); self.assertEqual(len(indices[1.0]),11426)
        with tempfile.TemporaryDirectory() as tmp:
            rows=json.loads(runner.SAMPLES_PATH.read_text()); item=next(x for x in rows if x['phase']=='feature_comparison' and x.get('train_fraction')==.1 and x['seed']==42); item['indices'][0]=item['indices'][1]
            path=Path(tmp)/'samples.json'; path.write_text(json.dumps(rows))
            with self.assertRaises(ValueError): runner.load_indices(self.plan,path)
    def test_pipeline_parameters_checked(self):
        runner.check_pipelines(self.plan)
        bad=copy.deepcopy(self.plan); bad['parameters']['classifier']['solver']='wrong'
        with self.assertRaises(ValueError): runner.check_pipelines(bad)
    def test_synthetic_six_models_and_bootstrap(self):
        train=pd.DataFrame({'sentence':['bad lesson','ordinary class','good lesson']*8,'label':['negative','neutral','positive']*8})
        test=pd.DataFrame({'sentence':['bad lesson','ordinary class','good lesson']*3,'label':['negative','neutral','positive']*3,'row_index':range(9)})
        configs=copy.deepcopy(self.plan['configurations']); indices={1.:np.arange(24),.1:np.arange(12)}
        results,predictions,conf=runner.run_models(train,test,configs,indices)
        self.assertEqual(len(results),6); self.assertEqual(set(predictions.config_id),{'A1','A2','B1','B2','C1','C2'})
        self.assertEqual(predictions.groupby('config_id').size().tolist(),[9]*6); self.assertEqual(len(conf),54)
        both=np.array(['negative','neutral','positive']*3)
        boot=runner.paired_bootstrap(both,both,both,50,2026)
        self.assertEqual((boot['lower'],boot['upper'],boot['point_estimate']),(0.,0.,0.))
        self.assertEqual(boot,runner.paired_bootstrap(both,both,both,50,2026))
    def test_synthetic_entrypoint_writes_six_prediction_sets_and_one_bootstrap(self):
        train=pd.DataFrame({'sentence':['bad lesson','ordinary class','good lesson']*8,'label':['negative','neutral','positive']*8})
        evaluation=pd.DataFrame({'sentence':['bad lesson','ordinary class','good lesson']*1055+['good lesson'], 'label':['negative','neutral','positive']*1055+['positive'], 'row_index':range(3166)})
        data={'train':train,'test':evaluation}
        synthetic_indices={1.:np.arange(24),.1:np.arange(12)}
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'run'
            with patch.object(base,'load_source',return_value=(object(),{'loading_method':'synthetic'})), patch.object(base,'prepare',return_value=data), patch.object(runner,'load_indices',return_value=synthetic_indices), patch.object(runner,'verify_source',return_value={'synthetic':'only'}), patch.object(runner,'verify_sample_counts'), patch.object(runner,'paired_bootstrap',return_value={'comparison':'A2 minus A1'}) as bootstrap:
                runner.main(['--approve-final-test','--plan-sha256',runner.FROZEN_PLAN_SHA256,'--output-dir',str(out)])
            predictions=pd.read_csv(out/'test_predictions.csv')
            self.assertEqual(predictions.groupby('config_id').size().tolist(),[3166]*6)
            self.assertEqual(len(pd.read_csv(out/'test_results.csv')),6)
            self.assertEqual(len(list(out.glob('*bootstrap*'))),1)
            self.assertEqual(bootstrap.call_count,1)
            self.assertEqual(json.loads((out/'manifest.json').read_text())['status'],'COMPLETED')
            with self.assertRaises(FileExistsError):
                runner.main(['--approve-final-test','--plan-sha256',runner.FROZEN_PLAN_SHA256,'--output-dir',str(out)])

    def test_source_hash_mismatch_fails_before_fit(self):
        data={'train':pd.DataFrame({'sentence':['x']*11426,'label':['negative']*11426,'topic':['lecturer']*11426})}
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch: train'): runner.verify_source(data,self.plan)

if __name__=='__main__': unittest.main()
