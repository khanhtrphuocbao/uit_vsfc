import unittest
import numpy as np
import pandas as pd
import extended_validation as e
import run_experiment as base

class ExtendedTests(unittest.TestCase):
    def setUp(self):
        self.y=np.array(['negative']*5325+['neutral']*458+['positive']*5643)
    def test_ratio_samples_nested_reproducible_and_train_only(self):
        sets=[]
        for count in (46,114,228):
            idx=e.ratio_sample(self.y,count,42)
            self.assertEqual(len(idx),1142)
            self.assertEqual(len(set(idx)),1142)
            self.assertEqual(sum(self.y[idx]=='neutral'),count)
            self.assertTrue((idx>=0).all() and (idx<len(self.y)).all())
            np.testing.assert_array_equal(idx,e.ratio_sample(self.y,count,42))
            sets.append(set(idx))
        for label in ('neutral','negative','positive'):
            subsets=[{i for i in s if self.y[i]==label} for s in sets]
            if label=='neutral': self.assertTrue(subsets[0]<=subsets[1]<=subsets[2])
            else: self.assertTrue(subsets[2]<=subsets[1]<=subsets[0])
    def test_training_only_vectorizer_and_paired_indices(self):
        train=pd.DataFrame({'sentence':['dog apple','cat banana','bird cherry']*4,'label':['negative','neutral','positive']*4})
        val=pd.DataFrame({'sentence':['validationexclusive dog','banana'], 'label':['negative','neutral']})
        idx=np.arange(len(train)); a,pa,model=e.evaluate(train,val,idx,42,'none','word'); b,pb,_=e.evaluate(train,val,idx,42,'balanced','word')
        self.assertNotIn('validationexclusive',model['tfidf'].vocabulary_)
        self.assertEqual(a['indices_sha256'],b['indices_sha256']); self.assertEqual(len(pa),len(val)); self.assertEqual(len(pb),len(val))
        self.assertEqual(a['n_train'],len(train))
    def test_metrics_match_confusion(self):
        y=np.array(['negative','neutral','positive','positive']); p=np.array(['negative','positive','positive','neutral'])
        m=base.metrics(y,p); self.assertAlmostEqual(m['accuracy'],.5); self.assertAlmostEqual(m['neutral_f1'],0); self.assertAlmostEqual(m['positive_f1'],.5)
    def test_test_data_not_accepted(self):
        self.assertNotIn('test',e.evaluate.__code__.co_varnames[:e.evaluate.__code__.co_argcount])
if __name__=='__main__': unittest.main()
