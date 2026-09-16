import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from app.domain import disclosure_status,unique_action_ids,unambiguous_type_mapping
from app.facts import render_grounded_answer, GroundingError, verify_answer_numbers, load_facts
from app import indexing,config
from fastapi.testclient import TestClient
from app.main import app
import pandas as pd

class BackendTests(unittest.TestCase):
    def test_unique_action_counts_real_data(self):
        rows=json.loads(config.DATA_PATH.read_text());docs=indexing.build_record_documents()
        self.assertEqual(len(docs),12811)
        for row,doc in zip(rows,docs):self.assertEqual(doc['metadata']['plugin_count'],len(set(row['plugin_id_filenames'])))
    def test_all_audit_counts_match_notebook_facts(self):
        from collections import Counter
        counts=Counter(d['metadata']['status'] for d in indexing.build_audit_documents())
        self.assertEqual(counts,{'UNDISCLOSED':278,'DISCLOSED_CLEAR':16,'DISCLOSED_VAGUE':7,'DISCLOSED_AMBIGUOUS':1,'DISCLOSED_INCORRECT':6})
        self.assertEqual(disclosure_status({'collection':[{'label':'INCORRECT','sentence':'i'},{'label':'AMBIGUOUS','sentence':'a'}]}),('DISCLOSED_AMBIGUOUS','a'))
    def test_taxonomy_conflict_not_silently_collapsed(self):
        df=pd.DataFrame({'data_type':['x','x','y'],'main_data_type':['A','B','C']})
        self.assertEqual(unambiguous_type_mapping(df),{'y':'C'})
    def test_values_and_labels_render_from_same_fact(self):
        key='category_prediction_model.embedding_model.accuracy_pct'
        facts=load_facts()
        value=facts['category_prediction_model']['embedding_model']['accuracy_pct']
        answer=render_grounded_answer({'explanation':f'The word-vector result is {{{{{key}}}}} percent.'})
        self.assertIn(f'{value:g} percent',answer)
        with self.assertRaises(GroundingError):render_grounded_answer({'explanation':'TF-IDF accuracy is 54.9%.'})
        with self.assertRaises(GroundingError):render_grounded_answer({'explanation':'A result of {{invented.accuracy}}.'})
        with self.assertRaises(GroundingError):render_grounded_answer({'explanation':'A result.','fact_ids':[key]})
    def test_number_format_diagnostic(self):
        self.assertEqual(verify_answer_numbers('There are 12,811 records.'),[])
        self.assertEqual(verify_answer_numbers('There are 12811.0 records.'),[])
    def test_http_validation_and_static_chart(self):
        client=TestClient(app)
        self.assertEqual(client.get('/health').status_code,200)
        self.assertEqual(client.post('/ask',json={'question':' '}).status_code,400)
        for k in [-1,0,1000000]:self.assertEqual(client.post('/ask',json={'question':'test','top_k':k}).status_code,422)
        self.assertEqual(client.post('/ask',json={'question':'x'*4001}).status_code,422)
        for p in (config.STATIC_DIR/'charts').glob('*'):self.assertEqual(client.get('/static/charts/'+p.name).status_code,200)
    def test_stale_index_fails_readiness(self):
        with patch('app.index_state.index_is_current',return_value=False):
            self.assertEqual(TestClient(app).get('/ready').status_code,503)
    def test_request_without_ready_index_is_controlled(self):
        with patch('app.retrieval.index_is_current',return_value=False):
            r=TestClient(app).post('/ask',json={'question':'What is the model performance?'})
            self.assertEqual(r.status_code,503)
    def test_invalid_llm_payload_cannot_pass_through(self):
        from types import SimpleNamespace
        from app import llm
        fake=SimpleNamespace(messages=SimpleNamespace(create=lambda **kwargs:SimpleNamespace(content=[SimpleNamespace(type='text',text='{"explanation":"Accuracy is 54.9%."}')])))
        with patch('app.llm.get_client',return_value=fake):
            self.assertIn('could not validate',llm.ask('test','context'))

    def test_malformed_reply_is_retried_once_then_succeeds(self):
        from types import SimpleNamespace
        from app import llm
        key='dataset.total_records'
        replies=iter([
            '{"explanation":"Accuracy is 54.9%."}',
            '{"explanation":"There are {{'+key+'}} records."}',
        ])
        calls=[]
        def create(**kwargs):
            calls.append(kwargs['messages'][0]['content'])
            return SimpleNamespace(content=[SimpleNamespace(type='text',text=next(replies))])
        fake=SimpleNamespace(messages=SimpleNamespace(create=create))
        with patch('app.llm.get_client',return_value=fake):
            result=llm.ask('test','FACTS:\n{}')
        self.assertEqual(len(calls),2)
        self.assertIn('could not be validated',calls[1])
        self.assertIn(f'{load_facts()["dataset"]["total_records"]:g} records',result)

class IndexFreshnessTests(unittest.TestCase):
    def test_real_source_change_invalidates_manifest(self):
        import tempfile
        from app import index_state
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=root/'app';a.mkdir();k=root/'knowledge';k.mkdir();audit=root/'audit';audit.mkdir();idx=a/'index_store';idx.mkdir()
            for name in ['project_facts.json','domain.py','indexing.py','embeddings.py']:(a/name).write_text('{}')
            (root/'data.json').write_text('[]');(k/'rq1.md').write_text('old finding')
            paths={key:idx/name for key,name in [('FAISS_RECORDS_PATH','records.faiss'),('FAISS_KNOWLEDGE_PATH','knowledge.faiss'),('FAISS_AUDIT_PATH','audit.faiss'),('DOCUMENTS_PATH','documents.json')]}
            for p in paths.values():p.write_text('placeholder')
            # This checks freshness metadata, not FAISS parsing or semantic retrieval.
            with patch.multiple(config,BACKEND_DIR=root,APP_DIR=a,DATA_PATH=root/'data.json',KNOWLEDGE_DIR=k,FINAL_RESULTS_DIR=audit,**paths),patch.object(index_state,'MANIFEST_PATH',idx/'manifest.json'):
                self.assertFalse(index_state.index_is_current())
                (idx/'manifest.json').write_text(json.dumps({'source_sha256':index_state.source_fingerprint()}))
                self.assertTrue(index_state.index_is_current())
                (k/'rq1.md').write_text('corrected finding')
                self.assertFalse(index_state.index_is_current())
