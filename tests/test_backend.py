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
        answer=render_grounded_answer({'explanation':'The word-vector result is shown below.','fact_ids':[key]})
        self.assertIn('embedding model / accuracy pct:',answer)
        self.assertNotIn('tfidf baseline',answer)
        with self.assertRaises(GroundingError):render_grounded_answer({'explanation':'TF-IDF accuracy is 54.9%.','fact_ids':[]})
        with self.assertRaises(GroundingError):render_grounded_answer({'explanation':'A result.','fact_ids':['invented.accuracy']})
    def test_number_format_diagnostic(self):
        self.assertEqual(verify_answer_numbers('There are 12,811 records.'),[])
        self.assertEqual(verify_answer_numbers('There are 12811.0 records.'),[])
        # Fractions written as percentages, the CI level and Turkish decimal commas are known values.
        self.assertEqual(verify_answer_numbers('Doğruluk %76.20; %95 güven aralığı %74.56–%77.80.'),[])
        self.assertEqual(verify_answer_numbers('%90; %95 güven aralığı %69,9–%97,2; 12.811 kayıt.'),[])
        self.assertEqual(verify_answer_numbers('Accuracy is 54.9% and 99.9%.'),['54.9','99.9'])
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
class ConversationTests(unittest.TestCase):
    def fake_client(self, texts):
        from types import SimpleNamespace
        from unittest.mock import Mock
        response=SimpleNamespace(content=[SimpleNamespace(type='text',text=t) for t in texts],stop_reason='end_turn')
        return SimpleNamespace(messages=SimpleNamespace(create=Mock(return_value=response)))

    def test_natural_answers_including_f1_and_confidence_intervals(self):
        from app import llm
        for answer in ['Merhaba! Projende nasıl yardımcı olabilirim?',
                       'Macro F1, sınıfların F1 skorlarının ortalamasıdır.',
                       'Doğruluk %76.20; %95 güven aralığı %74.56–%77.80.']:
            with self.subTest(answer=answer), patch('app.llm.get_client',return_value=self.fake_client([answer])):
                self.assertEqual(llm.ask('Projemi açıklar mısın?','context'),answer)

    def test_unmatched_numbers_only_log(self):
        from app import llm
        answer='Örnek olarak 987654321 ele alalım.'
        with patch('app.llm.get_client',return_value=self.fake_client([answer])), self.assertLogs('chatbot.llm',level='WARNING'):
            self.assertEqual(llm.ask('Bir örnek ver','context'),answer)

    def test_multiple_text_blocks(self):
        from app import llm
        with patch('app.llm.get_client',return_value=self.fake_client(['Merhaba.','Nasıl yardımcı olabilirim?'])):
            self.assertEqual(llm.ask('Merhaba','context'),'Merhaba.\nNasıl yardımcı olabilirim?')

    def test_legacy_json_does_not_discard_numeric_explanation(self):
        from app import llm
        for ids in [[],['invented.accuracy']]:
            payload=json.dumps({'explanation':'Accuracy is 54.9%', 'fact_ids':ids})
            for text in [payload,'```json\n'+payload+'\n```']:
                with patch('app.llm.get_client',return_value=self.fake_client([text])):
                    self.assertEqual(llm.ask('test','context'),'Accuracy is 54.9%')

    def test_legacy_valid_fact_retains_explanation(self):
        from app import llm
        payload=json.dumps({'explanation':'F1 hakkında sonuç:', 'fact_ids':['category_prediction_model.embedding_model.accuracy_pct']})
        with patch('app.llm.get_client',return_value=self.fake_client([payload])):
            answer=llm.ask('test','context')
            self.assertIn('F1 hakkında sonuç:',answer)
            self.assertIn('embedding model / accuracy pct:',answer)

    def test_diagnostic_failure_does_not_break_conversation(self):
        from app import llm
        with patch('app.llm.get_client',return_value=self.fake_client(['Merhaba.'])), patch('app.llm.verify_answer_numbers',side_effect=ValueError('diagnostic')), self.assertLogs('chatbot.llm',level='WARNING'):
            self.assertEqual(llm.ask('Merhaba','context'),'Merhaba.')

    def test_empty_provider_response_is_controlled(self):
        with patch('app.rag.search',return_value=[]), patch('app.llm.get_client',return_value=self.fake_client([])):
            response=TestClient(app).post('/ask',json={'question':'Merhaba'})
            self.assertEqual(response.status_code,503)
            self.assertIn('empty response',response.json()['detail'])

    def test_http_natural_answer_and_facts_context(self):
        fake=self.fake_client(['Merhaba! F1 ve %95 güven aralığını açıklayabilirim.'])
        with patch('app.rag.search',return_value=[]), patch('app.llm.get_client',return_value=fake):
            response=TestClient(app).post('/ask',json={'question':'Merhaba'})
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json()['answer'],'Merhaba! F1 ve %95 güven aralığını açıklayabilirim.')
        self.assertEqual(response.json()['sources'],[])
        self.assertIn('chart_image',response.json())
        kwargs=fake.messages.create.call_args.kwargs
        system=' '.join(block['text'] for block in kwargs['system'])
        self.assertIn('ordinary text, not JSON',system)
        self.assertIn('FACTS (reference data',system)
        self.assertIn('ci95',system)
        # The stable prefix (instructions + FACTS) is marked for prompt caching.
        self.assertEqual(kwargs['system'][-1]['cache_control'],{'type':'ephemeral'})
        self.assertIn('QUESTION: Merhaba',kwargs['messages'][-1]['content'])

    def test_rq7_facts_and_intervals_reach_provider(self):
        from app.facts import format_facts_block
        from app.rag import build_context
        facts=load_facts()
        rq7=facts['sensitive_undisclosed_intersection']
        block=format_facts_block()
        self.assertEqual(json.loads(block.split('\n',1)[1])['sensitive_undisclosed_intersection'],rq7)
        expected='Hassas parametrelerin %90’ı açıklanmamış; %95 güven aralığı %69,9–%97,2.'
        fake=self.fake_client([expected])
        with patch('app.rag.search',return_value=[]), patch('app.llm.get_client',return_value=fake):
            response=TestClient(app).post('/ask',json={'question':'RQ7 sonucunu güven aralığıyla açıkla'})
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json()['answer'],expected)
        sent=' '.join(b['text'] for b in fake.messages.create.call_args.kwargs['system'])
        self.assertIn('sensitive_undisclosed_intersection',sent)
        self.assertIn(block,sent)
        self.assertEqual(fake.messages.create.call_count,1)


class AnswerStyleTests(unittest.TestCase):
    def test_prompt_targets_short_answers_with_bold_key_terms(self):
        from app.llm import SYSTEM_PROMPT
        self.assertIn('non-specialists',SYSTEM_PROMPT)
        self.assertIn('two or three sentences',SYSTEM_PROMPT)
        self.assertIn('Answer only the question that was asked',SYSTEM_PROMPT)
        self.assertIn('**bold**',SYSTEM_PROMPT)

    def test_markdown_tidying_keeps_bold_and_leaves_text_intact(self):
        from app.llm import tidy_markdown
        self.assertEqual(tidy_markdown('## Sonuç\nModel **%76.2** doğru.'),'**Sonuç**\nModel **%76.2** doğru.')
        self.assertEqual(tidy_markdown('* a\n+ b\n- c'),'- a\n- b\n- c')
        self.assertEqual(tidy_markdown('***önemli***'),'**önemli**')
        self.assertEqual(tidy_markdown('slightly *less* likely and _rarely_ **bold**'),'slightly less likely and rarely **bold**')
        for text in ['__init__ ve api_key','2 * 3 * 4 = 24','1. adım\n2. adım','- madde *','snake_case_name']:
            self.assertEqual(tidy_markdown(text),text)

    def test_history_becomes_alternating_messages(self):
        from app.llm import history_messages
        history=[{'role':'assistant','text':'Merhaba!'},{'role':'user','text':'Model ne kadar doğru?'},
                 {'role':'assistant','text':'**%76.2**'},{'role':'user','text':'başarısız istek'},{'role':'user','text':'  '}]
        self.assertEqual(history_messages(history),[{'role':'user','content':'Model ne kadar doğru?'},{'role':'assistant','content':'**%76.2**'}])
        self.assertEqual(history_messages(None),[])

    def test_http_history_reaches_provider_and_retrieval(self):
        fake=ConversationTests.fake_client(None,['**%74.6–%77.8** aralığında.'])
        history=[{'role':'user','text':'Model ne kadar doğru?'},{'role':'assistant','text':'Doğruluk **%76.2**.'}]
        with patch('app.rag.search',return_value=[]) as search, patch('app.llm.get_client',return_value=fake):
            response=TestClient(app).post('/ask',json={'question':'Peki güven aralığı?','history':history})
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json()['answer'],'**%74.6–%77.8** aralığında.')
        messages=fake.messages.create.call_args.kwargs['messages']
        self.assertEqual([m['role'] for m in messages],['user','assistant','user'])
        self.assertIn('Peki güven aralığı?',messages[-1]['content'])
        self.assertIn('Model ne kadar doğru?',search.call_args.args[0])
        bad=TestClient(app).post('/ask',json={'question':'x','history':[{'role':'system','text':'y'}]})
        self.assertEqual(bad.status_code,422)

    def test_chart_mapping_tolerates_formatting_and_every_chart_exists(self):
        from app.chart_mapping import QUESTION_CHART_MAP,get_chart_filename
        self.assertEqual(get_chart_filename('  What percentage of collected data is SENSITIVE  '),'rq1_category_distribution.png')
        self.assertEqual(get_chart_filename('Can mislabeled “Other” records be identified automatically'),'rq5_other_confidence_distribution.png')
        self.assertEqual(get_chart_filename('Are sensitive parameters also the undisclosed ones?'),'rq7_sensitive_undisclosed.png')
        self.assertIsNone(get_chart_filename('Tell me a joke'))
        for chart in QUESTION_CHART_MAP.values():self.assertTrue((config.STATIC_DIR/'charts'/chart).is_file(),chart)

    def test_frontend_suggestions_match_chart_questions(self):
        import re
        from app.chart_mapping import QUESTION_CHART_MAP
        source=(Path(__file__).resolve().parents[1]/'frontend/src/App.js').read_text(encoding='utf-8')
        block=source.split('const SUGGESTED_QUESTIONS = [',1)[1].split('];',1)[0]
        chips=[q.replace("\\'","'") for q in re.findall(r"'((?:[^'\\]|\\.)*)'",block)]
        self.assertEqual(sorted(c.lower() for c in chips),sorted(QUESTION_CHART_MAP))

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
