"""Validation-only model selection, followed by fixed-test paired uncertainty."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from .metrics import paired_intervals, wilson_interval


def build_model(name):
    word = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
    features = word if name != 'word_char_balanced' else FeatureUnion([
        ('word', word), ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=2, max_features=20000))])
    return make_pipeline(features, LogisticRegression(max_iter=1000, random_state=42,
                         class_weight=None if name=='baseline' else 'balanced'))


def run_validation(df):
    X = (df.name.fillna('') + '. ' + df.description.fillna('')).str.strip()
    y = df.main_data_type
    train, test = train_test_split(df.index, test_size=.2, random_state=42, stratify=y)
    fit, val = train_test_split(train, test_size=.2, random_state=43, stratify=y.loc[train])
    validation = {}
    for name in ['baseline', 'word_balanced', 'word_char_balanced']:
        pred = build_model(name).fit(X.loc[fit], y.loc[fit]).predict(X.loc[val])
        validation[name] = {'accuracy': accuracy_score(y.loc[val],pred), 'macro_f1': f1_score(y.loc[val],pred,average='macro')}
    selected = max(validation, key=lambda n: validation[n]['macro_f1'])
    predictions, reports = {}, {}
    for name in dict.fromkeys(['baseline', selected]):
        pred = build_model(name).fit(X.loc[train], y.loc[train]).predict(X.loc[test])
        predictions[name] = pred
        reports[name] = classification_report(y.loc[test],pred,output_dict=True,zero_division=0)
    result = paired_intervals(y.loc[test], predictions)
    result.update({'selected_model':selected, 'validation':validation,
                   'split_sizes': {'fit':len(fit),'validation':len(val),'train':len(train),'test':len(test)},
                   'selection_note':'Three candidates selected by validation macro F1. The original test set has already been inspected historically; these are exploratory estimates, not a new external evaluation.'})
    recalls = {}
    for name,pred in predictions.items():
        recalls[name] = {}
        for label in ['Security credentials','Personal information','Health information','Finance information']:
            mask = y.loc[test].to_numpy()==label
            successes = int((pred[mask]==label).sum());n=int(mask.sum())
            recalls[name][label] = {'estimate':successes/n,'correct':successes,'support':n,'ci95':wilson_interval(successes,n)}
    result['sensitive_recall_wilson'] = recalls
    normalized = X.str.lower().str.replace(r'\s+',' ',regex=True)
    conflicts = df.assign(normalized_text=normalized).groupby('normalized_text').main_data_type.nunique()
    tr_actions = set(v for row in df.loc[train,'plugin_id_filenames'] for v in row)
    te_actions = set(v for row in df.loc[test,'plugin_id_filenames'] for v in row)
    result['data_quality'] = {'duplicate_normalized_inputs':int(normalized.duplicated().sum()),
        'test_inputs_in_training':int(normalized.loc[test].isin(set(normalized.loc[train])).sum()),
        'conflicting_input_groups':int((conflicts>1).sum()),'test_action_ids':len(te_actions),
        'test_action_ids_in_training':len(te_actions & tr_actions)}
    return result, reports
