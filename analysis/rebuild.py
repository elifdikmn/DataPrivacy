"""Run corrected notebooks and refresh all dependent facts, reports and PNGs.

Usage: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m analysis.rebuild
This does not call an LLM or build the expensive retrieval embeddings.
"""
import ast
import base64
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import platform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from .metrics import paired_intervals, wilson_interval

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'analysis/results'
CHARTS = ROOT/'backend/static/charts'
EXPORT = ROOT/'notebooks/exported_charts'
PLOTS = {
 1:{11:'rq1_sensitive_breakdown.png',12:'rq1_category_distribution.png',19:'rq2_description_rate.png'},
 2:{18:'rq3_model_comparison.png',20:'rq3_model_comparison_per_category.png',24:'rq3_feature_importance.png',27:'rq3_password_breakdown.png',34:'rq3_confusion_matrix.png'},
 3:{6:'rq4_silhouette_scores.png',11:'rq4_cluster_sensitivity.png',17:'rq5_other_confidence_distribution.png',25:'rq6_policy_disclosure.png',28:'rq7_sensitive_undisclosed.png'},
 4:{1:'rq3_validation_confidence_intervals.png'}}

def write_json(path, data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def execute_notebook(path, part):
    nb=json.loads(path.read_text());env={'__name__':'__main__'};count=0
    old_cwd=Path.cwd();os.chdir(ROOT/'notebooks')
    try:
        for i,cell in enumerate(nb['cells']):
            if cell['cell_type']!='code':continue
            count+=1; outputs=[];stream=io.StringIO()
            def show(*args, **kwargs):
                for num in plt.get_fignums():
                    fig=plt.figure(num);buf=io.BytesIO();fig.savefig(buf,format='png',dpi=140,bbox_inches='tight')
                    outputs.append({'output_type':'display_data','metadata':{},'data':{'image/png':base64.b64encode(buf.getvalue()).decode(),'text/plain':['<Matplotlib figure>']}})
                    if i in PLOTS[part]:
                        name=PLOTS[part][i];(CHARTS/name).write_bytes(buf.getvalue());(EXPORT/name).write_bytes(buf.getvalue())
                plt.close('all')
            plt.show=show
            tree=ast.parse(''.join(cell['source']))
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                if tree.body and isinstance(tree.body[-1],ast.Expr):
                    last=tree.body.pop();exec(compile(tree,str(path),'exec'),env)
                    value=eval(compile(ast.Expression(last.value),str(path),'eval'),env)
                    if value is not None:
                        outputs.append({'output_type':'execute_result','execution_count':count,'metadata':{},'data':{'text/plain':[repr(value)]}})
                else:exec(compile(tree,str(path),'exec'),env)
            if stream.getvalue():outputs.insert(0,{'output_type':'stream','name':'stdout','text':stream.getvalue().splitlines(keepends=True)})
            cell['outputs']=outputs;cell['execution_count']=count
            print(f'{path.name} cell {i}: passed',flush=True)
    finally:os.chdir(old_cwd)
    write_json(path,nb)
    return env,count

def make_validation_notebook():
    code='''import sys, json
from pathlib import Path
ROOT = Path.cwd() if (Path.cwd() / 'backend').exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
import pandas as pd
import matplotlib.pyplot as plt
from analysis.model_validation import run_validation

df = pd.DataFrame(json.loads((ROOT / 'backend/data/data_entries_final.json').read_text()))
validation_result, validation_reports = run_validation(df)
print(json.dumps(validation_result, indent=2))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for ax, metric, title in zip(axes, ['accuracy','macro_f1'], ['Accuracy','Macro F1']):
    for j,(name,result) in enumerate(validation_result['models'].items()):
        m=result[metric]; lo,hi=m['ci95'];point=m['estimate']
        ax.plot([lo,hi],[j,j],color='#286da8',linewidth=3)
        ax.scatter(point,j,color='#286da8',s=55)
        ax.text(.02,j+.20,f'{point:.3f}  [95% CI {lo:.3f}, {hi:.3f}]',fontsize=10)
    ax.set_yticks(range(len(validation_result['models'])), ['Baseline','Word + character / balanced'])
    ax.set_xlim(0,1);ax.set_ylim(-.45,1.55);ax.set_title(title)
    ax.set_xlabel('Score');ax.grid(axis='x',alpha=.2)
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('Fixed-test performance with conditional 95% confidence intervals')
fig.text(.02,.015,'Paired class-stratified bootstrap, 2,000 resamples. Excludes retraining and shared-Action dependence.',fontsize=9)
plt.tight_layout(rect=[0,.05,1,.94])
plt.show()
'''
    cells=[{'cell_type':'markdown','metadata':{},'source':['# Part 4 — Validation and 95% confidence intervals\n\n','Model selection uses only a validation subdivision of the original training set. The selected model is then fitted on the full training set and measured on the original test set.\n\n','Accuracy, macro F1, weighted F1 and paired improvements use 2,000 class-stratified bootstrap samples with the same sampled records for both models. Sensitive-category recall uses Wilson intervals. All intervals assume independent parameter records and condition on fixed fitted models; they do not include retraining or model-selection uncertainty. The original test set has already been inspected; external or Action-grouped validation is still needed.\n\n','A prediction probability for an individual record is not a confidence interval for model performance. No interval for correctness on unlabeled Other records can be computed without reviewed labels.\n\n','Method reference: [SciPy bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html). Stratification and metric computation are implemented explicitly in analysis/metrics.py.\n']},
    {'cell_type':'code','metadata':{},'source':code.splitlines(keepends=True),'execution_count':None,'outputs':[]}]
    for i,c in enumerate(cells):c['id']=f'validation-{i}'
    write_json(ROOT/'notebooks/bolum4_dogrulama.ipynb',{'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':cells})

def refresh_derived(envs, total_cells):
    e1,e2,e3,e4=[envs[i] for i in range(1,5)];v=e4['validation_result'];df=e1['df']
    fine=paired_intervals(e2['y_test_dt'], {'fine_category':e2['y_pred_dt']})
    other=paired_intervals(e3['y_test_k'], {'known_categories_only':e3['y_pred_k']})
    embed=paired_intervals(e2['y_test'], {'baseline':e2['y_pred_baseline'],'spacy_vectors':e2['y_pred_embedding']})
    write_json(RESULTS/'model_validation.json',v);write_json(RESULTS/'fine_category_intervals.json',fine)
    write_json(RESULTS/'known_only_intervals.json',other);write_json(RESULTS/'embedding_intervals.json',embed)
    for name,report in e4['validation_reports'].items():pd.DataFrame(report).T.to_csv(RESULTS/f'{name}_report.csv')
    facts=json.loads((ROOT/'backend/app/project_facts.json').read_text())
    facts['_meta']['description']='Generated from the current notebook execution by python -m analysis.rebuild. Numeric confidence intervals are conditional estimates; consult uncertainty notes.'
    facts['_meta']['dataset_sha256']=hashlib.sha256((ROOT/'backend/data/data_entries_final.json').read_bytes()).hexdigest()
    facts['_meta']['environment']={'python':platform.python_version(),'sklearn':sklearn.__version__}
    # Keep original baseline metrics explicitly named, with the validated improvement alongside.
    model=facts['category_prediction_model']
    model['tfidf_baseline']={f'{metric}_pct':round(float(val)*100,1) for metric,val in [('accuracy',e2['acc_baseline']),('macro_f1',e2['f1_macro_baseline']),('weighted_f1',e2['f1_weighted_baseline'])]}
    model['embedding_model']={'accuracy_pct':round(e2['acc_embedding']*100,1),'macro_f1_pct':round(e2['f1_macro_embedding']*100,1)}
    model['uncertainty']=v
    model['embedding_uncertainty']=embed['models']['spacy_vectors']
    model['data_type_model_79_classes']['uncertainty']=fine['models']['fine_category']
    # These category rows refer explicitly to the original baseline.
    for cat in e3['SENSITIVE_CATEGORIES']:
        r=e2['report_baseline'][cat]
        model['per_class_metrics_sensitive_categories'][cat]={k:round(r[src],3) for k,src in [('precision','precision'),('recall','recall'),('f1','f1-score')]}
    pwd=df[df.data_type=='Password']
    facts['password_records']['total_plugin_instances']=sum(len(set(ids)) for ids in pwd.plugin_id_filenames)
    facts['password_records']['unique_plugins']=len(set(x for ids in pwd.plugin_id_filenames for x in ids))
    facts['password_records']['note']='total_plugin_instances counts unique parameter–Action pairs, while unique_plugins counts distinct Action IDs across all Password records. Record counts are not runtime transmission counts.'
    # Clusters: derive IDs, counts, shares and rankings from corrected profiles only.
    cent=e3['centroids'];sizes=e3['cluster_sizes'];shares=e3['sensitive_share']
    cluster_rows={str(c):{'n_plugins':int(sizes[c]),'sensitive_pct':round(float(shares[c])*100,1),'dominant_categories':cent.loc[c].sort_values(ascending=False).head(3).index.tolist()} for c in cent.index}
    eligible=len(e3['plugin_profile'])
    def ranking(ids):
        return {'clusters':[str(x) for x in ids],'combined_plugins':int(sizes.loc[ids].sum()),'combined_pct_of_eligible':round(sizes.loc[ids].sum()/eligible*100,1)}
    facts['clustering']={'best_k':int(e3['best_k']),'silhouette_score':round(e3['best_score'],4),
        'total_eligible_plugins':eligible,'min_params_threshold':3,'unique_parameter_action_pairs':len(e3['df_exploded']),
        'cluster_summary':cluster_rows,'highest_sensitive_share_clusters':ranking(shares.nlargest(2).index),
        'largest_by_size_clusters':ranking(sizes.nlargest(2).index),
        'note':'Distinct parameter–Action pairs. K chosen only among 2 through 10. Cluster IDs are arbitrary. Selected-category share is not a validated risk score.'}
    e3['cluster_summary'].to_csv(RESULTS/'corrected_clusters.csv')
    reclass=facts['other_reclassification'];reclass['uncertainty_known_label_test']=other['models']['known_categories_only']
    reclass['sensitive_flagged_in_confident_subset']=int(e3['n_sensitive_flagged'])
    reclass['unresolved_broad_mappings']=int(e3['confident_other']['predicted_main_type'].isna().sum())
    reclass['note']='Threshold predictions and sensitive flags are unverified. Known-label confidence intervals do not estimate correctness on unlabeled Other records.'
    audit=e3['status_counts'];n=sum(audit.values())
    facts['privacy_policy_audit'].update({'total_parameters_with_comparable_policy_text':n})
    for status,key in [('UNDISCLOSED','undisclosed'),('DISCLOSED_CLEAR','disclosed_clear'),('DISCLOSED_VAGUE','disclosed_vague'),('DISCLOSED_AMBIGUOUS','disclosed_ambiguous'),('DISCLOSED_INCORRECT','disclosed_incorrect')]:
        facts['privacy_policy_audit'][key+'_count']=audit[status];facts['privacy_policy_audit'][key+'_pct']=round(audit[status]/n*100,1)
    # RQ7: intersection of RQ1's sensitive categories with RQ6's disclosure audit.
    n_sens=int(e3['n_sensitive']);n_undisc_sens=int(e3['n_undisclosed_sensitive'])
    facts['sensitive_undisclosed_intersection']={
        'note':"Hand mapping of this audit's own data_type labels onto the main dataset's 4 sensitive main_data_type categories (Passwords, Email address, Name, Phone number, Other financial info, Purchase history only — ambiguous labels like Address, and the fact this audit has no Health-information label at all, are deliberately excluded rather than guessed). Small sample (n=20); see notebooks/bolum3_uygulama.ipynb RQ7 appendix for the mapping and its justification.",
        'total_parameters_with_comparable_policy_text':n,
        'sensitive_count':n_sens,'undisclosed_count':n_undisc_sens,
        'undisclosed_pct_of_sensitive':round(n_undisc_sens/n_sens*100,1),
        'undisclosed_pct_of_all_audited':round(n_undisc_sens/n*100,1),
        'ci95_wilson_of_sensitive':wilson_interval(n_undisc_sens,n_sens),
        'ci95_wilson_of_all_audited':wilson_interval(n_undisc_sens,n),
        'category_counts':{k:int(v) for k,v in e3['sensitive_audit']['main_type'].value_counts().items()},
    }
    # Independent-record description-rate uncertainty; binary bootstrap equals resampling each binary group.
    sensitive=df.main_data_type.isin(e3['SENSITIVE_CATEGORIES']);has=df.description.fillna('').str.strip().ne('')
    rng=np.random.default_rng(20260915);rates={};samples={}
    for label,mask in [('selected_sensitive',sensitive),('other_categories',~sensitive)]:
        total=int(mask.sum());yes=int(has[mask].sum());rates[label]={'estimate':yes/total,'ci95_wilson':wilson_interval(yes,total),'n':total}
        samples[label]=rng.binomial(total,yes/total,2000)/total
    rates['difference_sensitive_minus_other']={'estimate':rates['selected_sensitive']['estimate']-rates['other_categories']['estimate'],
        'ci95':np.quantile(samples['selected_sensitive']-samples['other_categories'],[.025,.975]).tolist()}
    rates['note']='Wilson intervals for group rates; percentile bootstrap for their difference. Assumes independent parameter records, not a causal or population-wide conclusion.'
    facts['description_rate_test']['uncertainty']=rates
    write_json(ROOT/'backend/app/project_facts.json',facts)
    # Knowledge texts: generated evidence replaces stale numerical narratives.
    selected=v['selected_model'];lines=['# Category prediction and uncertainty','\n## Measured results']
    for name,res in v['models'].items():
        lines.append(f'\n{name}:')
        for metric,m in res.items():lines.append(f"- {metric}: {m['estimate']:.4f}; conditional 95% CI [{m['ci95'][0]:.4f}, {m['ci95'][1]:.4f}].")
    lines+=['\n## Method and limitations',f'Selected model: {selected}. Validation-only selection among three candidates. '+v['limitations'],v['selection_note'],
        '\n## Baseline and fine categories',f"spaCy vectors: accuracy {e2['acc_embedding']:.4f}, macro F1 {e2['f1_macro_embedding']:.4f}. This is not a modern sentence-embedding benchmark.",
        f"Fine-category baseline: accuracy {e2['acc_dt']:.4f}, macro F1 {e2['f1_macro_dt']:.4f}.",
        'Confusion-matrix rows use all true records in each class as the denominator. Half of all true Finance and Health records, not half their errors, were predicted as Other by the original baseline.',
        'Per-class support means test examples. Coefficients do not establish model reliability. See FACTS for exact model-specific values and intervals.']
    (ROOT/'backend/knowledge/rq3_model_performansi.md').write_text('\n'.join(lines)+'\n')
    rows=['# Distinct-parameter Action profiles','\n## Corrected sample',f'{len(e3["df_exploded"])} distinct parameter–Action pairs; {eligible} Actions have at least three distinct parameter records.',
          '\n## Cluster results',f'Best K within 2–10: {e3["best_k"]}; silhouette {e3["best_score"]:.4f}.','|Cluster|Actions|Selected sensitive-category share|Top categories|','|---|---:|---:|---|']
    for c,r in cluster_rows.items():rows.append(f'|{c}|{r["n_plugins"]}|{r["sensitive_pct"]}%|{", ".join(r["dominant_categories"])}|')
    rows+=['\n## Interpretation','These are functional category profiles, not safe/unsafe labels. The four selected categories omit other potentially sensitive data, including location and messages. Cluster IDs changed after deduplication and cannot be matched by number to older results. K is exploratory; small clusters and the upper search boundary require caution. Rankings in FACTS distinguish cluster size from selected-category share.']
    (ROOT/'backend/knowledge/rq4_kumeleme.md').write_text('\n'.join(rows)+'\n')
    (ROOT/'backend/knowledge/rq5_other_siniflandirma.md').write_text(f'''# Review prioritization for Other records

## Results
The known-label-only classifier was applied to {len(e3['df_other'])} records originally labeled Other at the fine-category level. {len(e3['confident_other'])} exceed the uncalibrated prediction-score threshold of 0.5; {e3['n_sensitive_flagged']} map unambiguously to a selected sensitive category. {reclass['unresolved_broad_mappings']} above-threshold predictions have unresolved fine-to-broad mappings.

## Interpretation
These are unverified review candidates, not confirmed hidden sensitive records. For example, skills → API key and email_type → Email address may be false positives. A high probability does not prove correctness, and a low probability does not prove the original Other label is appropriate. Human labels are needed to measure precision on this population. Known-label test confidence intervals in FACTS do not transfer to unlabeled Other records. Use suggestions for review prioritization, not automatic relabeling.
''')
    # Replace notebook summary with generated corrected evidence.
    path=ROOT/'notebooks/bolum3_uygulama.ipynb';nb=json.loads(path.read_text());nb['cells'][31]['source']=('\n'.join(rows)+'\n\nAbove-threshold flags remain unverified; see Part 4 for confidence intervals.\n').splitlines(keepends=True);write_json(path,nb)
    write_json(RESULTS/'execution_manifest.json',{'executed_code_cells':total_cells,'dataset_sha256':facts['_meta']['dataset_sha256'],'environment':facts['_meta']['environment'],'retrieval_index':'Rebuild required after this command: python -m app.indexing, then restart backend.'})
    # Compact results report, including paired intervals rather than overlap heuristics.
    report=['# Corrected analysis results','\n|Model|Accuracy (95% CI)|Macro F1 (95% CI)|','|---|---|---|']
    for name,r in v['models'].items():
        a,f=r['accuracy'],r['macro_f1'];report.append(f"|{name}|{a['estimate']:.2%} [{a['ci95'][0]:.2%}, {a['ci95'][1]:.2%}]|{f['estimate']:.4f} [{f['ci95'][0]:.4f}, {f['ci95'][1]:.4f}]|")
    report+=['\n## Paired improvement intervals',json.dumps(v['paired_differences'],indent=2),'\n## Interpretation',v['limitations'],v['selection_note'],
             '\nPrediction scores are not confidence intervals. Unlabeled Other precision cannot be estimated without reviewed labels.',
             '\n## Corrected clustering',f'{eligible} eligible Actions; {len(e3["df_exploded"])} distinct parameter–Action pairs.',
             '\nMethod: [SciPy bootstrap reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html). The implementation uses explicit class-stratified paired resampling.',
             '\nAdditional JSON reports contain fine-category and known-label-only intervals, plus Wilson intervals for sensitive-category recall.']
    (RESULTS/'RESULTS.md').write_text('\n'.join(report)+'\n')


def main():
    for p in [RESULTS,CHARTS,EXPORT]:p.mkdir(exist_ok=True,parents=True)
    # Source changes invalidate retrieval; leave no manifest claiming current indices.
    (ROOT/'backend/app/index_store/manifest.json').unlink(missing_ok=True)
    make_validation_notebook();envs={};total=0
    for part in range(1,5):
        path=next((ROOT/'notebooks').glob(f'bolum{part}_*.ipynb'))
        envs[part],count=execute_notebook(path,part);total+=count
    refresh_derived(envs,total)
    print(f'COMPLETE: {total} code cells. Rebuild retrieval indices before live RAG use.',flush=True)

if __name__=='__main__':main()
