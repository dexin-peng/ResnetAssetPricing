"""Compute Table 5 with the same theme removed from shallow and deep forecasts.

Run locally with: uv run --with numpy --with pandas --with scipy
python -m utils.paper_source.analysis.theme_refinement. Raw .pkl.gz forecasts are read unchanged.
Only private cell-month results are retained in local_inputs/theme_refinement;
completed 156-cell estimates are published to the lightweight source directory.
"""
import argparse
import os
for option in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[option]='1'
from pathlib import Path
import sys
import multiprocessing as mp
import numpy as np
import pandas as pd
from scipy.stats import rankdata,norm

REPORT_DIR=Path(__file__).resolve().parents[1]
ROOT=REPORT_DIR.parents[1]
OUT=REPORT_DIR/'local_inputs/theme_refinement'
SOURCE_DIR=REPORT_DIR/'data'
sys.path.insert(0,str(ROOT/'utils/paper_source'))
from utils.paper_source.analysis.pairwise_ordering import prepare_anchor,prepare_scores,pairwise_statistics_from_anchor
from utils.paper_source.analysis.forecast_refinement import equal_cell_estimate,hac_mean_se

MODELS={'ResNet':'resnets','NN':'nns','ResNet+':'resnetps','NN+':'nnps'}
METRICS=('P','C','U','V')
SCALES={'P':1,'C':100,'U':100,'V':10000}
INPUTS=ROOT/'utils/paper_source/local_inputs/conditional_content_inputs'


def load_predictions(prefix):
    columns=[]
    for theme in THEMES:
        slug=theme.lower().replace(' ','_').replace('-','_')
        path=ROOT/f'asset/{prefix}/econ/ALL/group_ablation/{slug}/stock_predictions.pkl.gz'
        frame=pd.read_pickle(path)
        assert list(frame.columns)==['PredRet','RealRet'] and frame.index.equals(MASK_INDEX),path
        assert frame.attrs['model']==prefix and frame.attrs['theme']==theme,path
        assert (frame.attrs['start_ensemble'],frame.attrs['max_ensemble'])==(0,9),path
        assert np.array_equal(frame.RealRet.to_numpy()[ORDER],RET),path
        values=frame.PredRet.to_numpy()[ORDER]
        assert np.isfinite(values).all(),path
        columns.append(values)
    return np.column_stack(columns)


def width_task(task):
    model,width=task;stem=MODELS[model]
    depths=ELIG.loc[ELIG.Model.eq(model)&ELIG['Width Schedule'].eq(width),'Depth'].tolist()
    cell_paths=[OUT/'cells'/f'{stem}{width}d{d}.csv.gz' for d in depths]
    if all(p.exists() for p in cell_paths):
        return model,width,len(depths),'reused'
    sa=load_predictions(f'{stem}{width}d{width}')
    prepared=[];q_sg=[]
    for j,row in BLOCKS.iterrows():
        start,end=int(row.Start),int(row.End);r=RET[start:end]
        rscore=prepare_scores(r)
        anchors=[];q=[]
        for k,theme in enumerate(THEMES):
            a=prepare_anchor(sa[start:end,k],rscore)
            qa=float(rankdata(a.shallow.values,method='average')@r)
            anchors.append(a);q.append(qa)
        prepared.append(anchors);q_sg.append(q)
    del sa
    for depth,path in zip(depths,cell_paths):
        if path.exists():continue
        da=load_predictions(f'{stem}{width}d{depth}')
        original=SOURCE.loc[(model,width,depth)].sort_index()
        assert original.index.tolist()==BLOCKS.DATE.tolist()
        records=[]
        for j,row in BLOCKS.iterrows():
            start,end=int(row.Start),int(row.End);r=RET[start:end];b=original.iloc[j]
            for k,theme in enumerate(THEMES):
                stats=pairwise_statistics_from_anchor(prepared[j][k],da[start:end,k])
                v=(float(rankdata(da[start:end,k],method='average')@r)-q_sg[j][k])/stats['total_pairs']
                raw={'P':b.P,'C':b.F,'U':b.total_contribution,'V':b.V}
                masked={'P':stats['P'],'C':np.nan if stats['F'] is None else stats['F'],
                        'U':stats['total_contribution'],'V':v}
                assert np.isclose(masked['U'],(1-masked['P'])*masked['C'],atol=1e-12) if stats['changed_pairs'] else masked['U']==0
                rec=dict(Model=model,Width=width,Depth=depth,DATE=row.DATE,Theme=theme,
                         ChangedPairs=stats['changed_pairs'],TotalPairs=stats['total_pairs'])
                for metric in METRICS:
                    rec['Original_'+metric]=raw[metric]
                    rec['Masked_'+metric]=masked[metric]
                    rec[metric]=raw[metric]-masked[metric]
                records.append(rec)
        pd.DataFrame(records).to_csv(path,index=False)
        print('calculated',model,width,depth,flush=True)
    return model,width,len(depths),'complete'


def estimate(series):
    mean,influence,cells=equal_cell_estimate(series)
    se=hac_mean_se(influence)
    t=mean/se if se else (0. if mean==0 else np.sign(mean)*np.inf)
    return dict(mean=mean,se=se,t=t,p=2*norm.sf(abs(t)),cells=cells,
                lo=mean-1.95996398454*se,hi=mean+1.95996398454*se)


def holm(series):
    p=series.to_numpy();o=np.argsort(p);a=np.empty(len(p))
    assert np.isfinite(p).all()
    a[o]=np.minimum(1,np.maximum.accumulate(p[o]*np.arange(len(p),0,-1)))
    return pd.Series(a,index=series.index)


def summarize(raw):
    rows=[]
    for (model,theme),group in raw.groupby(['Model','Theme']):
        for metric in METRICS:
            matrix=group.pivot(index='DATE',columns=['Width','Depth'],values=metric).reindex(BLOCKS.DATE)
            rows.append(dict(Scope='Within model',Model=model,Theme=theme,Metric=metric,**estimate(matrix.to_numpy())))
    summary=pd.DataFrame(rows)
    summary['holm_p']=summary.groupby(['Scope','Model','Metric']).p.transform(holm)
    summary['Scale']=summary.Metric.map(SCALES)
    for col in ('mean','se','lo','hi'):summary['display_'+col]=summary[col]*summary.Scale
    summary.to_csv(OUT/'summary.csv',index=False)
    raw.groupby(['Model','Theme'])[[f'{stage}_{m}' for stage in ['Original','Masked'] for m in METRICS]].mean().to_csv(OUT/'levels.csv')
    # Match original P/C/V against the published group summaries.
    base=raw[raw.Theme.eq(THEMES[0])].groupby('Model')[['Original_P','Original_C','Original_V']].mean()
    formal=pd.read_csv(ROOT/'utils/paper_source/data/pairwise_refinement/group_summary.csv')
    for model in MODELS:
        for metric,key in [('P','Preservation'),('C','Refinement'),('V','Value')]:
            expected= float(formal[(formal.Model==model)&(formal['Depth Group']=='Deep')&(formal.Metric==key)].Estimate.iloc[0])
            assert abs(base.loc[model,'Original_'+metric]*SCALES[metric]-expected)<1e-10
    result=summary[summary.Metric.isin(['P','C','V'])].drop(columns='Scope').copy()
    result=result.rename(columns={'cells':'Specs'})
    result['Months']=443
    result['AnchorCondition']='same_theme_zeroed'
    result['Unit']=result.Metric.map({'P':'proportion','C':'percent','V':'bps per month'})
    assert len(result)==156 and not result.duplicated(['Model','Theme','Metric']).any()
    assert np.isfinite(result[['mean','se','p','holm_p']].to_numpy()).all()
    path=SOURCE_DIR/'theme_pairwise_loss_summary.csv'
    temp=path.with_suffix('.csv.tmp')
    result.to_csv(temp,index=False)
    os.replace(temp,path)
    print('PASS: original levels match Table 4; wrote 156 paired-depth theme estimates',flush=True)
    print(result[result.Metric.eq('V')][['Model','Theme','display_mean','holm_p']].to_string(index=False),flush=True)


def main():
    global BLOCKS,RET,MASK_INDEX,ORDER,THEMES,ELIG,SOURCE
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'cells').mkdir(exist_ok=True)
    ref=pd.read_pickle(INPUTS/'reference_index.pkl')
    RET=np.asarray(np.load(INPUTS/'base_panel.npy',mmap_mode='r')[:,0])
    BLOCKS=pd.read_csv(INPUTS/'date_blocks.csv')
    sample=pd.read_pickle(ROOT/'asset/resnets1d1/econ/ALL/group_ablation/accruals/stock_predictions.pkl.gz')
    MASK_INDEX=sample.index
    assert not MASK_INDEX.has_duplicates and len(MASK_INDEX)==len(ref)
    ORDER=MASK_INDEX.get_indexer(ref)
    assert (ORDER>=0).all() and np.array_equal(sample.RealRet.to_numpy()[ORDER],RET)
    THEMES=sorted(pd.read_csv(ROOT/'source_data/cluster_labels.csv').cluster.unique())
    assert len(THEMES)==13
    source=pd.read_csv(ROOT/'utils/paper_source/data/pairwise_refinement/monthly.csv')
    source=source[source.Depth.between(11,20)]
    ELIG=source[source['Primary Eligible']][['Model','Width Schedule','Depth']].drop_duplicates()
    SOURCE=source.set_index(['Model','Width Schedule','Depth','DATE']).sort_index()
    ELIG.to_csv(OUT/'eligible_specs.csv',index=False)
    assert len(ELIG)==139
    with mp.get_context('fork').Pool(8) as pool:
        for result in pool.imap_unordered(width_task,[(m,w) for m in MODELS for w in range(1,5)]):
            print('width',result,flush=True)
    raw=pd.concat([pd.read_csv(OUT/'cells'/f'{MODELS[r.Model]}{int(r["Width Schedule"])}d{int(r.Depth)}.csv.gz') for _,r in ELIG.iterrows()],ignore_index=True)
    assert len(raw)==139*13*443 and not raw.duplicated(['Model','Width','Depth','Theme','DATE']).any()
    summarize(raw)


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
