"""DataFrame tables and Matplotlib figures for the replication companion."""
from __future__ import annotations
import importlib
import re
import numpy as np
import pandas as pd
from .artifact_cache import Estimate, load_cache, write_cache
from .artifact_registry import ARTIFACTS

DEPTHS=['Shallow','Medium','Deep']
MODELS=['NN','ResNet','NN+','ResNet+']
TITLES={
 'ExperimentalDesignGrid':'Empirical design',
 'FinalBaselinePerformance':'Long-short portfolio performance',
 'DepthGroupSharpeContrasts':'Deep minus shallow Sharpe ratios',
 'FullDecileMonotonicity':'Decile portfolio performance',
 'ForecastRefinement':'Forecast preservation and refinement',
 'ThemePairwiseLosses':'Forecast preservation and revision losses by theme',
 'TransactionCostAdjustedPerformance':'Performance after transaction costs',
 'SizeExclusionRobustness':'Robustness to size exclusions',
 'EqualWeightedRobustness':'Equal-weighted portfolio robustness',
 'MarketStateRobustness':'Performance across market states',
 'PredictorList':'Predictors for one-month-ahead excess returns',
 'S1D7DiagnosticSummary':'Forecast collapse diagnostics',
 'AppendixForecastRankingValues':'Shallow and deep pairwise ranking values',
 'AppendixThemeValueLosses':'Shallow and deep ranking-value losses by theme',
 'PairedSpecificationDepthProfile':'NN+ and ResNet+ by width and depth',
 'ResNetVsNNDepthProfile':'NN and ResNet by width and depth',
 'RollingTrainTestSplit':'Annually expanding training window',
 'SpecificationDeltaSRHeatmaps':'Residual minus feedforward Sharpe ratios',
 'DecileMonotonicitySelected':'Forecast-decile return and volatility profiles',
 'ForecastRankPreservationAndRefinementByDepth':'Preservation and revision accuracy by depth',
 'ParameterScaleGap':'Network parameter counts'}
NOTES={
 'FinalBaselinePerformance':'Mean, SD, FF5 alpha and turnover are monthly percentages; SR is annualized. Model levels average available noncollapsed specifications. Delta SR uses matched specifications, so it need not equal the difference of displayed model-family means. FF5 stars use Newey–West (12 lags); delta SR stars use a studentized circular block bootstrap (12-month blocks, 1,000 draws).',
 'DepthGroupSharpeContrasts':'Within each model, equal-weighted mean deep SR minus mean shallow SR. Available specifications differ by model. Two-sided studentized circular block bootstrap, 12-month blocks, 1,000 draws.',
 'FullDecileMonotonicity':'Fcst, mean and SD are monthly percentages; SR is annualized. Statistics are computed per specification before equal-weighted depth aggregation. Each model uses its own completed nonconstant-forecast specifications, matching its baseline performance sample.',
 'DecileMonotonicitySelected':'Monthly mean returns and volatility are in percent; SR is annualized. Statistics are computed per specification, then averaged equally across completed specifications shared by both models in each comparison. Constant forecasts enter this descriptive figure using seeded tie splitting.',
 'ForecastRefinement':'P is the unchanged-pair share; C is revision accuracy among changed pairs, in percent; V is revision economic value in monthly basis points, not a portfolio return. Average each specification over months, then equally across specifications. Difference is residual minus feedforward; its eligible specification sets can differ. Newey–West (12 lags); no stars on model-level P.',
 'ThemePairwiseLosses':'Original minus theme-zeroed forecasts, removing the same theme from shallow and deep models. Delta P is unscaled; delta C is in percentage points; delta V is in monthly basis points. Stars use Holm-adjusted p-values across 13 themes. These are input-sensitivity comparisons.',
 'AppendixThemeValueLosses':'Delta P compares unchanged-pair shares over all stock pairs. Delta C compares revision accuracy calculated on each original or theme-zeroed forecast comparison\'s own changed pairs. Delta V(S) and delta V(D) are original minus theme-zeroed shallow and deep ranking values over all pairs, in monthly basis points. Their difference reconciles to the main theme revision-value loss. Stars use Holm adjustment.',
 'AppendixForecastRankingValues':'P is unscaled; C is in percent; shallow and deep ranking values are monthly basis points over all pairs. Deep minus shallow ranking value equals revision economic value. Newey–West standard errors use 12 lags.',
 'TransactionCostAdjustedPerformance':'Annualized net Sharpe ratios at one-way costs of 0, 10 and 50 bps. Monthly gross long-short turnover is in percent. Comparisons use matched noncollapsed specifications; bootstrap stars test residual minus feedforward SR.',
 'SizeExclusionRobustness':'Annualized value-weighted long-short SR; exclude the smallest or largest 20 percent by market capitalization. Each comparison uses its matched noncollapsed specifications. Stars use the same 12-month-block Sharpe bootstrap.',
 'EqualWeightedRobustness':'Matched equal-weighted long-short portfolios. SR is annualized; alpha, R2 and turnover differences are in percentage points. SR stars use the block bootstrap.',
 'MarketStateRobustness':'Monthly value-weighted long-short returns, in percent, averaged across matched specifications before time aggregation. Newey–West inference is omitted for intervals shorter than 13 months. Periods use realized-return months.',
 'PredictorList':'153 firm characteristics and eight macro predictors. Firm characteristics are median-imputed by month and rank-normalized; macro predictors enter in levels.',
 'S1D7DiagnosticSummary':'NN and ResNet at width schedule 1, depths 3 and 7. Constant-forecast months are a ranking failure diagnostic.',
 'PairedSpecificationDepthProfile':'Unrounded values are available through table(..., raw=True). Return, alpha, drawdown and turnover measures are percentages; SR is annualized. This reference snapshot retains historical FF5 alphas. New evaluations use realized-return-month alignment and no extra RF subtraction; recompute alphas before comparing runs. Missing specifications are not imputed.',
 'ResNetVsNNDepthProfile':'Constant forecasts are excluded; missing specifications are not imputed. This reference snapshot retains historical FF5 alphas. New evaluations use realized-return-month alignment and no extra RF subtraction; recompute alphas before comparing runs.'}

def clean_label(value):
    if not isinstance(value,str): return value
    text=value.replace(r'\$5', 'USD 5').replace(r'\%', '%').replace(r'\Delta','Δ').replace(r'\alpha','α').replace(r'\ell','ℓ').replace(r'\times','×').replace(r'\rightarrow','→')
    text=re.sub(r'\\(?:mathrm|text|mbox|ensuremath)\{([^{}]*)\}',r'\1',text)
    text=text.replace('$','').replace('{','').replace('}','').replace('\\','')
    return text.replace('--','–')

def available():
    return pd.DataFrame([{'Exhibit':name,'Kind':'figure' if record[4]=='figure' else 'table','Title':TITLES[name]} for name,record in ARTIFACTS.items()])

def _static(name):
    shared=[('Data','Target','One-month-ahead stock excess return'),('Data','Predictors','153 JKP characteristics + 8 macro predictors'),('Data','Sample','Jan 1963–Dec 2023'),('Data','Universe','U.S. common stocks'),('Data','Forecast origins','Jan 1987–Nov 2023'),('Data','Realized returns','Feb 1987–Dec 2023 (443 months)'),('Training','Window','Annually expanding'),('Training','Loss','Mean squared error'),('Training','Ensemble','10 seeds (0–9), average forecasts'),('Variation','Widths','[32]; [32,16]; [32,16,8]; [32,16,8,4]'),('Variation','Depth','s through 20 for width schedule s')]
    rows=[(a,b,c,c) for a,b,c in shared]+[('Model','Feedforward','NN','NN+'),('Model','Residual','ResNet','ResNet+'),('Model','Shortcut','Identity','Identity / projection')]
    return pd.DataFrame(rows,columns=['Group','Component','ResNet vs NN','ResNet+ vs NN+']).set_index(['Group','Component'])

def _stars(t):
    if pd.isna(t): return ''
    return '***' if abs(t)>=2.576 else '**' if abs(t)>=1.96 else '*' if abs(t)>=1.645 else ''

def _pstars(p):
    return '***' if p<.01 else '**' if p<.05 else '*' if p<.1 else ''

def table(name,*,raw=False,model=None):
    """Return a fresh DataFrame; raw=True exposes all numerical source columns."""
    if name not in ARTIFACTS: raise ValueError(f'Unknown exhibit: {name}')
    if ARTIFACTS[name][4]=='figure': raise ValueError(f'{name} is a figure')
    if ARTIFACTS[name][4]=='static': return _static(name)
    payload=load_cache(name)
    if raw:
        if 'table' in payload: frame = payload['table'].copy(deep=True)
        elif 'rows' in payload: frame = pd.DataFrame(payload['rows'])
        else: frame = pd.concat(payload['performance'], names=['Model','Depth','Rank'])
        for column in list(frame.columns):
            if frame[column].map(lambda value: isinstance(value, Estimate)).any():
                frame[f"{column} stars"] = frame[column].map(lambda value: value.stars if isinstance(value, Estimate) else '')
                frame[column] = frame[column].map(lambda value: value.value if isinstance(value, Estimate) else value)
        return frame
    if name=='FullDecileMonotonicity':
        panels={}
        for (family,depth), frame in payload['performance'].items():
            if model and family!=model: continue
            block=frame.reindex([str(i) for i in range(1,11)]+['LS']).copy()
            block.index=['Low']+[str(i) for i in range(2,10)]+['High','H−L']
            block.columns=['Fcst (%)','Mean (%)','SD (%)','SR']
            panels[(family,depth)]=block
        frame=pd.concat(panels,axis=1)
    elif name=='FinalBaselinePerformance':
        frame=payload['table'].copy()
        frame['FF5 α (%)']=[Estimate(v,_stars(t)) for v,t in zip(frame[r'FF5 $\alpha$'],frame['$t$(FF5)'])]
        frame['Δ SR']=[Estimate(v,s or '') for v,s in zip(frame[r'$\Delta$ VW SR'],frame['Delta VW SR stars'])]
        frame['Depth']=pd.Categorical(frame.Depth,DEPTHS,ordered=True)
        frame['Model']=pd.Categorical(frame.Model,MODELS,ordered=True)
        frame=frame.sort_values(['Depth','Model']).rename(columns={'VW Mean':'Mean (%)','VW Vol.':'SD (%)','VW SR':'SR','EW SR':'SR (EW)','$R^2_{OOS}$':'R2 OOS (%)','Turnover':'Turnover (%)'})
        frame=frame.set_index(['Depth','Model'])[['Mean (%)','SD (%)','SR','Δ SR','SR (EW)','R2 OOS (%)','FF5 α (%)','Turnover (%)']]
    elif name in {'ForecastRefinement','AppendixForecastRankingValues'}:
        frame=payload['table'].copy()
        frame['value']=[Estimate(v,'' if m=='Preservation' and family!='Difference' else _stars(t)) for v,t,m,family in zip(frame.Estimate,frame['NW t-stat'],frame.Metric,frame.Model)]
        frame=frame.pivot(index=['Depth Group','Comparison','Model'],columns='Metric',values='value')
        order=['Preservation','Refinement','Value'] if name=='ForecastRefinement' else ['Preservation','Refinement','Shallow','Deep']
        frame=frame.reindex(columns=order).rename(columns={'Preservation':'P','Refinement':'C (%)','Value':'V (bps)','Shallow':'V(S) (bps)','Deep':'V(D) (bps)'})
    elif name in {'ThemePairwiseLosses','AppendixThemeValueLosses'}:
        frame=payload['table'].copy()
        frame['value']=[Estimate(v,_pstars(p)) for v,p in zip(frame.display_mean,frame.holm_p)]
        frame=frame.pivot(index='Theme',columns=['Model','Metric'],values='value')
        if name=='AppendixThemeValueLosses':
            values=payload['values'].copy()
            values['value']=[Estimate(v,_pstars(p)) for v,p in zip(values.Estimate,values['Holm PValue'])]
            frame=pd.concat([frame,values.pivot(index='Theme',columns=['Model','Metric'],values='value')],axis=1)
        metrics=['P','C','V'] if name=='ThemePairwiseLosses' else ['P','C','Deep','Shallow']
        families=[model] if model else ['ResNet','NN','ResNet+','NN+']
        frame=frame.reindex(columns=pd.MultiIndex.from_product([families,metrics])).rename(columns={'P':'Δ P','C':'Δ C (%)','V':'Δ V (bps)','Deep':'Δ V(D) (bps)','Shallow':'Δ V(S) (bps)'},level=1)
    elif name=='TransactionCostAdjustedPerformance':
        raw_frame=payload['table'];rows=[]
        for _,row in raw_frame.iterrows():
            for cost in (0,10,50):
                rows.append([row.Comparison,row.Depth,cost,row[f'Residual SR, {cost} bps'],row[f'Benchmark SR, {cost} bps'],row[rf'$\Delta$ SR, {cost} bps'],row[f'Residual Turnover, {cost} bps'],row[f'Benchmark Turnover, {cost} bps']])
        frame=pd.DataFrame(rows,columns=['Comparison','Depth','Cost (bps)','Residual SR','Feedforward SR','Δ SR','Residual turnover (%)','Feedforward turnover (%)']).set_index(['Comparison','Depth','Cost (bps)'])
    elif name=='MarketStateRobustness':
        rows=[]
        for row in payload['rows']:
            for family,code in [('ResNet vs NN','RN-NN'),('ResNet+ vs NN+','RNP-NNP')]:
                rows.append([family,row['Period'],row['State'],row['Months'],row[f'{code} Residual Mean'],row[f'{code} Benchmark Mean'],Estimate(row[f'{code} Mean'],row[f'{code} Mean Stars']),row[f'{code} t-stat']])
        frame=pd.DataFrame(rows,columns=['Comparison','Period','State','Months','Residual mean (%)','Feedforward mean (%)','Δ Mean (%)','t']).set_index(['Comparison','Period','State'])
    elif name=='SizeExclusionRobustness':
        frame=payload['table'].copy()
        frame['Δ SR']=[Estimate(v,s) for v,s in zip(frame[r'$\Delta$ SR'],frame['SR stars'])]
        frame=frame.set_index(['Comparison','Universe','Depth Group'])[['Res SR','Base SR','Δ SR',r'$t_{\mathrm{LW}}$']]
    elif name=='S1D7DiagnosticSummary':
        frame=payload['table'][['family','depth','prediction_std','prediction_unique_12dp','Constant-Predict Decile Fallback Months','min_hidden_col_std','max_zero_var_frac']].rename(columns={'family':'Model','depth':'Depth','prediction_std':'Forecast SD','prediction_unique_12dp':'Distinct forecasts','Constant-Predict Decile Fallback Months':'Collapsed months','min_hidden_col_std':'Min. hidden SD','max_zero_var_frac':'Max. zero-variance share'}).set_index(['Model','Depth'])
    else:
        frame=payload['table'].copy()
        if model and 'Model' in frame: frame=frame[frame.Model.eq(model)]
        if name=='DepthGroupSharpeContrasts': frame=frame.set_index('Model')[['Deep - Shallow SR','$t$','$p$','Shallow Cells','Deep Cells','Months']]
        elif name=='PredictorList': frame=frame.rename(columns={'order':'No.','acronym':'Acronym','description':'Description','block':'Block','source':'Source','frequency':'Frequency'}).set_index('No.')
        elif name=='EqualWeightedRobustness': frame=frame.rename(columns={'Pair':'Comparison'}).set_index(['Comparison','Depth'])
        elif 'DepthProfile' in name: frame=frame.set_index(['Width','Depth','Model']).drop(columns=['Spec','Months','Scale','$t(\\alpha)$'],errors='ignore')
    if isinstance(frame.columns,pd.MultiIndex): frame.columns=pd.MultiIndex.from_tuples([tuple(clean_label(v) for v in c) for c in frame.columns],names=frame.columns.names)
    else: frame=frame.rename(columns=clean_label)
    if isinstance(frame.index,pd.MultiIndex): frame.index=pd.MultiIndex.from_tuples([tuple(clean_label(v) for v in c) for c in frame.index],names=frame.index.names)
    elif frame.index.dtype==object: frame.index=frame.index.map(clean_label)
    return frame

def styled_table(name,*,model=None):
    """Paper-like display, retaining full precision in table(..., raw=True)."""
    frame=table(name,model=model)
    def fmt(value,precision=2):
        if isinstance(value,Estimate): return '—' if pd.isna(value.value) else f'{0.0 if round(value.value, precision)==0 else value.value:.{precision}f}{value.stars}'
        if isinstance(value,(float,np.floating)): return '—' if pd.isna(value) else f'{0.0 if round(value, precision)==0 else value:.{precision}f}'
        if value is None: return '—'
        return str(value)
    style=frame.style.format(fmt,escape='html').set_caption(TITLES[name]).set_table_styles([
        {'selector':'','props':[('border-collapse','collapse'),('font-family','Georgia, serif'),('font-size','12px'),('color','#202020')]},
        {'selector':'caption','props':[('text-align','left'),('font-size','15px'),('font-weight','bold'),('padding','10px 0')]},
        {'selector':'thead th','props':[('border-top','1px solid #333'),('border-bottom','1px solid #999'),('padding','7px 10px'),('text-align','center')]},
        {'selector':'tbody th','props':[('font-weight','normal'),('text-align','left'),('padding','5px 10px')]},
        {'selector':'td','props':[('text-align','right'),('padding','5px 10px'),('white-space','nowrap')]},
        {'selector':'tbody tr:last-child','props':[('border-bottom','1px solid #333')]},
    ])
    for column in frame:
        label=column[-1] if isinstance(column,tuple) else column
        if label in ['Δ P','Δ SR']:
            style=style.format(lambda value:fmt(value,3),subset=[column],escape='html')
        if name=='S1D7DiagnosticSummary' and label in ['Forecast SD','Min. hidden SD']:
            style=style.format('{:.6g}',subset=[column],na_rep='—')
    if name in {'ExperimentalDesignGrid', 'PredictorList'}:
        style = style.set_table_styles([{'selector': 'td', 'props': [('white-space', 'normal'), ('text-align', 'left'), ('max-width', '380px'), ('overflow-wrap', 'break-word')]}], overwrite=False)
    return style

def figure(name):
    record=ARTIFACTS[name]
    if record[4]!='figure': raise ValueError(f'{name} is a table')
    function=getattr(importlib.import_module(record[2]),record[3])
    function(usetex=False)
    from . import style
    return style._LAST_FIGURE

def rebuild(name,*,cache_root=None):
    record=ARTIFACTS[name]
    if record[0] is None: return None
    payload=getattr(importlib.import_module(record[0]),record[1])()
    write_cache(name,payload,cache_root=cache_root)
    return payload
