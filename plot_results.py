"""Regenerate the five public figures from aggregate CSV files."""
from pathlib import Path
import json,re,html,base64,os,textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

OUT=Path(__file__).resolve().parent; FIG=OUT
labels=json.loads((OUT/'topic_labels.json').read_text())
tr=pd.read_csv(OUT/'topic_by_round.csv'); topics=pd.read_csv(OUT/'topics.csv'); pairs=pd.read_csv(OUT/'paired_changes.csv')
diag=pd.read_csv(OUT/'model_candidates.csv');sens=pd.read_csv(OUT/'sensitivity.csv')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
def save(fig,name):
    fig.savefig(FIG/f'{name}.png',dpi=180,bbox_inches='tight');fig.savefig(FIG/f'{name}.svg',bbox_inches='tight');plt.close(fig)
def en(c,t):return labels[c]['english'][int(t)-1]

fig,axes=plt.subplots(3,1,figsize=(12,11),gridspec_kw={'height_ratios':[4,3,4]})
for ax,c in zip(axes,['RQ1','RQ2','RQ3']):
    a=tr[tr.corpus==c];tab=a.pivot(index='topic',columns='round',values='mean_weight')
    im=ax.imshow(tab.values,cmap='Blues',vmin=0,vmax=.6,aspect='auto')
    ax.set_yticks(range(len(tab)),[en(c,t) for t in tab.index]);n=a.groupby('round').n_model.first()
    ax.set_xticks(range(len(tab.columns)),[f'Round {r} (n={n[r]})' for r in tab.columns]);ax.tick_params(length=0)
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):ax.text(j,i,f'{tab.iloc[i,j]:.1%}',ha='center',va='center',color='white' if tab.iloc[i,j]>.35 else '#17324d')
    ax.set_title({'RQ1':'RQ1  Student-reported feedback','RQ2':'RQ2  Student-reported revisions','RQ3':'RQ3  Reported learning: comparable questions only'}[c],loc='left',fontweight='bold',pad=14)
fig.suptitle('Exploratory themes across rounds',x=.02,ha='left',fontsize=19,fontweight='bold')
fig.text(.02,.016,'Cells = mean normalized NMF loading; not percentages of students with a verified code.\nRQ3 has no comparable Round 4 learning question. Models fitted to pooled rounds within each corpus.',fontsize=10,color='#526273')
fig.subplots_adjust(left=.34,right=.97,top=.91,bottom=.085,hspace=.7);save(fig,'01_topic_patterns')

coverage=pd.read_csv(OUT/'analysis_denominators.csv')
fig,ax=plt.subplots(figsize=(11,6.7))
y=np.arange(len(coverage))
valid=coverage['有效学生文本'].to_numpy(); modeled=coverage['模型文本'].to_numpy()
ax.barh(y,modeled,color='#1f647e',label='Modeled')
ax.barh(y,valid-modeled,left=modeled,color='#dba34d',label='Qualitative review only')
ax.barh(y,39-valid,left=valid,color='#e7e9ee',label='Missing / editorial-only')
ax.set_yticks(y,[f"{r['语料'].replace('R4_affect','Supplement')} / Round {r['轮次']}" for _,r in coverage.iterrows()])
ax.invert_yaxis();ax.set_xlim(0,41);ax.set_xlabel('Number of students (39 expected per question)')
for i,(v,m) in enumerate(zip(valid,modeled)):
    ax.text(m-.5,i,str(m),ha='right',va='center',color='white',fontsize=9)
ax.set_title('Response coverage by question and round',loc='left',fontsize=17,fontweight='bold',pad=18)
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.13),ncol=3,frameon=False,fontsize=9)
fig.subplots_adjust(left=.22,right=.97,top=.9,bottom=.17);save(fig,'02_response_coverage')

p=pairs[pairs.corpus!='RQ3_extended'].copy().reset_index(drop=True)
fig,ax=plt.subplots(figsize=(12,7.2));colors={'RQ1':'#286882','RQ2':'#ad6944','RQ3':'#7363a2'}
for i,row in p.iterrows():
    ax.plot([100*row.ci_low,100*row.ci_high],[i,i],color=colors[row.corpus],linewidth=2)
    ax.scatter(100*row.paired_change,i,color=colors[row.corpus],s=45)
ax.axvline(0,color='#8b96a4',linewidth=1,linestyle='--');ax.invert_yaxis()
ax.set_yticks(range(len(p)),[f'{r.corpus}: {en(r.corpus,r.topic)}' for r in p.itertuples()],fontsize=9)
ax.set_xlabel('Change in mean normalized loading (percentage points)')
ax.set_title('Within-student changes remain uncertain',loc='left',fontsize=17,fontweight='bold',pad=16)
ax.grid(axis='x',alpha=.15);fig.text(.02,.016,'Complete modeled panels: RQ1 n=32 and RQ2 n=27 (R4 - R1); RQ3 n=30 (R3 - R1).\n95% participant bootstrap intervals condition on the fitted topics; model uncertainty is not included. No multiplicity-adjusted tests.',fontsize=9,color='#526273')
fig.subplots_adjust(left=.4,right=.97,top=.87,bottom=.16);save(fig,'03_paired_changes')

fig,axes=plt.subplots(1,2,figsize=(12,5))
for c in labels:
    sub=diag[diag.corpus==c];axes[0].plot(sub.k,sub.npmi,'o-',label=c.replace('_extended',' mixed prompts'))
axes[0].set(xlabel='Number of topics',ylabel='Document-level NPMI',title='Coherence depends on the corpus');axes[0].axhline(0,color='#777',ls='--',lw=.8);axes[0].legend(fontsize=8,frameon=False)
tb=topics.groupby('corpus',sort=False).first(); axes[1].bar(range(4),tb.bootstrap_topic_similarity_mean,color=['#286882','#ad6944','#7363a2','#9a9da6'])
axes[1].set_xticks(range(4),['RQ1','RQ2','RQ3','RQ3 mixed\nprompts']);axes[1].set(ylim=(0,1),ylabel='Mean aligned topic cosine',title='Participant resampling stability')
for i,v in enumerate(tb.bootstrap_topic_similarity_mean):axes[1].text(i,v+.025,f'{v:.2f}',ha='center')
fig.suptitle('Diagnostics support cautious, exploratory use',x=.02,ha='left',fontsize=16,fontweight='bold');fig.text(.02,.01,'40 participant-bootstrap refits per selected model. Initialization stability alone does not establish semantic validity.',fontsize=9,color='#526273');fig.subplots_adjust(top=.8,bottom=.19,wspace=.3);save(fig,'04_model_diagnostics')

ex=tr[tr.corpus=='RQ3_extended'];tab=ex.pivot(index='topic',columns='round',values='mean_weight')
fig,ax=plt.subplots(figsize=(11,5.7));bottom=np.zeros(4)
for t,color in zip(tab.index,['#3b738f','#4b998c','#c98450','#9b94b5']):
    vals=tab.loc[t].values;ax.bar([0,1,2,4],vals,bottom=bottom,color=color,width=.7,label=en('RQ3_extended',t));bottom+=vals
ax.set_xticks([0,1,2,4],['R1 learning','R2 learning','R3 learning','R4 affect']);ax.set(ylim=(0,1),ylabel='Mean normalized NMF loading')
ax.axvline(3,color='#888',ls='--');ax.set_title('Changing the question changes the apparent topic mix',loc='left',fontweight='bold',fontsize=16,pad=20)
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.15),ncol=2,frameon=False,fontsize=9)
fig.text(.02,.018,'Auxiliary pooled discovery only: R4 asks about alleviating challenges, not the learning question used in R1-R3.\nThe R4 shift is confounded by the prompt change and must not be reported as growth or decline in learning.',fontsize=9,color='#526273')
fig.subplots_adjust(top=.82,bottom=.30,left=.09,right=.97);save(fig,'05_prompt_change')

