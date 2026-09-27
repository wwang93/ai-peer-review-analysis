"""Refit pooled topic models using authorized responses in private_run/."""
from pathlib import Path
import re,json,os,warnings,sys,collections,itertools,platform

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import pandas as pd
import scipy,sklearn
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer,ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from nltk.stem.snowball import SnowballStemmer
from sklearn.exceptions import ConvergenceWarning

OUT=Path(os.environ.get('AI_PEER_REVIEW_WORKDIR',Path(__file__).resolve().parent/'private_run'))
(OUT/'models').mkdir(parents=True,exist_ok=True)
(OUT/'tables').mkdir(exist_ok=True)
label_path=OUT/'models/topic_labels.json'
if not label_path.exists():
    label_path.write_text((Path(__file__).resolve().parent/'topic_labels.json').read_text())
stemmer=SnowballStemmer('english')
GENERIC='chatbot chatbots chat chatbox ai oi gpt peer review process folio processfolio pf writing write written essay essays paper draft final feedback suggest suggested suggestion suggestions help helped helpful using use used really feel felt think thought learned learn learning thing things time make made making need needed lot overall able got gave given way ways work assignment course student students'.split()
STOP={stemmer.stem(w) for w in set(ENGLISH_STOP_WORDS)|set(GENERIC)}

def tokens(s):
    raw=re.findall(r"[a-z]+",s.lower().replace('’',"'"))
    return [stemmer.stem(w) for w in raw if len(w)>2 and stemmer.stem(w) not in STOP]

def vectorize(texts,min_df=2,student_ids=None):
    vec=TfidfVectorizer(tokenizer=tokens,token_pattern=None,lowercase=False,ngram_range=(1,2),min_df=min_df,max_df=.85,sublinear_tf=True)
    X=vec.fit_transform(texts)
    if student_ids is not None:
        terms=vec.get_feature_names_out();presence=np.zeros(X.shape[1])
        ids=np.array(student_ids)
        for sid in np.unique(ids):presence+=(np.asarray((X[ids==sid]>0).sum(axis=0)).ravel()>0)
        vocab=terms[presence>=2]
        vec=TfidfVectorizer(tokenizer=tokens,token_pattern=None,lowercase=False,ngram_range=(1,2),vocabulary=list(vocab),sublinear_tf=True)
        X=vec.fit_transform(texts)
    return vec,X

def fit(X,k,seed=42):
    m=NMF(n_components=k,init='nndsvdar',random_state=seed,max_iter=2000,tol=1e-4)
    with warnings.catch_warnings(record=True) as warns:
        warnings.simplefilter('always',ConvergenceWarning)
        W=m.fit_transform(X)
    H=m.components_.copy()
    # Resolve NMF scale indeterminacy before row-normalizing document weights.
    norms=np.linalg.norm(H,axis=1); W*=norms; H/=np.maximum(norms[:,None],1e-15)
    P=W/np.maximum(W.sum(axis=1,keepdims=True),1e-15)
    return dict(model=m,H=H,W=W,P=P,error=m.reconstruction_err_,iterations=m.n_iter_,converged=not any(isinstance(w.message,ConvergenceWarning) for w in warns))

def alignment(A,B):
    sim=cosine_similarity(A,B); aa,bb=linear_sum_assignment(-sim)
    return bb,float(sim[aa,bb].mean())

def metrics(H,X,terms,P):
    idx=np.argsort(H,axis=1)[:,-10:][:,::-1]
    binX=(X>0).astype(float)
    scores=[]
    for top in idx:
        arr=binX[:,top].toarray();probs=arr.mean(axis=0); vals=[]
        for a,b in itertools.combinations(range(len(top)),2):
            joint=(arr[:,a]*arr[:,b]).mean()
            vals.append(-1 if joint==0 else (0 if joint==1 else np.log(joint/(probs[a]*probs[b]))/-np.log(joint)))
        scores.append(np.mean(vals))
    return dict(npmi=float(np.mean(scores)),topic_diversity=len(set(idx.ravel()))/idx.size,min_dominant_n=int(np.bincount(P.argmax(axis=1),minlength=len(H)).min()),top_terms=['; '.join(terms[t]) for t in idx])

df=pd.read_csv(OUT/'tables/responses_39.csv').fillna('')
for c in ['missing','short_text','editorial_marker']:
    df[c]=df[c].astype(str).str.lower().eq('true')

def docs(corpus,threshold=8,exclude_editorial=False):
    mask=df.corpus.isin(['RQ3','R4_affect']) if corpus=='RQ3_extended' else df.corpus==corpus
    d=df[mask&~df.missing&(df.word_count>=threshold)].copy()
    if exclude_editorial:d=d[~d.editorial_marker]
    return d.reset_index(drop=True)

def candidates():
    stats=[]; representatives=[]
    for corpus in ['RQ1','RQ2','RQ3','RQ3_extended']:
        d=docs(corpus);vec,X=vectorize(d.model_text,student_ids=d.student_id)
        keep=np.asarray(X.sum(axis=1)).ravel()>0;d=d[keep].reset_index(drop=True);X=X[keep]
        terms=vec.get_feature_names_out()
        for k in range(2,8):
            fits=[fit(X,k,s) for s in [11,23,42,71,101]]
            best=min(fits,key=lambda f:f['error'])
            stability=np.mean([alignment(best['H'],f['H'])[1] for f in fits])
            met=metrics(best['H'],X,terms,best['P'])
            top=met.pop('top_terms')
            stats.append(dict(corpus=corpus,k=k,n=len(d),vocab=X.shape[1],seed_stability=stability,reconstruction_error=best['error'],all_converged=all(f['converged'] for f in fits),**met))
            for t,words in enumerate(top):
                ix=np.argsort(best['W'][:,t])[::-1][:4]
                representatives.append(dict(corpus=corpus,k=k,topic=t+1,words=words,examples=[dict(id=d.iloc[i].response_id,text=d.iloc[i].text,weight=float(best['P'][i,t])) for i in ix]))
            print(corpus,k,round(met['npmi'],3),round(stability,3),met['min_dominant_n'],flush=True)
    pd.DataFrame(stats).to_csv(OUT/'tables/model_candidates.csv',index=False)
    (OUT/'models/candidate_topics.json').write_text(json.dumps(representatives,indent=2,ensure_ascii=False))

def final():
    choices=json.loads((OUT/'models/topic_labels.json').read_text())
    all_topics=[];all_weights=[];all_trends=[];all_pairs=[];all_sens=[];review=[]
    rng=np.random.default_rng(9272026)
    for corpus,choice in choices.items():
        k=len(choice['labels']);d=docs(corpus);vec,X=vectorize(d.model_text,student_ids=d.student_id)
        keep=np.asarray(X.sum(axis=1)).ravel()>0;d=d[keep].reset_index(drop=True);X=X[keep]
        terms=vec.get_feature_names_out()
        best=min([fit(X,k,s) for s in [11,23,42,71,101]],key=lambda f:f['error'])
        P=best['P']; H=best['H'];W=best['W']
        np.savez_compressed(OUT/'models'/f'{corpus}_factors.npz',H=H,W=W,P=P,terms=terms,response_ids=d.response_id.values.astype(str))
        # Participant bootstrap with refitting quantifies topic recovery, not a causal confidence claim.
        stable=[]
        sids=d.student_id.unique()
        for b in range(40):
            sampled=rng.choice(sids,len(sids),replace=True)
            indexes=np.concatenate([np.where(d.student_id.values==s)[0] for s in sampled])
            alt=fit(X[indexes],k,int(rng.integers(1,100000)))
            stable.append(alignment(H,alt['H'])[1])
        for t in range(k):
            top=terms[np.argsort(H[t])[::-1][:15]].tolist()
            idx=np.argsort(W[:,t])[::-1][:5]
            all_topics.append(dict(corpus=corpus,topic=t+1,label=choice['labels'][t],top_terms='; '.join(top),n_model=len(d),dominant_n=int((P.argmax(axis=1)==t).sum()),mean_weight=float(P[:,t].mean()),bootstrap_topic_similarity_mean=float(np.mean(stable)),bootstrap_topic_similarity_p10=float(np.quantile(stable,.1)),example_ids='; '.join(d.iloc[idx].response_id)))
            for rank,i in enumerate(idx,1):review.append(dict(corpus=corpus,topic=t+1,label=choice['labels'][t],selection=f'top_loading_{rank}',response_id=d.iloc[i].response_id,round=int(d.iloc[i]['round']),text=d.iloc[i].text,source=d.iloc[i].source,model_weight=float(P[i,t]),reviewer_code='',reviewer_decision='',reviewer_notes=''))
        for i,row in d.iterrows():
            for t in range(k):all_weights.append(dict(response_id=row.response_id,student_id=row.student_id,round=int(row['round']),corpus=corpus,topic=t+1,label=choice['labels'][t],weight=float(P[i,t]),dominant=(P[i].argmax()==t),source=row.source))
        rounds=sorted(d['round'].unique())
        for r in rounds:
            mask=d['round'].values==r; arr=P[mask]; n=len(arr)
            boots=arr[rng.integers(0,n,size=(2000,n))].mean(axis=1)
            for t in range(k):all_trends.append(dict(corpus=corpus,round=int(r),topic=t+1,label=choice['labels'][t],n_model=n,dominant_n=int((arr.argmax(axis=1)==t).sum()),mean_weight=float(arr[:,t].mean()),ci_low=float(np.quantile(boots[:,t],.025)),ci_high=float(np.quantile(boots[:,t],.975)),interval='participant resampling, fixed topic model'))
        complete=d.groupby('student_id')['round'].nunique(); complete=complete[complete==len(rounds)].index
        for t in range(k):
            sub=d.assign(weight=P[:,t]); pivot=sub[sub.student_id.isin(complete)].pivot(index='student_id',columns='round',values='weight')
            delta=(pivot[rounds[-1]]-pivot[rounds[0]]).values
            boots=delta[rng.integers(0,len(delta),size=(2000,len(delta)))].mean(axis=1)
            all_pairs.append(dict(corpus=corpus,topic=t+1,label=choice['labels'][t],n_complete=len(delta),first_round=int(rounds[0]),last_round=int(rounds[-1]),first_mean=pivot[rounds[0]].mean(),last_mean=pivot[rounds[-1]].mean(),paired_change=float(delta.mean()),ci_low=float(np.quantile(boots,.025)),ci_high=float(np.quantile(boots,.975))))
        for setting,threshold,mindf,exclude in [('include_short',1,2,False),('document_df_only',8,2,False),('exclude_editorial',8,2,True),('min_15_words',15,2,False),('deduplicate_text',8,2,False)]:
            sd=docs(corpus,threshold,exclude)
            if setting=='deduplicate_text':sd=sd.drop_duplicates('model_text').reset_index(drop=True)
            sv,sx=vectorize(sd.model_text,mindf,student_ids=None if setting=='document_df_only' else sd.student_id)
            keep=np.asarray(sx.sum(axis=1)).ravel()>0;sd=sd[keep].reset_index(drop=True);sx=sx[keep]
            alt=fit(sx,k,42)
            # Align using a shared vocabulary to avoid comparing dimensions with different meanings.
            st=sv.get_feature_names_out();common=np.intersect1d(terms,st);a=np.searchsorted(terms,common);b=np.searchsorted(st,common)
            order,sim=alignment(H[:,a],alt['H'][:,b]); ap=alt['P'][:,order]
            for t in range(k):
                p1=ap[sd['round']==rounds[0],t].mean();plast=ap[sd['round']==rounds[-1],t].mean()
                all_sens.append(dict(corpus=corpus,setting=setting,topic=t+1,label=choice['labels'][t],n=len(sd),aligned_similarity=sim,first_mean=p1,last_mean=plast,change=plast-p1))
        # Round 4 affect responses projected into RQ3 model, explicitly incomparable prompt.
        if corpus=='RQ3':
            ad=docs('R4_affect'); ax=vec.transform(ad.model_text)
            keep=np.asarray(ax.sum(axis=1)).ravel()>0;ad=ad[keep].reset_index(drop=True);ax=ax[keep]
            aw=best['model'].transform(ax)
            norms=np.linalg.norm(best['model'].components_,axis=1);aw*=norms
            ap=aw/np.maximum(aw.sum(axis=1,keepdims=True),1e-15)
            for i,row in ad.iterrows():
                for t in range(k):all_weights.append(dict(response_id=row.response_id,student_id=row.student_id,round=4,corpus='R4_affect_projection',topic=t+1,label=choice['labels'][t],weight=float(ap[i,t]),dominant=(ap[i].argmax()==t),source=row.source))
        # Low-information/mixed cases remain in review, never treated as confident codes.
        idx=np.argsort(P.max(axis=1))[:8]
        for i in idx:
            t=int(P[i].argmax());review.append(dict(corpus=corpus,topic=t+1,label=choice['labels'][t],selection='mixed_topic',response_id=d.iloc[i].response_id,round=int(d.iloc[i]['round']),text=d.iloc[i].text,source=d.iloc[i].source,model_weight=float(P[i,t]),reviewer_code='',reviewer_decision='',reviewer_notes=''))
        print(corpus,'final',k,'n',len(d),'complete',len(complete),'bootstrap similarity',np.mean(stable),flush=True)
    for name,rows in [('topics',all_topics),('topic_weights',all_weights),('topic_by_round',all_trends),('paired_changes',all_pairs),('sensitivity',all_sens),('review_sample',review)]:pd.DataFrame(rows).to_csv(OUT/'tables'/f'{name}.csv',index=False)
    (OUT/'models/run_config.json').write_text(json.dumps(dict(python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__,scipy=scipy.__version__,stopwords=sorted(STOP),custom_generic=GENERIC,seeds=[11,23,42,71,101],min_words=8,min_df=2,min_distinct_students_per_term=2,max_df=.85,ngram_range=[1,2],bootstrap_refits=40,bootstrap_means=2000,choices=choices),ensure_ascii=False,indent=2))

if __name__=='__main__':
    candidates() if len(sys.argv)<2 or sys.argv[1]=='candidates' else final()
