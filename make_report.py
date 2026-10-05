"""Generate publication plots, tables and numerical claims from measured CSVs."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_experiments import ROOT,RESULT,CACHE,load_ref
from audio_pipeline import decode,compare

OUTPUT=ROOT/'artifacts'/'reproduced'
FIG=OUTPUT/'figures'
TABLE=OUTPUT/'tables'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
 'axes.spines.right':False,'figure.dpi':140,'savefig.dpi':300,'pdf.fonttype':42})
COL={'wav':'#777777','wavfloat':'#444444','mp3':'#0072B2','aac':'#D55E00','opus':'#009E73'}
POL=['MP3-128','AAC-128','Opus-64','PCM-16','Global-only','Tail-constrained']
PLAB=['MP3 128','AAC 128','Opus 64','PCM 16','Global only','Tail constraint']

def save(fig,name):
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(FIG/(name+'.png'),bbox_inches='tight')
    plt.close(fig)

def paired_bootstrap(values,seed=20261004):
    v=np.asarray(values)
    rng=np.random.default_rng(seed)
    estimates=np.mean(v[rng.integers(0,len(v),(10000,len(v)))],axis=1)
    return dict(mean=float(v.mean()),lo=float(np.quantile(estimates,.025)),hi=float(np.quantile(estimates,.975)))

def main(results_dir=None, output_dir=None):
    global RESULT,OUTPUT,FIG,TABLE
    RESULT=Path(results_dir) if results_dir else ROOT/'results'/'recorded'
    OUTPUT=Path(output_dir) if output_dir else ROOT/'artifacts'/'reproduced'
    FIG=OUTPUT/'figures'; TABLE=OUTPUT/'tables'
    FIG.mkdir(parents=True,exist_ok=True); TABLE.mkdir(parents=True,exist_ok=True)
    candidates=pd.read_csv(RESULT/'candidate_metrics.csv')
    policies=pd.read_csv(RESULT/'policy_metrics.csv')
    sources=json.loads((RESULT/'source_manifest.json').read_text(encoding='utf8'))
    manifest={r['source_id']:r for r in sources}
    public=policies[policies.group.str.startswith('public')]
    summary={'source_count':len(sources),'candidate_count':len(candidates),'public':{},'supplied':{},'alignment':{}}
    rows=[]
    for group,n in [('public_speech',40),('public_music',50)]:
        subset=public[public.group==group]
        if subset.empty:
            continue
        n=subset.source_id.nunique()
        summary['public'][group]={}
        for policy in POL:
            sub=subset[subset.policy==policy]
            summary['public'][group][policy]=dict(n=len(sub),mean_kib=float(sub.bytes.mean()/1024),
                median_snr=float(sub.snr_db.median()),median_tail=float(sub.tail_snr_db.median()),
                global_failures=int((~sub.passes_global).sum()),tail_failures=int((~sub.passes_tail).sum()),
                both_passes=int(sub.passes_both.sum()),mean_stoi=float(sub.stoi.mean()) if group=='public_speech' else None,
                mean_lsd=float(sub.lsd_db.mean()),mean_sc=float(sub.spectral_convergence.mean()))
            r=summary['public'][group][policy]
            label='Speech' if group=='public_speech' else 'Music'
            rows.append(f"{label} & {PLAB[POL.index(policy)]} & {r['mean_kib']:.1f} & {r['median_snr']:.1f} & {r['median_tail']:.1f} & {r['both_passes']}/{n} & {r['mean_sc']:.3f}" + r" \\")
        pivot=subset.pivot(index='source_id',columns='policy',values='bytes')
        for baseline in ['MP3-128','AAC-128','Opus-64','PCM-16','Global-only']:
            savings=100*(1-pivot['Tail-constrained']/pivot[baseline])
            summary['public'][group]['savings_vs_'+baseline]=paired_bootstrap(savings)
        selected=subset[subset.policy=='Tail-constrained']
        summary['public'][group]['choices']=selected.groupby('config').size().to_dict()
        summary['public'][group]['codec_counts']=selected.groupby('codec').size().to_dict()
    (TABLE/'policy_table.tex').write_text('\n'.join(rows),encoding='utf8')
    supplied=policies[policies.group.str.startswith('supplied')]
    lines=[]
    for sid in ['supplied_speech','supplied_music']:
        subset=supplied[supplied.source_id==sid]
        if subset.empty:
            continue
        for policy in ['Global-only','Tail-constrained']:
            r=subset[subset.policy==policy].iloc[0]
            codec=r.codec.upper()
            depth=f"{int(r.depth)} bit" if r.codec.startswith('wav') else f"{int(r.bitrate/1000)} kbps"
            label='Speech' if 'speech' in sid else 'Music'
            method='Global only' if policy=='Global-only' else 'Tail constraint'
            lines.append(f"{label} & {method} & {codec} & {r.rate/1000:g} & {depth} & {r.bytes/1024:.2f} & {r.snr_db:.2f} & {r.tail_snr_db:.2f}" + r" \\")
            summary['supplied'][sid+'_'+policy]={k:float(r[k]) for k in ['bytes','snr_db','tail_snr_db']}
    (TABLE/'supplied_table.tex').write_text('\n'.join(lines),encoding='utf8')
    aac=candidates[candidates.codec=='aac']
    summary['alignment']=dict(aac_count=len(aac),median_raw=float(aac.raw_snr_db.median()),
        median_aligned=float(aac.snr_db.median()),median_gain=float((aac.snr_db-aac.raw_snr_db).median()),
        delay_min=float(aac.delay_ms.min()),delay_max=float(aac.delay_ms.max()))
    robust=pd.read_csv(RESULT/'frame_activity_sensitivity.csv')
    summary['robustness']={}
    for (group,policy,frame,activity),r in robust.groupby(['group','policy','frame_ms','activity_db']):
        summary['robustness'][f'{group}/{policy}/{frame}/{activity}']=dict(n=len(r),tail_failures=int((r.tail_snr_db<15).sum()),median_tail=float(r.tail_snr_db.median()))
    summary['timing']=dict(total_encoding_seconds=float(candidates.encoded_ms.sum()/1000),
        total_decoding_seconds=float(candidates.decoded_ms.sum()/1000),total_metrics_seconds=float(candidates.metrics_ms.sum()/1000),
        per_source_search_seconds=candidates.assign(total=(candidates.encoded_ms+candidates.decoded_ms+candidates.metrics_ms)/1000).groupby('source_id').total.sum().describe().to_dict())
    summary['fall_back_count']=int(((policies.policy=='Tail-constrained')&(policies.codec=='wavfloat')).sum())
    summary['all_configs_valid']=bool('error' not in candidates or candidates.error.isna().all())
    (OUTPUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf8')

    if not supplied.empty:
        fig,axs=plt.subplots(2,2,figsize=(7,4.7),constrained_layout=True)
        for col,sid in enumerate(['supplied_speech','supplied_music']):
            sub=candidates[(candidates.source_id==sid)&(candidates.codec!='wavfloat')]
            for k,key,threshold in [(0,'snr_db',20),(1,'tail_snr_db',15)]:
                ax=axs[k,col]
                for codec,r in sub.groupby('codec'):
                    ax.scatter(r.bytes/1024,r[key],s=15,alpha=.75,c=COL[codec],label=codec.upper())
                ax.axhline(threshold,color='#555555',ls='--',lw=.8)
                for policy,marker in [('Global-only','*'),('Tail-constrained','D')]:
                    r=supplied[(supplied.source_id==sid)&(supplied.policy==policy)].iloc[0]
                    ax.scatter(r.bytes/1024,r[key],s=100 if marker=='*' else 55,marker=marker,
                        facecolors='none',edgecolors='black',linewidth=1.2,zorder=5)
                ax.set_xscale('log'); ax.set_ylim(-10,125)
                ax.set_xlabel('File size (KiB)'); ax.set_ylabel('Global SNR (dB)' if k==0 else 'Tail SNR (dB)')
                ax.set_title(('Speech' if col==0 else 'Music')+(' — global' if k==0 else ' — tail'))
        axs[0,0].legend(fontsize=7,ncol=2,loc='upper left')
        save(fig,'supplied_rate_fidelity')

    if not public.empty:
        fig,axs=plt.subplots(2,2,figsize=(7,4.6),constrained_layout=True)
        for col,group in enumerate(['public_speech','public_music']):
            sub=public[public.group==group]
            vals=[sub[sub.policy==p].bytes/1024 for p in POL]
            axs[0,col].boxplot(vals,showfliers=False,patch_artist=False,tick_labels=PLAB,widths=.5)
            axs[0,col].set_yscale('log'); axs[0,col].set_ylabel('File size (KiB)')
            axs[0,col].set_title('Speech (40 speakers)' if col==0 else 'Music (50 tracks)')
            rates=[100*sub[sub.policy==p].passes_both.mean() for p in POL]
            axs[1,col].bar(range(len(POL)),rates,color=['#777777']*4+['#0072B2','#D55E00'])
            axs[1,col].set_xticks(range(len(POL)),PLAB); axs[1,col].set_ylim(0,110)
            axs[1,col].set_ylabel('Both constraints passed (%)')
            for i,v in enumerate(rates): axs[1,col].text(i,v+2,f'{v:.0f}',ha='center',fontsize=8)
            for ax in axs[:,col]: ax.tick_params(axis='x',rotation=35)
        save(fig,'public_outcomes')

    s=pd.read_csv(RESULT/'threshold_sensitivity.csv')
    if not public.empty:
        fig,axs=plt.subplots(1,2,figsize=(7,2.75),constrained_layout=True)
        for ax,group in zip(axs,['public_speech','public_music']):
            data=s[s.group==group].pivot_table(index='tail_db',columns='global_db',values='bytes',aggfunc='mean')/1024
            im=ax.imshow(data,origin='lower',aspect='auto',cmap='cividis')
            ax.set_xticks(range(len(data.columns)),data.columns); ax.set_yticks(range(len(data.index)),data.index)
            ax.set_xlabel('Global threshold (dB)'); ax.set_ylabel('Tail threshold (dB)')
            ax.set_title('Speech: mean KiB' if group=='public_speech' else 'Music: mean KiB')
            for y in range(len(data.index)):
                for x in range(len(data.columns)):
                    ax.text(x,y,f'{data.iloc[y,x]:.1f}',ha='center',va='center',color='white' if data.iloc[y,x]<data.to_numpy().mean() else 'black',fontsize=9)
            fig.colorbar(im,ax=ax,shrink=.75)
        save(fig,'threshold_sensitivity')

    sid='supplied_speech'
    reference=ROOT/'data'/'C题'/'附件1'/'原始语音_48kHz_24bit.wav'
    selected=supplied[(supplied.source_id==sid)&supplied.policy.isin(['Global-only','Tail-constrained'])]
    can_plot_temporal=(sid in manifest and reference.exists() and len(selected)==2
                       and all((CACHE/sid/(r.config+r.extension)).exists() for r in selected.itertuples()))
    if can_plot_temporal:
        row=manifest[sid]; x,sr=load_ref(row)
        fig,axs=plt.subplots(2,1,figsize=(7,3.3),sharex=True,constrained_layout=True)
        t=np.arange(len(x))/sr
        axs[0].plot(t[::24],x[::24],lw=.45,c='#555555'); axs[0].set_ylabel('Reference amplitude')
        for policy,color in [('Global-only','#0072B2'),('Tail-constrained','#D55E00')]:
            r=supplied[(supplied.source_id==sid)&(supplied.policy==policy)].iloc[0]
            y,dsr,_=decode(CACHE/sid/(r.config+r.extension)); m=compare(x,y,sr,decoded_rate=dsr)
            tt=np.arange(len(m['frame_snr_db']))*.02
            vals=m['frame_snr_db'].copy(); vals[~m['frame_active']]=np.nan
            axs[1].plot(tt,vals,c=color,label=policy,lw=.85)
        axs[1].axhline(15,ls='--',c='#555555',lw=.7)
        axs[1].set_ylabel('Active-frame SNR (dB)'); axs[1].set_xlabel('Time (s)'); axs[1].legend(fontsize=8,ncol=2)
        axs[1].set_ylim(-15,60); axs[1].set_xlim(0,len(x)/sr)
        save(fig,'temporal_case')

    else:
        print('Skipped temporal_case: optional original supplied speech and selected bitstreams are unavailable.')

    fig,axs=plt.subplots(1,2,figsize=(7,2.75),constrained_layout=True)
    aa=aac[aac.group.str.startswith('public')]
    axs[0].scatter(aa.raw_snr_db,aa.snr_db,s=6,c='#D55E00',alpha=.3)
    axs[0].set_xlabel('Unaligned AAC SNR (dB)'); axs[0].set_ylabel('Aligned AAC SNR (dB)')
    chosen=public[public.policy=='Tail-constrained']
    for i,g in enumerate(['public_speech','public_music']):
        counts=chosen[chosen.group==g].codec.value_counts()
        offset=0
        for c in ['wav','mp3','aac','opus','wavfloat']:
            v=int(counts.get(c,0)); axs[1].barh(i,v,left=offset,color=COL[c],label=c.upper() if i==0 else None)
            if v:axs[1].text(offset+v/2,i,str(v),ha='center',va='center',color='white',fontsize=8)
            offset+=v
    axs[1].set_yticks([0,1],['Speech','Music']); axs[1].set_xlabel('Selected clips'); axs[1].legend(fontsize=7,ncol=2)
    save(fig,'alignment_choices')
    print(json.dumps(summary['public'],indent=2))
    print('alignment',summary['alignment'],'timing',summary['timing'])

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=ROOT/'results'/'recorded')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts'/'reproduced')
    args=parser.parse_args()
    main(args.results_dir,args.output_dir)
