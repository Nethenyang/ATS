"""Execute measured, finite-candidate offline codec selection experiments."""
from pathlib import Path
import argparse, hashlib, json, platform, time, sys
from datetime import datetime, timezone
import numpy as np
import scipy
from scipy import signal
import soundfile as sf
import pandas as pd
import av
from pystoi import stoi
from audio_pipeline import encode, decode, compare, fidelity_arrays, choose, resample

ROOT=Path(__file__).resolve().parent
WORK=ROOT/'data'
RESULT=ROOT/'results'/'recorded'
CACHE=ROOT/'data'/'encoded'; CACHE.mkdir(parents=True,exist_ok=True)

def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def supplied_manifest():
    rows=[]
    for p in sorted((WORK/'C题'/'附件1').glob('原始*.wav')):
        x,sr=sf.read(p,dtype='float32')
        kind='speech' if '语音' in p.name else 'music'
        rows.append(dict(source_id='supplied_'+kind,kind=kind,group='supplied_'+kind,
            reference_path=p.relative_to(WORK).as_posix(), independent_unit=p.name,
            original_member=p.name,rate=sr,duration=len(x)/sr,sha256=hash_file(p)))
    return rows

def load_ref(row):
    path = (WORK if row['group'].startswith('supplied') else ROOT)/Path(row['reference_path'].replace(chr(92), '/'))
    x,sr=sf.read(path,dtype='float32')
    if x.ndim>1:
        x=x.mean(axis=1)
    return x,sr

def configurations(sr):
    configs=[]
    rates=sorted({r for r in [8000,16000,32000,sr] if r<=sr})
    for rate in rates:
        for depth in [8,16,24]:
            configs.append(dict(codec='wav',rate=rate,depth=depth,bitrate=0,config=f'wav_{rate}_{depth}',extension='.wav'))
        for codec in ['mp3','aac']:
            for bitrate in [32000,64000,96000,128000]:
                configs.append(dict(codec=codec,rate=rate,depth=0,bitrate=bitrate,config=f'{codec}_{rate}_{bitrate}',extension='.'+codec))
    for bitrate in [32000,64000,96000,128000]:
        configs.append(dict(codec='opus',rate=48000,depth=0,bitrate=bitrate,config=f'opus_48000_{bitrate}',extension='.ogg'))
    configs.append(dict(codec='wavfloat',rate=sr,depth=32,bitrate=0,config=f'wavfloat_{sr}_32',extension='.wav'))
    return configs

def scalar_metrics(m):
    keys=['snr_db','tail_snr_db','raw_snr_db','delay_samples','delay_ms','distortion','tail_distortion','active_frames','decoded_samples','reference_samples']
    return {k:m[k] for k in keys}

def spectral_metrics(x,y,sr):
    frame=2**int(np.ceil(np.log2(.032*sr)))
    _,_,X=signal.stft(x,sr,nperseg=frame,noverlap=frame//2,boundary='zeros')
    _,_,Y=signal.stft(y,sr,nperseg=frame,noverlap=frame//2,boundary='zeros')
    X,Y=np.abs(X),np.abs(Y)
    floor=max(X.max()*1e-3,1e-12)
    logdiff=20*np.log10(np.maximum(X,floor)/np.maximum(Y,floor))
    return dict(spectral_convergence=float(np.linalg.norm(X-Y)/max(np.linalg.norm(X),1e-12)),
        lsd_db=float(np.sqrt(np.mean(logdiff**2))))

def measure_source(row):
    target=RESULT/'per_source'/(row['source_id']+'.json')
    if target.exists():
        cached=json.loads(target.read_text(encoding='utf8'))
        if all(r.get('protocol_version')=='1.1'
               and r.get('reference_sha256')==row['sha256']
               and (CACHE/row['source_id']/(r['config']+r['extension'])).is_file()
               and hash_file(CACHE/row['source_id']/(r['config']+r['extension']))==r.get('encoded_sha256')
               for r in cached):
            return cached
    target.parent.mkdir(exist_ok=True)
    x,sr=load_ref(row)
    results=[]
    for config in configurations(sr):
        path=CACHE/row['source_id']/(config['config']+config['extension'])
        start=time.perf_counter()
        try:
            encode(x,sr,config,path)
            encoded_ms=1000*(time.perf_counter()-start)
            start=time.perf_counter()
            y,decoded_rate,actual=decode(path)
            decoded_ms=1000*(time.perf_counter()-start)
            start=time.perf_counter()
            m=compare(x,y,sr,decoded_rate=decoded_rate)
            metrics_ms=1000*(time.perf_counter()-start)
            result=dict(source_id=row['source_id'],group=row['group'],kind=row['kind'],protocol_version='1.1',
                reference_sha256=row['sha256'],**config,
                **scalar_metrics(m),bytes=path.stat().st_size,actual_decoder=actual,
                decoded_rate=decoded_rate,encoded_ms=encoded_ms,decoded_ms=decoded_ms,metrics_ms=metrics_ms,
                effective_kbps=8*path.stat().st_size/len(x)*sr/1000,encoded_sha256=hash_file(path))
            results.append(result)
        except Exception as exc:
            results.append(dict(source_id=row['source_id'],group=row['group'],**config,error=repr(exc)))
    target.write_text(json.dumps(results,indent=2),encoding='utf8')
    return results

def policies(rows,row):
    sr=row['rate']
    byid={r['config']:r for r in rows}
    mapping={'Global-only':choose(rows,20,None),'Tail-constrained':choose(rows,20,15),
        'MP3-128':byid[f'mp3_{sr}_128000'],'AAC-128':byid[f'aac_{sr}_128000'],
        'Opus-64':byid['opus_48000_64000'],'PCM-16':byid[f'wav_{sr}_16']}
    return mapping

def main(supplied_only=False, public_only=False, output_dir=None):
    global RESULT
    RESULT=Path(output_dir) if output_dir else ROOT/'runs'/'latest'/'results'
    RESULT.mkdir(parents=True,exist_ok=True)
    sources=[] if public_only else supplied_manifest()
    if supplied_only and not sources:
        raise FileNotFoundError('Place the optional original references in data/C题/附件1 before a supplied-only run.')
    if not supplied_only:
        public=ROOT/'data'/'public_manifest.json'
        if not public.exists():
            raise FileNotFoundError('run download_data.py before full experiments')
        sources+=json.loads(public.read_text(encoding='utf8'))
    (RESULT/'source_manifest.json').write_text(json.dumps(sources,indent=2,ensure_ascii=False),encoding='utf8')
    (RESULT/'environment.json').write_text(json.dumps(dict(python=sys.version,platform=platform.platform(),
        numpy=np.__version__,scipy=scipy.__version__,soundfile=sf.__version__,pyav=av.__version__,
        av_libraries=av.library_versions,primary=dict(global_db=20,tail_db=15,frame_ms=20,activity_db=-40,tail_fraction=.1),
        experiment_date=datetime.now(timezone.utc).date().isoformat(),recorded_protocol_date='2026-10-04',selection='exact finite enumeration; no training'),indent=2),encoding='utf8')
    allrows,selected,sensitivity,robustness=[],[],[],[]
    for index,row in enumerate(sources):
        start=time.perf_counter()
        rows=measure_source(row)
        valid=[r for r in rows if 'error' not in r]
        allrows+=rows
        x,sr=load_ref(row)
        metrics_cache={}; signal_cache={}
        for policy,r in policies(valid,row).items():
            if r is None:
                continue
            if r['config'] not in metrics_cache:
                path=CACHE/row['source_id']/(r['config']+r['extension'])
                y,dsr,_=decode(path)
                m=compare(x,y,sr,decoded_rate=dsr)
                signal_cache[r['config']]=m['aligned']
                extra=spectral_metrics(x,m['aligned'],sr)
                if row['kind']=='speech':
                    extra['stoi']=float(stoi(x,m['aligned'],sr,extended=False))
                else:
                    extra['stoi']=None
                metrics_cache[r['config']]=extra
            if policy in ['Global-only','Tail-constrained']:
                for frame_ms in [10,40]:
                    q=fidelity_arrays(x,signal_cache[r['config']],sr,frame_ms=frame_ms)
                    robustness.append(dict(source_id=row['source_id'],group=row['group'],policy=policy,frame_ms=frame_ms,
                        activity_db=-40,tail_snr_db=q['tail_snr_db']))
                for activity_db in [-30,-50]:
                    q=fidelity_arrays(x,signal_cache[r['config']],sr,activity_db=activity_db)
                    robustness.append(dict(source_id=row['source_id'],group=row['group'],policy=policy,frame_ms=20,
                        activity_db=activity_db,tail_snr_db=q['tail_snr_db']))
            selected.append(dict(**r,policy=policy,**metrics_cache[r['config']],
                passes_global=r['snr_db']>=20,passes_tail=r['tail_snr_db']>=15,
                passes_both=r['snr_db']>=20 and r['tail_snr_db']>=15))
        for global_db in [15,20,25,30]:
            for tail_db in [10,15,20]:
                r=choose(valid,global_db,tail_db)
                sensitivity.append(dict(source_id=row['source_id'],group=row['group'],global_db=global_db,tail_db=tail_db,
                    config=r['config'] if r else None,bytes=r['bytes'] if r else None,
                    snr_db=r['snr_db'] if r else None,tail_snr_db=r['tail_snr_db'] if r else None))
        print(index+1,'/',len(sources),row['source_id'],'candidates',len(valid),'seconds',round(time.perf_counter()-start,2),flush=True)
    pd.DataFrame(allrows).to_csv(RESULT/'candidate_metrics.csv',index=False,encoding='utf8')
    pd.DataFrame(selected).to_csv(RESULT/'policy_metrics.csv',index=False,encoding='utf8')
    pd.DataFrame(sensitivity).to_csv(RESULT/'threshold_sensitivity.csv',index=False,encoding='utf8')
    pd.DataFrame(robustness).to_csv(RESULT/'frame_activity_sensitivity.csv',index=False,encoding='utf8')
    # Supplied variants are existing bitstreams, not independent evaluation units.
    inventory=[]
    for row in supplied_manifest():
        x,sr=load_ref(row)
        label='语音' if row['kind']=='speech' else '音乐'
        for p in sorted((WORK/'C题'/'附件1').glob(label+'_*')):
            y,dsr,actual=decode(p)
            m=compare(x,y,sr,decoded_rate=dsr)
            inventory.append(dict(source_id=row['source_id'],filename=p.name,bytes=p.stat().st_size,
                decoded_rate=dsr,actual_decoder=actual,**scalar_metrics(m)))
    pd.DataFrame(inventory).to_csv(RESULT/'supplied_variant_audit.csv',index=False,encoding='utf8')
    print('saved all experiment CSV files',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    scope=parser.add_mutually_exclusive_group()
    scope.add_argument('--supplied-only',action='store_true')
    scope.add_argument('--public-only',action='store_true',help='Evaluate only the 90 official public excerpts.')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'runs'/'latest'/'results',help='New run output; recorded results are preserved.')
    args=parser.parse_args()
    main(args.supplied_only,args.public_only,args.output_dir)
