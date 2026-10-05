"""Reference-aware feasibility experiments and reference-free real-clip diagnostics.
These results do not support a new denoising-method claim in the main paper.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal, ndimage
from audio_pipeline import compare
from run_experiments import supplied_manifest,load_ref,ROOT,WORK

OUT=ROOT/'runs'/'supplementary'; OUT.mkdir(parents=True,exist_ok=True)

def denoise_channel(y,sr):
    window=2**int(np.ceil(np.log2(.032*sr)))
    hop=window//4
    _,_,Y=signal.stft(y,sr,nperseg=window,noverlap=window-hop,boundary='zeros',padded=True)
    power=np.abs(Y)**2
    frame_power=power.sum(axis=0)
    quiet=frame_power<=np.quantile(frame_power,.2)
    noise=np.median(power[:,quiet],axis=1)/np.log(2)
    local=ndimage.uniform_filter(power,size=(1,3),mode='nearest')
    gain=np.maximum(1-noise[:,None]/np.maximum(local,1e-15),.15)
    gain=ndimage.uniform_filter(gain,size=(3,3),mode='nearest')
    _,out=signal.istft(Y*gain,sr,nperseg=window,noverlap=window-hop,boundary=True)
    return out[:len(y)]

def noise_signal(kind,size,sr,rng):
    if kind=='white':
        n=rng.standard_normal(size)
    elif kind=='hum':
        t=np.arange(size)/sr
        n=sum(np.sin(2*np.pi*f*t+rng.uniform(0,2*np.pi))/(i+1) for i,f in enumerate([50,100,150]))
        n+=.1*rng.standard_normal(size)
    else:
        n=.25*rng.standard_normal(size)
        count=max(1,int(size/sr*15))
        indices=rng.choice(size-128,count,replace=False)
        for i in indices:
            n[i:i+128]+=rng.normal(0,6)*signal.windows.hann(128)
    return n

def main():
    sources=supplied_manifest()
    if not sources:
        raise FileNotFoundError('Optional original competition references are required in data/C题/附件1.')
    results=[]
    for row in sources:
        x,sr=load_ref(row)
        for kind in ['white','hum','impulsive']:
            for db in [0,5,10]:
                for seed in range(5):
                    rng=np.random.default_rng(10000+seed)
                    n=noise_signal(kind,len(x),sr,rng)
                    n*=np.sqrt(np.sum(x*x)/np.sum(n*n)*10**(-db/10))
                    y=x+n
                    z=denoise_channel(y,sr)
                    before=compare(x,y,sr,align=False)
                    after=compare(x,z,sr,align=False)
                    results.append(dict(source_id=row['source_id'],noise=kind,input_snr_db=db,seed=seed,
                        measured_input_snr_db=before['snr_db'],output_snr_db=after['snr_db'],
                        improvement_db=after['snr_db']-before['snr_db']))
    pd.DataFrame(results).to_csv(OUT/'synthetic_denoising_validation.csv',index=False)
    diagnostics=[]
    for p in sorted((WORK/'C题'/'附件2').glob('*.wav')):
        x,sr=sf.read(p,dtype='float64',always_2d=True)
        z=np.stack([denoise_channel(x[:,i],sr) for i in range(x.shape[1])],axis=1)
        output=OUT/(p.stem+'_denoised.wav')
        overrange=int(np.count_nonzero(np.abs(z)>=1))
        scale=min(1.,.999/max(float(np.max(np.abs(z))),1e-12))
        z*=scale
        sf.write(output,z,sr,subtype='PCM_16')
        mono=x.mean(axis=1)
        f,psd=signal.welch(mono,sr,nperseg=8192)
        peaks,_=signal.find_peaks(10*np.log10(np.maximum(psd,1e-20)),prominence=10)
        ordered=sorted(peaks,key=lambda i:psd[i],reverse=True)[:8]
        rms_in=float(np.sqrt(np.mean(x*x))); rms_out=float(np.sqrt(np.mean(z*z)))
        diagnostics.append(dict(file=p.name,sr=sr,channels=x.shape[1],duration=len(x)/sr,
            input_rms=rms_in,output_rms=rms_out,attenuation_db=20*np.log10(rms_out/rms_in),
            narrowband_peak_hz=[float(f[i]) for i in ordered],pre_limit_overrange_samples=overrange,
            peak_safety_scale=scale,clipped_output_samples=int(np.count_nonzero(np.abs(z)>=1)),
            true_snr='unavailable: no supplied clean reference',interpretation='peaks may be noise or wanted tonal content; attenuation does not prove improvement'))
    (OUT/'real_clip_diagnostics.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf8')
    print(pd.DataFrame(results).groupby(['source_id','noise'])['improvement_db'].mean().round(3).to_string())
    print(json.dumps(diagnostics,indent=2))

if __name__=='__main__':
    main()
