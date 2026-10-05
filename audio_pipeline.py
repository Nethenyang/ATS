"""Actual audio coding and conservative full-reference fidelity evaluation."""
from pathlib import Path
from math import gcd
from fractions import Fraction
import time
import numpy as np
import soundfile as sf
from scipy import signal
import av

def resample(x, old_rate, new_rate):
    if old_rate == new_rate:
        return np.asarray(x, dtype=np.float64)
    factor = gcd(int(old_rate), int(new_rate))
    return signal.resample_poly(x, new_rate//factor, old_rate//factor).astype(np.float64)

def pad_reference_support(y, n, delay=0):
    result = np.zeros(n, dtype=np.float64)
    start_ref = max(0, -delay)
    start_y = max(0, delay)
    size = min(n-start_ref, len(y)-start_y)
    if size > 0:
        result[start_ref:start_ref+size] = y[start_y:start_y+size]
    return result

def estimate_delay(x, y, sr, maximum_ms=200):
    # Bound correlation lag; one constant delay per whole recording.
    xx = np.asarray(x, dtype=np.float32)
    yy = np.asarray(y, dtype=np.float32)
    xx = xx - xx.mean()
    yy = yy - yy.mean()
    corr = signal.correlate(yy, xx, mode='full', method='fft')
    origin = len(xx)-1
    maximum = int(sr*maximum_ms/1000)
    lo, hi = max(0,origin-maximum), min(len(corr),origin+maximum+1)
    return int(lo+np.argmax(corr[lo:hi])-origin)

def fidelity_arrays(x, y, sr, frame_ms=20, activity_db=-40, tail_fraction=.1):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    error = (x-y)**2
    power = x**2
    epsilon = 1e-15
    normalized = float(error.sum()/max(float(power.sum()),epsilon))
    snr = -10*np.log10(max(normalized,1e-12))
    frame = max(1,round(sr*frame_ms/1000))
    starts = np.arange(0,len(x),frame)
    powers = np.add.reduceat(power, starts)
    errors = np.add.reduceat(error, starts)
    lengths = np.minimum(frame,len(x)-starts)
    average_power = powers/lengths
    active = average_power >= average_power.max()*10**(activity_db/10)
    distortion = errors[active]/np.maximum(powers[active],epsilon)
    k = max(1,int(np.ceil(tail_fraction*len(distortion))))
    tail = float(np.sort(distortion)[-k:].mean())
    return dict(snr_db=float(snr),tail_snr_db=float(-10*np.log10(max(tail,1e-12))),
        distortion=normalized,tail_distortion=tail,active_frames=int(active.sum()),
        frame_distortions=distortion,frame_powers=average_power,frame_active=active,
        frame_snr_db=-10*np.log10(np.maximum(errors/np.maximum(powers,epsilon),1e-12)))

def compare(x, y, sr, decoded_rate=None, align=True, frame_ms=20, activity_db=-40, tail_fraction=.1):
    x = np.asarray(x,dtype=np.float64)
    if decoded_rate is not None:
        y = resample(y,decoded_rate,sr)
    y = np.asarray(y,dtype=np.float64)
    raw_y = pad_reference_support(y,len(x))
    raw_snr = -10*np.log10(max(float(np.sum((x-raw_y)**2)/max(np.sum(x*x),1e-15)),1e-12))
    delay = estimate_delay(x,y,sr) if align else 0
    aligned = pad_reference_support(y,len(x),delay)
    result = fidelity_arrays(x,aligned,sr,frame_ms,activity_db,tail_fraction)
    result.update(raw_snr_db=float(raw_snr),delay_samples=delay,delay_ms=1000*delay/sr,
        aligned=aligned,decoded_samples=len(y),reference_samples=len(x))
    return result

def choose(rows,global_db=20,tail_db=15):
    eligible = [r for r in rows if np.isfinite(r['snr_db']) and r['snr_db']>=global_db
        and (tail_db is None or (np.isfinite(r['tail_snr_db']) and r['tail_snr_db']>=tail_db))]
    return min(eligible,key=lambda r:(r['bytes'],-r['tail_snr_db'],r['config'])) if eligible else None

def encode(x, source_rate, config, path):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    rate, codec = config['rate'],config['codec']
    samples = resample(x,source_rate,rate).astype(np.float32)
    if codec.startswith('wav'):
        subtype = 'FLOAT' if codec=='wavfloat' else {8:'PCM_U8',16:'PCM_16',24:'PCM_24'}[config['depth']]
        sf.write(str(path),samples,rate,subtype=subtype)
        return
    encoder = {'mp3':'libmp3lame','aac':'aac','opus':'libopus'}[codec]
    container = {'mp3':'mp3','aac':'adts','opus':'ogg'}[codec]
    with av.open(str(path),'w',format=container) as output:
        stream = output.add_stream(encoder,rate=rate)
        stream.layout = 'mono'
        stream.bit_rate = config['bitrate']
        stream.codec_context.format = 'fltp' if codec!='opus' else 'flt'
        if codec=='opus':
            stream.options = {'vbr':'off','application':'audio'}
        for start in range(0,len(samples),4096):
            chunk = samples[start:start+4096]
            frame = av.AudioFrame.from_ndarray(chunk[None,:],format='fltp' if codec!='opus' else 'flt',layout='mono')
            frame.sample_rate = rate
            frame.pts = start
            frame.time_base = Fraction(1,rate)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode(None):
            output.mux(packet)

def decode(path):
    chunks=[]
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        rate = stream.codec_context.sample_rate
        actual = stream.codec_context.name
        resampler = av.AudioResampler(format='fltp',layout='mono',rate=rate)
        for frame in container.decode(stream):
            for f in resampler.resample(frame):
                chunks.append(f.to_ndarray()[0])
        for f in resampler.resample(None):
            chunks.append(f.to_ndarray()[0])
    if not chunks:
        raise ValueError('empty decoded audio')
    return np.concatenate(chunks).astype(np.float64),rate,actual
