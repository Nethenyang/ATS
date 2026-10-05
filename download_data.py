"""Download official public audio releases and prepare a deterministic subset.
No source audio is redistributed in the manuscript source package.
"""
from pathlib import Path
import hashlib, io, json, tarfile, zipfile, time
import requests
import numpy as np
import soundfile as sf
import av

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
ARCH = DATA / 'archives'
REF = DATA / 'references'
for p in [ARCH, REF]:
    p.mkdir(parents=True, exist_ok=True)

def download(url, target):
    if target.exists():
        print('cached', target.name, flush=True)
        return
    tmp = target.with_suffix(target.suffix + '.partial')
    with requests.get(url, stream=True, timeout=(30, 90)) as response:
        response.raise_for_status()
        expected = int(response.headers.get('Content-Length', 0))
        count, last = 0, time.monotonic()
        with tmp.open('wb') as f:
            for chunk in response.iter_content(1024 * 1024):
                f.write(chunk)
                count += len(chunk)
                if time.monotonic() - last > 10:
                    print(target.name, round(count / 1e6, 1), 'MB', flush=True)
                    last = time.monotonic()
    if expected and count != expected:
        raise RuntimeError('incomplete download')
    tmp.replace(target)
    print('downloaded', target.name, count, flush=True)

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def decode_mixture(payload):
    parts = []
    with av.open(io.BytesIO(payload)) as c:
        stream = c.streams.audio[0]
        rate = stream.codec_context.sample_rate
        resampler = av.AudioResampler(format='fltp', layout='mono', rate=rate)
        for frame in c.decode(stream):
            # Arithmetic stereo mean rather than FFmpeg's center-pan coefficient.
            array = frame.to_ndarray()
            if frame.format.is_planar:
                mono = array.mean(axis=0)
            else:
                mono = array.reshape(-1, len(frame.layout.channels)).mean(axis=1)
            parts.append(mono.astype(np.float32))
    return np.concatenate(parts), rate

def prepare():
    manifest = []
    speech_url = 'https://www.openslr.org/resources/12/dev-clean.tar.gz'
    music_url = 'https://github.com/sigsep/sigsep-mus-db/releases/download/v0.4.0/MUSDB18-7-STEMS.zip'
    download(speech_url, ARCH / 'dev-clean.tar.gz')
    download(music_url, ARCH / 'MUSDB18-7-STEMS.zip')
    with tarfile.open(ARCH / 'dev-clean.tar.gz', 'r:gz') as archive:
        eligible = []
        for m in archive.getmembers():
            if m.isfile() and m.name.endswith('.flac'):
                eligible.append(m)
        speakers = sorted({m.name.split('/')[-3] for m in eligible}, key=int)
        for speaker in speakers:
            files = sorted([m for m in eligible if m.name.split('/')[-3] == speaker], key=lambda m:m.name)
            chosen = None
            for member in files:
                payload = archive.extractfile(member).read()
                x, sr = sf.read(io.BytesIO(payload), dtype='float32')
                if len(x) >= 6 * sr:
                    chosen = (member, x[:6 * sr], sr)
                    break
            if chosen is None:
                continue
            member, x, sr = chosen
            sid = 'libri_' + Path(member.name).stem
            out = REF / (sid + '.wav')
            sf.write(out, x, sr, subtype='FLOAT')
            manifest.append(dict(source_id=sid, group='public_speech', kind='speech', independent_unit=speaker,
                original_member=member.name, original_url=speech_url, reference_path=out.relative_to(ROOT).as_posix(),
                rate=sr, duration=len(x)/sr, sha256=digest(out), selection='first lexicographic utterance >=6 s per numeric-sorted speaker; first6s'))
    with zipfile.ZipFile(ARCH / 'MUSDB18-7-STEMS.zip') as archive:
        members = sorted(m for m in archive.namelist() if m.startswith('test/') and m.endswith('.mp4'))
        for i, member in enumerate(members):
            x, sr = decode_mixture(archive.read(member))
            x = x[:7 * sr]
            sid = 'musdb_' + f'{i:02d}'
            out = REF / (sid + '.wav')
            sf.write(out, x, sr, subtype='FLOAT')
            manifest.append(dict(source_id=sid, group='public_music', kind='music', independent_unit=Path(member).stem,
                original_member=member, original_url=music_url, reference_path=out.relative_to(ROOT).as_posix(),
                rate=sr, duration=len(x)/sr, sha256=digest(out), selection='all test tracks; mixture stream0; mono arithmetic mean; first7s'))
    (DATA / 'public_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    (DATA / 'download_provenance.json').write_text(json.dumps({
        'archives':[{ 'file':p.name, 'bytes':p.stat().st_size, 'sha256':digest(p)} for p in ARCH.iterdir() if not p.name.endswith('.partial')],
        'notes':'LibriSpeech CC BY4.0. MUSDB18 official academic sample release; mixture already AAC-compressed. No audio redistribution.'
    }, indent=2), encoding='utf-8')
    print('prepared', {g:sum(m['group']==g for m in manifest) for g in ['public_speech','public_music']}, flush=True)

if __name__ == '__main__':
    prepare()
