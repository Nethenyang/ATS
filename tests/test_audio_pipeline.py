import unittest
import numpy as np
import tempfile
from pathlib import Path
from audio_pipeline import compare, choose, encode, decode

class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.sr = 16000
        self.x = np.random.default_rng(7).normal(0, .1, self.sr).astype(np.float32)

    def test_known_delay_is_removed_without_losing_reference_support(self):
        y = np.r_[np.zeros(351), self.x]
        m = compare(self.x, y, self.sr)
        self.assertEqual(m['delay_samples'], 351)
        self.assertGreater(m['snr_db'], 100)
        self.assertLess(m['raw_snr_db'], 0)

    def test_gain_error_remains_a_real_distortion(self):
        m = compare(self.x, self.x * .5, self.sr)
        self.assertAlmostEqual(m['snr_db'], 6.0205999, places=5)

    def test_low_rate_aac_priming_can_exceed_100_ms(self):
        y = np.r_[np.zeros(2048), self.x]
        m = compare(self.x,y,self.sr)
        self.assertEqual(m['delay_samples'],2048)
        self.assertGreater(m['snr_db'],100)

    def test_truncation_counts_as_error(self):
        m = compare(self.x, self.x[:-self.sr//10], self.sr)
        self.assertLess(m['snr_db'], 12)
        self.assertLess(m['tail_snr_db'], 1)

    def test_tail_constraint_catches_error_in_quiet_active_frames(self):
        x = np.r_[np.ones(14400)*.1, np.ones(1600)*.002].astype(np.float32)
        y = x.copy(); y[-1600:] = 0
        m = compare(x, y, self.sr, align=False)
        self.assertGreater(m['snr_db'], 40)
        self.assertLess(m['tail_snr_db'], 1)
        candidates = [dict(config='bad',bytes=10,**m),dict(config='good',bytes=20,snr_db=90,tail_snr_db=90)]
        self.assertEqual(choose(candidates,20,None)['config'],'bad')
        self.assertEqual(choose(candidates,20,15)['config'],'good')

    def test_no_feasible_candidate_is_explicit(self):
        self.assertIsNone(choose([dict(config='bad',bytes=1,snr_db=2,tail_snr_db=1)],20,15))

    def test_actual_mp3_aac_opus_and_float_roundtrip(self):
        t = np.arange(self.sr*2)/self.sr
        x = (.2*np.sin(2*np.pi*443*t)+.1*np.sin(2*np.pi*721*t)).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            for codec in ['mp3','aac','opus','wavfloat']:
                config = dict(codec=codec,rate=self.sr if codec!='opus' else 48000,bitrate=64000,depth=32)
                path = Path(directory)/(codec + {'mp3':'.mp3','aac':'.aac','opus':'.ogg','wavfloat':'.wav'}[codec])
                encode(x,self.sr,config,path)
                y,sr,actual = decode(path)
                m = compare(x,y,sr if sr==self.sr else self.sr, decoded_rate=sr)
                self.assertGreater(m['snr_db'], 15, codec)
                self.assertIn(actual, {'mp3float','mp3','aac','opus','pcm_f32le'})

if __name__ == '__main__':
    unittest.main()
