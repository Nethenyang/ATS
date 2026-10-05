"""Integration checks for the public repository's offline workflows."""
from pathlib import Path
import json,subprocess,sys,tempfile,unittest

ROOT=Path(__file__).resolve().parents[1]

class RepositoryWorkflows(unittest.TestCase):
    def invoke(self,script,args,cwd):
        result=subprocess.run([sys.executable,str(ROOT/script),*map(str,args)],cwd=cwd,capture_output=True,text=True,encoding='utf8',errors='replace')
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        return result

    def test_record_verification_needs_no_audio_or_manuscript(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'verification.json'
            self.invoke('verify_results.py',['--output',output],directory)
            report=json.loads(output.read_text(encoding='utf8'))
            self.assertEqual(report['status'],'PASS')
            self.assertEqual(report['candidate_files'],3628)
            self.assertEqual(report['sources'],92)
            self.assertEqual(report['local_candidate_hashes_verified'],0)

    def test_tables_and_public_plots_reproduce_from_records_alone(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'analysis'
            self.invoke('make_report.py',['--output-dir',output],directory)
            for name in ['policy_table.tex','supplied_table.tex']:
                self.assertEqual((output/'tables'/name).read_text(encoding='utf8'),(ROOT/'artifacts/recorded/tables'/name).read_text(encoding='utf8'))
            recorded=json.loads((ROOT/'results/recorded/summary.json').read_text(encoding='utf8'))
            recreated=json.loads((output/'summary.json').read_text(encoding='utf8'))
            self.assertEqual(recreated['public'],recorded['public'])
            for name in ['public_outcomes.pdf','threshold_sensitivity.pdf','supplied_rate_fidelity.pdf']:
                self.assertTrue((output/'figures'/name).is_file(),name)
            self.assertFalse((output/'figures/temporal_case.pdf').exists())

    def test_new_experiment_output_and_public_scope_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            result=self.invoke('run_experiments.py',['--help'],directory)
            self.assertIn('--public-only',result.stdout)
            self.assertIn('--output-dir',result.stdout)

if __name__=='__main__': unittest.main()
