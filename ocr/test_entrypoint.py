"""Synthetic process tests for the actual startup selector, with isolated executables."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class StartupTests(unittest.TestCase):
    def run_selector(self, selector=None, features='native-ocr', feature_status=0):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / 'noevia-ocr'
            binary.write_text(f'''#!/bin/sh
if [ "${{1-}}" = --features ]; then printf '%s\\n' '{features}'; exit {feature_status}; fi
printf 'rust-selected\\n'
''')
            binary.chmod(0o755)
            python = root / 'python'
            python.write_text('#!/bin/sh\nprintf "python-selected\\n"\n')
            python.chmod(0o755)
            script = (Path(__file__).parent / 'entrypoint.sh').read_text().replace('/usr/local/bin/noevia-ocr', str(binary))
            env = {'PATH': f'{root}:/usr/bin:/bin'}
            if selector is not None:
                env['NOEVIA_OCR_IMPL'] = selector
            return subprocess.run(['/bin/sh', '-c', script], env=env, capture_output=True, text=True, timeout=3)

    def test_unset_defaults_to_python(self):
        result = self.run_selector()
        self.assertEqual((result.returncode, result.stdout), (0, 'python-selected\n'))

    def test_explicit_python(self):
        result = self.run_selector('python')
        self.assertEqual((result.returncode, result.stdout), (0, 'python-selected\n'))

    def test_rust_executes_native_service(self):
        result = self.run_selector('rust')
        self.assertEqual((result.returncode, result.stdout), (0, 'rust-selected\n'))

    def test_unsupported_and_failed_feature_probe_refuse(self):
        for feature, status in [('', 0), ('other', 0), ('native-ocr-extra', 0), ('', 1), ('native-ocr', 1)]:
            with self.subTest(feature=feature, status=status):
                result = self.run_selector('rust', feature, status)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, '')

    def test_invalid_selector_refuses_without_fallback(self):
        for selector in ['', 'Rust', 'node', 'rust python']:
            with self.subTest(selector=selector):
                result = self.run_selector(selector)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, '')


if __name__ == '__main__':
    unittest.main()
