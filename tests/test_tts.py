import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import dubber
from dubber import get_supported_voices, speed_adjust_audio, get_audio_duration

class TTSQATests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_supported_voices(self):
        voices = get_supported_voices()
        self.assertIsInstance(voices, list)
        self.assertGreater(len(voices), 0)
        # Check that Vietnamese voices exist
        vn_voices = [v for v in voices if "vi" in str(v.get("locale", "")).lower() or "vi" in str(v.get("name", "")).lower()]
        self.assertGreater(len(vn_voices), 0)

    def test_speed_adjust_audio(self):
        # Create a silent audio wav file with pydub
        from pydub import AudioSegment
        sample_audio = AudioSegment.silent(duration=2000) # 2 seconds
        in_wav = os.path.join(self.temp_dir, "test_audio.wav")
        out_wav = os.path.join(self.temp_dir, "test_speed.wav")
        sample_audio.export(in_wav, format="wav")

        # Double speed factor = 2.0 -> duration should become approx 1.0s
        speed_adjust_audio(in_wav, out_wav, factor=2.0)
        self.assertTrue(os.path.exists(out_wav))
        new_dur = get_audio_duration(out_wav)
        self.assertAlmostEqual(new_dur, 1.0, delta=0.3)

    def test_generate_tts_censorship_bleep(self):
        # Test generation with censorship ***
        out_mp3 = os.path.join(self.temp_dir, "bleep_test.mp3")
        success = dubber.generate_tts("Đoạn này bị *** rồi nhé", voice="google-translate-vi", output_path=out_mp3)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(out_mp3))
        self.assertGreater(os.path.getsize(out_mp3), 0)

if __name__ == "__main__":
    unittest.main()
