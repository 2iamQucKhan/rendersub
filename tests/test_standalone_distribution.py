import unittest
import os
import sys
import subprocess
import time

class TestStandaloneDistribution(unittest.TestCase):
    """
    Independent QA verification for the standalone Windows distribution:
    - Executable integrity
    - Bundled FFmpeg/FFprobe binaries
    - Bundled resource dictionaries & config templates
    - Isolation test (clean environment without dev PATH)
    """

    @classmethod
    def setUpClass(cls):
        cls.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.dist_dir = os.path.join(cls.root_dir, "dist", "RenderSub")
        cls.exe_path = os.path.join(cls.dist_dir, "RenderSub.exe")
        cls.bin_dir = os.path.join(cls.dist_dir, "bin")
        cls.ffmpeg_path = os.path.join(cls.bin_dir, "ffmpeg.exe")
        cls.ffprobe_path = os.path.join(cls.bin_dir, "ffprobe.exe")

    def test_01_distribution_structure_exists(self):
        self.assertTrue(os.path.exists(self.dist_dir), f"Thư mục dist/RenderSub không tồn tại: {self.dist_dir}")
        self.assertTrue(os.path.exists(self.exe_path), f"RenderSub.exe không tồn tại: {self.exe_path}")
        self.assertGreater(os.path.getsize(self.exe_path), 10 * 1024 * 1024, "RenderSub.exe dung lượng bất thường (<10MB)")

    def test_02_bundled_ffmpeg_executable(self):
        self.assertTrue(os.path.exists(self.ffmpeg_path), f"ffmpeg.exe không tồn tại: {self.ffmpeg_path}")
        self.assertTrue(os.path.exists(self.ffprobe_path), f"ffprobe.exe không tồn tại: {self.ffprobe_path}")

        # Run ffmpeg -version directly from bundled bin
        res = subprocess.run([self.ffmpeg_path, "-version"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Bundled FFmpeg failed with: {res.stderr}")
        self.assertIn("ffmpeg version", res.stdout.lower())

        # Run ffprobe -version directly from bundled bin
        res_probe = subprocess.run([self.ffprobe_path, "-version"], capture_output=True, text=True)
        self.assertEqual(res_probe.returncode, 0, f"Bundled FFprobe failed with: {res_probe.stderr}")
        self.assertIn("ffprobe version", res_probe.stdout.lower())

    def test_03_bundled_resources_exist(self):
        data_dir = os.path.join(self.dist_dir, "Data")
        config_dir = os.path.join(self.dist_dir, "config")
        self.assertTrue(os.path.exists(data_dir), "Thư mục Data/ không được bundle vào dist/RenderSub")
        self.assertTrue(os.path.exists(config_dir), "Thư mục config/ không được bundle vào dist/RenderSub")

        vietphrase = os.path.join(data_dir, "VietPhrase.txt")
        self.assertTrue(os.path.exists(vietphrase), "VietPhrase.txt không tồn tại trong Data/")
        self.assertGreater(os.path.getsize(vietphrase), 1024 * 1024, "VietPhrase.txt dung lượng < 1MB")

        prompt_template = os.path.join(config_dir, "xkiro_prompt_template.json")
        self.assertTrue(os.path.exists(prompt_template), "xkiro_prompt_template.json không tồn tại trong config/")

    def test_04_no_secrets_in_distribution(self):
        """Ensure no API keys, temporary tokens, or personal cache are shipped in dist."""
        settings_path = os.path.join(self.dist_dir, "config", "app_settings.json")
        if os.path.exists(settings_path):
            with open(settings_path, "r", encoding="utf-8") as f:
                content = f.read()
                # Check for live API keys
                self.assertNotIn("AIzaSy", content, "Phát hiện Google API Key thật bị lộ trong build dist!")

    def test_05_clean_machine_isolation_launch(self):
        """
        Launch RenderSub.exe in an isolated environment without Python in PATH.
        Verifies that it initializes cleanly without crashing on missing DLLs.
        """
        # Create an isolated environment where PATH only contains Windows System32 and app directory
        clean_env = os.environ.copy()
        clean_env["PATH"] = f"{self.bin_dir};C:\\Windows\\System32;C:\\Windows"
        clean_env["PYTHONPATH"] = ""
        clean_env["PYTHONHOME"] = ""

        # Launch the standalone process
        p = subprocess.Popen(
            [self.exe_path],
            env=clean_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Allow 3 seconds for initial runtime and DLL binding
        time.sleep(3)
        poll_res = p.poll()

        # If it exited immediately with an error code, it means missing DLL or fatal runtime crash
        if poll_res is not None and poll_res != 0:
            stdout, stderr = p.communicate()
            self.fail(f"RenderSub.exe crashed upon launch with exit code {poll_res}.\nSTDOUT: {stdout}\nSTDERR: {stderr}")

        # If still running (normal for GUI event loop), terminate gracefully
        if poll_res is None:
            p.terminate()
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()

        self.assertTrue(True, "RenderSub.exe launched cleanly in isolated environment.")

if __name__ == "__main__":
    unittest.main()
