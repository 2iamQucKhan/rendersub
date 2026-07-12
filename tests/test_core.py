import tempfile
import unittest
from pathlib import Path

import downloader
import dubber
import transcriber


class DownloaderTests(unittest.TestCase):
    def test_download_video_rejects_invalid_url(self):
        with self.assertRaises(ValueError):
            downloader.download_video("not-a-url", "")


class TranscriberDataTests(unittest.TestCase):
    def test_prompt_and_history_json_are_lists_and_limited(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = transcriber.DATA_DIR
            old_prompts = transcriber.CUSTOM_PROMPTS_FILE
            old_history = transcriber.SCRIPT_HISTORY_FILE
            try:
                transcriber.DATA_DIR = Path(tmp) / "Data"
                transcriber.CUSTOM_PROMPTS_FILE = transcriber.DATA_DIR / "custom_prompts.json"
                transcriber.SCRIPT_HISTORY_FILE = transcriber.DATA_DIR / "script_history.json"

                transcriber.save_custom_prompts({"bad": "shape"})
                self.assertEqual(transcriber.load_custom_prompts(), [])

                for idx in range(transcriber.HISTORY_LIMIT + 5):
                    transcriber.add_script_history(f"title {idx}", "hello world", 1)
                self.assertEqual(len(transcriber.load_script_history()), transcriber.HISTORY_LIMIT)
            finally:
                transcriber.DATA_DIR = old_data_dir
                transcriber.CUSTOM_PROMPTS_FILE = old_prompts
                transcriber.SCRIPT_HISTORY_FILE = old_history

    def test_srt_time_format_and_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.srt"
            transcriber.export_srt_with_silence(
                [{"index": 1, "text": "Xin chao", "duration": 1.25}],
                out,
                silence_between=0.5,
            )
            data = out.read_text(encoding="utf-8")
            self.assertIn("00:00:00,000 --> 00:00:01,250", data)
            self.assertIn("Xin chao", data)

    def test_script_split_removes_production_notes(self):
        script = """
        Hook: Đừng lướt qua video này.
        [Nhạc nền căng thẳng]
        00:00 - 00:03 Câu chuyện bắt đầu từ một chi tiết rất nhỏ, nhưng nó khiến mọi thứ đổi hướng hoàn toàn, và không ai kịp nhận ra điều đó.
        (zoom cận mặt)
        Đây là phần lời đọc thật.
        """
        segments = transcriber.split_script_to_sentences(script, max_chars=90)
        texts = [seg["text"] for seg in segments]
        joined = " ".join(texts)
        self.assertNotIn("Hook", joined)
        self.assertNotIn("Nhạc nền", joined)
        self.assertNotIn("zoom", joined)
        self.assertTrue(all(len(text) <= 95 for text in texts))
        self.assertIn("Đừng lướt qua video này.", joined)
        self.assertIn("Đây là phần lời đọc thật.", joined)


class DubberTests(unittest.TestCase):
    def test_format_ass_time_rounding(self):
        self.assertEqual(dubber.format_ass_time(61.234), "0:01:01.23")

    def test_missing_video_duration_is_zero(self):
        self.assertEqual(dubber.get_video_duration_ms("missing-file.mp4"), 0)


if __name__ == "__main__":
    unittest.main()
