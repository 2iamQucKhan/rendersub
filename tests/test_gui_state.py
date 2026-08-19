import os
import sys
import unittest
from PyQt6.QtWidgets import QApplication

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import MainWindow

class GUIStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        self.window = MainWindow()

    def tearDown(self):
        self.window.close()

    def test_initial_state_idle(self):
        # When no video is running: Cancel button should be disabled
        self.assertFalse(self.window.btn_cancel_main.isEnabled())

    def test_batch_buttons_initial_state(self):
        self.assertTrue(self.window.btn_run_batch.isEnabled() or not self.window.batch_queue)
        self.assertFalse(self.window.btn_stop_batch.isEnabled())

    def test_tts_checkbox_controls_voice_combo(self):
        # Toggling TTS checkbox should enable/disable voice dropdown
        self.window.chk_enable_dubbing.setChecked(False)
        self.assertFalse(self.window.cb_voice.isEnabled())

        self.window.chk_enable_dubbing.setChecked(True)
        self.assertTrue(self.window.cb_voice.isEnabled())

if __name__ == "__main__":
    unittest.main()
