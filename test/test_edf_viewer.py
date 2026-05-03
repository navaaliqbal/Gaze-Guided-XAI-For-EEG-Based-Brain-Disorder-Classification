"""
Simple GUI tests for EDF Viewer application.
Tests basic workflow: open app → load EDF → start recording → save
"""
# cd "c:\Users\Qadri laptop\Downloads\64(1)\64" ; python -m pytest test\test_edf_viewer.py -v
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# ============================================================================
# MOCK EXTERNAL DEPENDENCIES
# ============================================================================
# Mock hardware dependencies
sys.modules['tobii_research'] = MagicMock()
sys.modules['tobiiresearch'] = MagicMock()
sys.modules['tobiiresearch.interop'] = MagicMock()
sys.modules['tobiiresearch.interop.interop'] = MagicMock()
sys.modules['tobiiresearch.interop.python3'] = MagicMock()
sys.modules['tobiiresearch.interop.python3.tobii_research_interop'] = MagicMock()
sys.modules['tobiiresearch.implementation'] = MagicMock()
sys.modules['tobiiresearch.internal'] = MagicMock()
sys.modules['pyaudio'] = MagicMock()
sys.modules['speech_recognition'] = MagicMock()
sys.modules['speech_thread'] = MagicMock()
sys.modules['speech_thread.SpeechRecognitionThread'] = MagicMock()

# ============================================================================
# IMPORTS
# ============================================================================
from PyQt5.QtWidgets import QApplication, QFileDialog

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_DIR)

from main_window_NMT import MainWindow

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def qapp():
    """Create QApplication for the test session."""
    app = QApplication.instance()
    if app is None:
        # Use offscreen platform to prevent GUI from displaying during tests
        app = QApplication(sys.argv + ['-platform', 'offscreen'])
    yield app


@pytest.fixture
def test_edf_files():
    """Get test EDF files from edf/ folder."""
    edf_dir = os.path.join(WORKSPACE_DIR, "edf")
    if os.path.exists(edf_dir):
        files = [os.path.join(edf_dir, f) for f in os.listdir(edf_dir) if f.endswith(".edf")]
        return sorted(files)
    return []


@pytest.fixture
def main_window(qapp):
    """Create MainWindow instance."""
    window = MainWindow()
    window.show()
    yield window
    window.close()


# ============================================================================
# SIMPLE TESTS - BASIC WORKFLOW
# ============================================================================

class TestBasicWorkflow:
    """Test the essential EDF Viewer workflow."""
    
    def test_1_app_opens(self, main_window):
        """Test that the app opens and displays."""
        assert main_window is not None
        assert main_window.isVisible()
        print("\n✅ App opened successfully")
    
    def test_2_load_edf_file(self, main_window, test_edf_files):
        """Test loading an EDF file."""
        if not test_edf_files:
            pytest.skip("No EDF files available in edf/ folder")
        
        edf_file = test_edf_files[0]
        
        # Mock file dialog and load EDF
        with patch.object(QFileDialog, 'getOpenFileName', return_value=(edf_file, '')):
            main_window.open_edf()
        
        # Verify file loaded
        assert len(main_window.canvas.signals) > 0
        assert main_window.current_edf_file == os.path.basename(edf_file)
        print(f"\n✅ Loaded EDF file: {os.path.basename(edf_file)}")
    
    def test_3_load_multiple_edf_files(self, main_window, test_edf_files):
        """Test loading multiple EDF files one after another."""
        if len(test_edf_files) < 2:
            pytest.skip("Need at least 2 EDF files")
        
        # Load first file
        with patch.object(QFileDialog, 'getOpenFileName', return_value=(test_edf_files[0], '')):
            main_window.open_edf()
        assert len(main_window.canvas.signals) > 0
        first_file = main_window.current_edf_file
        
        # Load second file
        with patch.object(QFileDialog, 'getOpenFileName', return_value=(test_edf_files[1], '')):
            main_window.open_edf()
        assert len(main_window.canvas.signals) > 0
        second_file = main_window.current_edf_file
        
        assert first_file != second_file
        print(f"\n✅ Loaded multiple EDF files: {first_file}, {second_file}")
    
    def test_4_start_recording(self, main_window, test_edf_files):
        """Test starting a recording session."""
        if not test_edf_files:
            pytest.skip("No EDF files available")
        
        # Load EDF first
        edf_file = test_edf_files[0]
        with patch.object(QFileDialog, 'getOpenFileName', return_value=(edf_file, '')):
            main_window.open_edf()
        
        # Start recording
        main_window.start_session_recording()
        
        assert main_window.session_recording_active == True
        assert main_window.session_data == []
        print("\n✅ Recording started")
    
    def test_5_stop_recording(self, main_window, test_edf_files):
        """Test stopping a recording session."""
        if not test_edf_files:
            pytest.skip("No EDF files available")
        
        # Load EDF and start recording
        edf_file = test_edf_files[0]
        with patch.object(QFileDialog, 'getOpenFileName', return_value=(edf_file, '')):
            main_window.open_edf()
        
        main_window.start_session_recording()
        assert main_window.session_recording_active == True
        
        # Stop recording
        with patch("builtins.open", create=True):
            with patch("json.dump"):
                main_window.stop_session_recording()
        
        assert main_window.session_recording_active == False
        print("\n✅ Recording stopped")
    
    def test_6_save_recording(self, main_window, test_edf_files):
        """Test saving a recording."""
        if not test_edf_files:
            pytest.skip("No EDF files available")
        
        # Load EDF
        edf_file = test_edf_files[0]
        with patch.object(QFileDialog, 'getOpenFileName', return_value=(edf_file, '')):
            main_window.open_edf()
        
        # Start recording
        main_window.start_session_recording()
        
        # Add some dummy data
        main_window.session_data.append({"test": "data"})
        
        # Save recording
        with patch("builtins.open", create=True):
            with patch("json.dump"):
                main_window.save_session_data()
        
        assert hasattr(main_window, 'session_name')
        print("\n✅ Recording saved")
    
    def test_7_toggle_recording_action(self, main_window):
        """Test toggling recording through UI action."""
        # Start via toggle
        main_window.toggle_session_recording(True)
        assert main_window.session_recording_active == True
        
        # Stop via toggle
        main_window.toggle_session_recording(False)
        assert main_window.session_recording_active == False
        print("\n✅ Recording toggle works")
    
    def test_8_window_has_menu(self, main_window):
        """Test that window has menu bar."""
        menubar = main_window.menuBar()
        assert menubar is not None
        
        menu_titles = [action.text() for action in menubar.actions()]
        assert len(menu_titles) > 0
        print(f"\n✅ Menu bar exists with {len(menu_titles)} menus")
    
    def test_9_window_has_canvas(self, main_window):
        """Test that main window has canvas widget."""
        assert hasattr(main_window, 'canvas')
        assert main_window.canvas is not None
        print("\n✅ Canvas widget exists")
    
    def test_10_gaze_tracking_toggle(self, main_window):
        """Test toggling gaze tracking on and off."""
        # Enable gaze tracking
        with patch('gaze_tracking.GazeTrackingThread') as MockThread:
            mock_thread = MagicMock()
            MockThread.return_value = mock_thread
            main_window.toggle_gaze_tracking(True)
        
        assert main_window.gaze_tracking_active == True
        
        # Disable gaze tracking
        main_window.toggle_gaze_tracking(False)
        assert main_window.gaze_tracking_active == False
        print("\n✅ Gaze tracking toggle works")
    
    def test_11_complete_workflow(self, main_window, test_edf_files):
        """Test complete workflow: open → load → record → save."""
        if not test_edf_files:
            pytest.skip("No EDF files available")
        
        edf_file = test_edf_files[0]
        
        # 1. Load EDF file
        with patch.object(QFileDialog, 'getOpenFileName', return_value=(edf_file, '')):
            main_window.open_edf()
        assert len(main_window.canvas.signals) > 0
        
        # 2. Start recording
        main_window.start_session_recording()
        assert main_window.session_recording_active == True
        
        # 3. Simulate some data
        main_window.session_data.append({"timestamp": 123, "data": "test"})
        
        # 4. Stop and save
        with patch("builtins.open", create=True):
            with patch("json.dump"):
                main_window.stop_session_recording()
        
        assert main_window.session_recording_active == False
        print("\n✅ Complete workflow tested successfully")


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("EDF VIEWER - SIMPLE GUI TESTS")
    print("="*70)
    pytest.main([__file__, "-v", "-s"])
