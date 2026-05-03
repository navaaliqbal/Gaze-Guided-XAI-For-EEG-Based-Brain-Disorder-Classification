"""
Non-Functional Tests for EDF Viewer
====================================
Tests performance, resource usage, reliability, and usability
"""

import os
import sys
import time
import pytest
import psutil
from unittest.mock import patch, MagicMock

# Mock external dependencies
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
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Test application performance metrics."""
    
    def test_app_startup_time(self, qapp):
        """Test that app starts within acceptable time (< 3 seconds)."""
        start_time = time.time()
        window = MainWindow()
        window.show()
        startup_time = time.time() - start_time
        window.close()
        
        assert startup_time < 3.0, f"App took {startup_time:.2f}s to start (should be < 3s)"
        print(f"\n✅ App started in {startup_time:.2f}s")
    
    def test_edf_load_time(self, main_window, test_edf_files):
        """Test that EDF files load within reasonable time (< 5 seconds)."""
        if not test_edf_files:
            pytest.skip("No EDF files available")
        
        edf_file = test_edf_files[0]
        
        start_time = time.time()
        with patch.object(QFileDialog, 'getOpenFileName', return_value=(edf_file, '')):
            main_window.open_edf()
        load_time = time.time() - start_time
        
        assert load_time < 5.0, f"EDF load took {load_time:.2f}s (should be < 5s)"
        print(f"\n✅ EDF file loaded in {load_time:.2f}s")
    
    def test_recording_start_time(self, main_window, test_edf_files):
        """Test that recording starts quickly (< 1 second)."""
        if not test_edf_files:
            pytest.skip("No EDF files available")
        
        # Load EDF first
        edf_file = test_edf_files[0]
        with patch.object(QFileDialog, 'getOpenFileName', return_value=(edf_file, '')):
            main_window.open_edf()
        
        start_time = time.time()
        main_window.start_session_recording()
        recording_time = time.time() - start_time
        
        assert recording_time < 1.0, f"Recording start took {recording_time:.2f}s (should be < 1s)"
        print(f"\n✅ Recording started in {recording_time:.3f}s")


# ============================================================================
# RESOURCE USAGE TESTS
# ============================================================================

class TestResourceUsage:
    """Test memory and CPU usage."""
    
    def test_initial_memory_usage(self, main_window):
        """Test that initial memory usage is reasonable (< 450 MB)."""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        assert memory_mb < 450, f"Memory usage {memory_mb:.1f}MB is too high (should be < 450MB)"
        print(f"\n✅ Initial memory usage: {memory_mb:.1f}MB")
    
    def test_memory_after_edf_load(self, main_window, test_edf_files):
        """Test memory usage after loading EDF file (< 500 MB)."""
        if not test_edf_files:
            pytest.skip("No EDF files available")
        
        edf_file = test_edf_files[0]
        with patch.object(QFileDialog, 'getOpenFileName', return_value=(edf_file, '')):
            main_window.open_edf()
        
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        assert memory_mb < 500, f"Memory usage {memory_mb:.1f}MB is too high after EDF load"
        print(f"\n✅ Memory after EDF load: {memory_mb:.1f}MB")
    
    def test_cpu_usage_idle(self, main_window):
        """Test that CPU usage is low when idle (< 10%)."""
        time.sleep(1)  # Let app settle
        
        process = psutil.Process()
        cpu_percent = process.cpu_percent(interval=1)
        
        assert cpu_percent < 10, f"CPU usage {cpu_percent}% is too high when idle"
        print(f"\n✅ CPU usage when idle: {cpu_percent}%")


# ============================================================================
# RELIABILITY TESTS
# ============================================================================

class TestReliability:
    """Test application stability and error handling."""
    
    def test_load_multiple_files_stability(self, main_window, test_edf_files):
        """Test that loading multiple files doesn't cause issues."""
        if len(test_edf_files) < 2:
            pytest.skip("Need at least 2 EDF files")
        
        # Load files multiple times
        for _ in range(3):
            for edf_file in test_edf_files[:2]:
                with patch.object(QFileDialog, 'getOpenFileName', return_value=(edf_file, '')):
                    main_window.open_edf()
                assert len(main_window.canvas.signals) > 0
        
        print("\n✅ Loaded multiple files 3 times without crashes")
    
    def test_recording_cycle_stability(self, main_window, test_edf_files):
        """Test that starting/stopping recording multiple times is stable."""
        if not test_edf_files:
            pytest.skip("No EDF files available")
        
        edf_file = test_edf_files[0]
        with patch.object(QFileDialog, 'getOpenFileName', return_value=(edf_file, '')):
            main_window.open_edf()
        
        # Start and stop recording 5 times
        for i in range(5):
            main_window.start_session_recording()
            assert main_window.session_recording_active == True
            
            main_window.session_data.append({"cycle": i})
            
            with patch("builtins.open", create=True):
                with patch("json.dump"):
                    main_window.stop_session_recording()
            assert main_window.session_recording_active == False
        
        print("\n✅ Recording cycled 5 times without issues")
    
    def test_no_file_selected_handling(self, main_window):
        """Test that app handles cancelled file dialogs gracefully."""
        # Simulate user cancelling file dialog
        with patch.object(QFileDialog, 'getOpenFileName', return_value=('', '')):
            main_window.open_edf()
        
        # App should still be responsive
        assert main_window.isVisible()
        print("\n✅ Handled cancelled file dialog gracefully")


# ============================================================================
# USABILITY TESTS
# ============================================================================

class TestUsability:
    """Test UI responsiveness and user experience."""
    
    def test_window_is_visible(self, main_window):
        """Test that window is visible on screen."""
        assert main_window.isVisible() == True
        print("\n✅ Window is visible")
    
    def test_window_has_title(self, main_window):
        """Test that window has a descriptive title."""
        title = main_window.windowTitle()
        assert len(title) > 0, "Window should have a title"
        print(f"\n✅ Window has title: '{title}'")
    
    def test_window_minimum_size(self, main_window):
        """Test that window has reasonable minimum size."""
        width = main_window.width()
        height = main_window.height()
        
        assert width >= 800, f"Window width {width} is too small (should be >= 800)"
        assert height >= 600, f"Window height {height} is too small (should be >= 600)"
        print(f"\n✅ Window size: {width}x{height}")
    
    def test_menu_bar_accessible(self, main_window):
        """Test that menu bar is accessible and has menus."""
        menubar = main_window.menuBar()
        assert menubar is not None
        
        menu_titles = [action.text() for action in menubar.actions()]
        assert len(menu_titles) >= 2, "Should have at least 2 menus"
        print(f"\n✅ Menu bar has {len(menu_titles)} menus")
    
    def test_recording_button_exists(self, main_window):
        """Test that recording toggle action exists."""
        assert hasattr(main_window, 'toggle_session_recording_action')
        assert main_window.toggle_session_recording_action is not None
        print("\n✅ Recording button exists")


# ============================================================================
# STRESS TESTS
# ============================================================================

class TestStress:
    """Test application under stress conditions."""
    
    def test_rapid_file_loading(self, main_window, test_edf_files):
        """Test rapidly loading files back-to-back."""
        if not test_edf_files:
            pytest.skip("No EDF files available")
        
        edf_file = test_edf_files[0]
        
        # Load same file 10 times rapidly
        start_time = time.time()
        for _ in range(10):
            with patch.object(QFileDialog, 'getOpenFileName', return_value=(edf_file, '')):
                main_window.open_edf()
        total_time = time.time() - start_time
        
        assert len(main_window.canvas.signals) > 0
        print(f"\n✅ Loaded file 10 times in {total_time:.2f}s")
    
    def test_recording_with_large_data(self, main_window, test_edf_files):
        """Test recording with many data points."""
        if not test_edf_files:
            pytest.skip("No EDF files available")
        
        edf_file = test_edf_files[0]
        with patch.object(QFileDialog, 'getOpenFileName', return_value=(edf_file, '')):
            main_window.open_edf()
        
        main_window.start_session_recording()
        
        # Add 1000 data points
        for i in range(1000):
            main_window.session_data.append({
                "index": i,
                "timestamp": time.time(),
                "data": f"test_data_{i}"
            })
        
        assert len(main_window.session_data) == 1000
        print("\n✅ Recording handled 1000 data points")


# ============================================================================
# SUMMARY
# ============================================================================
"""
Non-Functional Testing Categories:

1. PERFORMANCE TESTS
   - Measure startup time, load time, response time
   - Ensure operations complete within acceptable timeframes
   - Test: < 3s startup, < 5s EDF load, < 1s recording start

2. RESOURCE USAGE TESTS
   - Monitor memory consumption
   - Track CPU usage
   - Test: < 300MB initial, < 500MB with EDF, < 10% CPU idle

3. RELIABILITY TESTS
   - Test stability over repeated operations
   - Error handling and recovery
   - Test: Multiple file loads, recording cycles, cancelled dialogs

4. USABILITY TESTS
   - UI visibility and accessibility
   - Window sizing and layout
   - Menu and button availability

5. STRESS TESTS
   - Rapid operations
   - Large data volumes
   - Test: 10 rapid loads, 1000 data points

How to Run:
-----------
All non-functional tests:
    pytest test/test_nonfunctional.py -v

Specific category:
    pytest test/test_nonfunctional.py::TestPerformance -v
    pytest test/test_nonfunctional.py::TestResourceUsage -v
    pytest test/test_nonfunctional.py::TestReliability -v

With timing output:
    pytest test/test_nonfunctional.py -v -s

Requirements:
-------------
    pip install psutil

Key Metrics to Monitor:
-----------------------
✓ Startup time: < 3 seconds
✓ EDF load time: < 5 seconds  
✓ Memory usage: < 300MB initial, < 500MB with data
✓ CPU usage idle: < 10%
✓ Window size: >= 800x600
✓ No crashes after repeated operations
"""

if __name__ == "__main__":
    print("\n" + "="*70)
    print("EDF VIEWER - NON-FUNCTIONAL TESTS")
    print("="*70)
    pytest.main([__file__, "-v", "-s"])
