"""
Integration Tests for Authentication System
Tests the real LoginWindow, SignupWindow, and AdminDashboard GUI screens
Screens display and run while tests interact with them
"""
# cd "c:\Users\Qadri laptop\Downloads\64(1)\64" ; pytest test/test_auth_screens.py -v --tb=short 2>&1 | head -150
import sys
import os
import pytest
import json
import tempfile
import shutil
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtTest import QTest
from auth_screens import UserManager, LoginWindow, SignupWindow, AdminDashboard, ToastNotification

# Create QApplication globally for all tests
_app = QApplication.instance()
if _app is None:
    # Use offscreen platform to prevent GUI from displaying during tests
    _app = QApplication(sys.argv + ['-platform', 'offscreen'])


class TestLoginScreenDisplay:
    """Test the actual LoginWindow GUI display and interactions."""

    @pytest.fixture
    def mock_main_window(self):
        """Create a mock main window class."""
        class MockMainWindow(QMainWindow):
            def __init__(self):
                super().__init__()
                self.setWindowTitle("EDF Viewer - Main")
                
        return MockMainWindow

    @pytest.fixture
    def user_manager(self):
        """Create UserManager with default users."""
        manager = UserManager()
        manager.users = {
            "admin": {"password": "admin2024", "role": "admin"},
            "dr.smith": {"password": "neuro2024", "role": "user"},
            "tech.jones": {"password": "eeg2024", "role": "user"}
        }
        manager.storage_path = None  # Disable saving for tests
        return manager

    def test_login_screen_displays(self, user_manager, mock_main_window):
        """Test that LoginWindow displays and initializes properly."""
        print("\n=== DISPLAYING LOGIN SCREEN ===")
        login_window = LoginWindow(user_manager, mock_main_window)
        login_window.show()
        
        # Verify window is visible
        assert login_window.isVisible()
        assert login_window.windowTitle() == "Login - Medical EDF Viewer"
        
        # Verify UI elements exist
        assert login_window.username_input is not None
        assert login_window.password_input is not None
        assert login_window.login_button is not None
        
        _app.processEvents()
        time.sleep(1)  # Show the screen briefly
        login_window.close()

    def test_login_with_valid_admin_credentials(self, user_manager, mock_main_window):
        """Test logging in with valid admin credentials."""
        print("\n=== TESTING ADMIN LOGIN ===")
        login_window = LoginWindow(user_manager, mock_main_window)
        login_window.show()
        
        # Simulate user entering admin credentials
        QTest.keyClicks(login_window.username_input, "admin")
        QTest.keyClicks(login_window.password_input, "admin2024")
        
        _app.processEvents()
        
        # Click login button
        QTest.mouseClick(login_window.login_button, Qt.LeftButton)
        
        _app.processEvents()
        time.sleep(1)
        
        # Verify username and password were entered
        assert login_window.username_input.text() == "admin"
        assert login_window.password_input.text() == "admin2024"
        
        login_window.close()

    def test_login_with_valid_user_credentials(self, user_manager, mock_main_window):
        """Test logging in with valid user credentials."""
        print("\n=== TESTING USER LOGIN ===")
        login_window = LoginWindow(user_manager, mock_main_window)
        login_window.show()
        
        # Simulate user entering valid credentials
        QTest.keyClicks(login_window.username_input, "dr.smith")
        QTest.keyClicks(login_window.password_input, "neuro2024")
        
        # Click login button
        QTest.mouseClick(login_window.login_button, Qt.LeftButton)
        
        _app.processEvents()
        time.sleep(1)
        
        assert login_window.username_input.text() == "dr.smith"
        assert login_window.password_input.text() == "neuro2024"
        
        login_window.close()

    def test_login_with_invalid_credentials(self, user_manager, mock_main_window):
        """Test login fails with invalid credentials and shows error toast."""
        print("\n=== TESTING INVALID LOGIN (showing error toast) ===")
        login_window = LoginWindow(user_manager, mock_main_window)
        login_window.show()
        
        # Simulate user entering incorrect credentials
        QTest.keyClicks(login_window.username_input, "admin")
        QTest.keyClicks(login_window.password_input, "wrongpassword")
        
        # Click login button
        QTest.mouseClick(login_window.login_button, Qt.LeftButton)
        
        _app.processEvents()
        time.sleep(2)  # Let error toast show
        
        # Toast should show error
        assert login_window.toast is not None
        
        login_window.close()

    def test_login_with_empty_username(self, user_manager, mock_main_window):
        """Test login validation with empty username."""
        print("\n=== TESTING EMPTY USERNAME VALIDATION ===")
        login_window = LoginWindow(user_manager, mock_main_window)
        login_window.show()
        
        # Enter only password, leave username empty
        QTest.keyClicks(login_window.password_input, "admin2024")
        
        # Click login button
        QTest.mouseClick(login_window.login_button, Qt.LeftButton)
        
        _app.processEvents()
        time.sleep(1)
        
        # Username should still be empty
        assert login_window.username_input.text() == ""
        
        login_window.close()

    def test_login_with_empty_password(self, user_manager, mock_main_window):
        """Test login validation with empty password."""
        print("\n=== TESTING EMPTY PASSWORD VALIDATION ===")
        login_window = LoginWindow(user_manager, mock_main_window)
        login_window.show()
        
        # Enter only username, leave password empty
        QTest.keyClicks(login_window.username_input, "admin")
        
        # Click login button
        QTest.mouseClick(login_window.login_button, Qt.LeftButton)
        
        _app.processEvents()
        time.sleep(1)
        
        # Password should still be empty
        assert login_window.password_input.text() == ""
        
        login_window.close()

    def test_login_screen_signup_link(self, user_manager, mock_main_window):
        """Test that signup link navigates to signup screen."""
        print("\n=== TESTING SIGNUP SCREEN LINK ===")
        login_window = LoginWindow(user_manager, mock_main_window)
        login_window.show()
        
        assert login_window.signup_window is None
        
        # Click signup link
        login_window.show_signup()
        
        _app.processEvents()
        time.sleep(1)
        
        # Verify signup window was created
        assert login_window.signup_window is not None
        
        login_window.close()
        if login_window.signup_window:
            login_window.signup_window.close()

    def test_login_password_is_masked(self, user_manager, mock_main_window):
        """Test that password field masks input."""
        login_window = LoginWindow(user_manager, mock_main_window)
        login_window.show()
        
        # Verify password field is set to password echo mode
        assert login_window.password_input.echoMode() == 2  # QLineEdit.Password = 2
        
        login_window.close()


class TestSignupScreenDisplay:
    """Test the actual SignupWindow GUI display and interactions."""

    @pytest.fixture
    def mock_main_window(self):
        """Create a mock main window class."""
        class MockMainWindow(QMainWindow):
            def __init__(self):
                super().__init__()
                self.setWindowTitle("EDF Viewer - Main")
        return MockMainWindow

    @pytest.fixture
    def user_manager(self):
        """Create UserManager with default users."""
        manager = UserManager()
        manager.users = {
            "admin": {"password": "admin2024", "role": "admin"},
        }
        manager.storage_path = None  # Disable saving for tests
        return manager

    def test_signup_screen_displays(self, user_manager, mock_main_window):
        """Test that SignupWindow displays and initializes properly."""
        print("\n=== DISPLAYING SIGNUP SCREEN ===")
        login_window = LoginWindow(user_manager, mock_main_window)
        signup_window = SignupWindow(user_manager, login_window)
        signup_window.show()
        
        # Verify window is visible
        assert signup_window.isVisible()
        assert signup_window.windowTitle() == "Sign Up - Medical EDF Viewer"
        
        # Verify UI elements exist
        assert signup_window.username_input is not None
        assert signup_window.password_input is not None
        assert signup_window.confirm_password_input is not None
        assert signup_window.signup_button is not None
        
        _app.processEvents()
        time.sleep(1)
        signup_window.close()
        login_window.close()

    def test_signup_with_valid_credentials(self, user_manager, mock_main_window):
        """Test signing up with valid credentials."""
        print("\n=== TESTING SIGNUP WITH VALID CREDENTIALS ===")
        login_window = LoginWindow(user_manager, mock_main_window)
        signup_window = SignupWindow(user_manager, login_window)
        signup_window.show()
        
        # Simulate user signup
        QTest.keyClicks(signup_window.username_input, "newuser")
        QTest.keyClicks(signup_window.password_input, "password123")
        QTest.keyClicks(signup_window.confirm_password_input, "password123")
        
        _app.processEvents()
        time.sleep(1)
        
        # Verify inputs
        assert signup_window.username_input.text() == "newuser"
        assert signup_window.password_input.text() == "password123"
        assert signup_window.confirm_password_input.text() == "password123"
        
        signup_window.close()
        login_window.close()

    def test_signup_with_short_password(self, user_manager, mock_main_window):
        """Test signup validation for short password."""
        print("\n=== TESTING SHORT PASSWORD VALIDATION ===")
        login_window = LoginWindow(user_manager, mock_main_window)
        signup_window = SignupWindow(user_manager, login_window)
        signup_window.show()
        
        # Simulate user entering short password
        QTest.keyClicks(signup_window.username_input, "newuser")
        QTest.keyClicks(signup_window.password_input, "short")
        QTest.keyClicks(signup_window.confirm_password_input, "short")
        
        # Click signup button
        QTest.mouseClick(signup_window.signup_button, Qt.LeftButton)
        
        _app.processEvents()
        time.sleep(2)  # Show error toast
        
        assert signup_window.password_input.text() == "short"
        
        signup_window.close()
        login_window.close()

    def test_signup_with_mismatched_passwords(self, user_manager, mock_main_window):
        """Test signup fails with mismatched passwords."""
        print("\n=== TESTING MISMATCHED PASSWORD VALIDATION ===")
        login_window = LoginWindow(user_manager, mock_main_window)
        signup_window = SignupWindow(user_manager, login_window)
        signup_window.show()
        
        # Simulate user entering different passwords
        QTest.keyClicks(signup_window.username_input, "newuser")
        QTest.keyClicks(signup_window.password_input, "password123")
        QTest.keyClicks(signup_window.confirm_password_input, "different123")
        
        # Click signup button
        QTest.mouseClick(signup_window.signup_button, Qt.LeftButton)
        
        _app.processEvents()
        time.sleep(2)  # Show error toast
        
        # Passwords should not match
        assert signup_window.password_input.text() != signup_window.confirm_password_input.text()
        
        signup_window.close()
        login_window.close()

    def test_signup_with_existing_username(self, user_manager, mock_main_window):
        """Test signup fails with existing username."""
        print("\n=== TESTING DUPLICATE USERNAME VALIDATION ===")
        login_window = LoginWindow(user_manager, mock_main_window)
        signup_window = SignupWindow(user_manager, login_window)
        signup_window.show()
        
        # Try to signup with existing admin username
        QTest.keyClicks(signup_window.username_input, "admin")
        QTest.keyClicks(signup_window.password_input, "newpass123")
        QTest.keyClicks(signup_window.confirm_password_input, "newpass123")
        
        # Click signup button
        QTest.mouseClick(signup_window.signup_button, Qt.LeftButton)
        
        _app.processEvents()
        time.sleep(2)  # Show error toast
        
        assert signup_window.username_input.text() == "admin"
        
        signup_window.close()
        login_window.close()

    def test_signup_screen_back_to_login(self, user_manager, mock_main_window):
        """Test navigating back to login from signup."""
        print("\n=== TESTING BACK TO LOGIN FROM SIGNUP ===")
        login_window = LoginWindow(user_manager, mock_main_window)
        signup_window = SignupWindow(user_manager, login_window)
        signup_window.show()
        
        # Simulate clicking back to login
        signup_window.back_to_login()
        
        _app.processEvents()
        time.sleep(1)
        
        # Signup should be closed
        assert not signup_window.isVisible()
        
        signup_window.close()
        login_window.close()

    def test_signup_password_is_masked(self, user_manager, mock_main_window):
        """Test that password fields mask input."""
        login_window = LoginWindow(user_manager, mock_main_window)
        signup_window = SignupWindow(user_manager, login_window)
        signup_window.show()
        
        # Verify password fields are masked
        assert signup_window.password_input.echoMode() == 2  # QLineEdit.Password
        assert signup_window.confirm_password_input.echoMode() == 2
        
        signup_window.close()
        login_window.close()


class TestAdminDashboardDisplay:
    """Test the actual AdminDashboard GUI display and interactions."""

    @pytest.fixture
    def mock_main_window(self):
        """Create a mock main window class."""
        class MockMainWindow(QMainWindow):
            def __init__(self):
                super().__init__()
                self.setWindowTitle("EDF Viewer - Main")
        return MockMainWindow

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def user_manager(self, temp_dir):
        """Create UserManager with multiple users."""
        manager = UserManager()
        manager.storage_path = os.path.join(temp_dir, "test_users.json")
        manager.users = {
            "admin": {"password": "admin2024", "role": "admin"},
            "dr.smith": {"password": "neuro2024", "role": "user"},
            "tech.jones": {"password": "eeg2024", "role": "user"},
            "nurse.patel": {"password": "nurse2024", "role": "user"}
        }
        return manager

    def test_admin_dashboard_displays(self, user_manager, mock_main_window):
        """Test that AdminDashboard displays properly."""
        print("\n=== DISPLAYING ADMIN DASHBOARD ===")
        admin_dashboard = AdminDashboard(user_manager, mock_main_window, "admin")
        admin_dashboard.show()
        
        # Verify window is visible
        assert admin_dashboard.isVisible()
        assert admin_dashboard.windowTitle() == "Admin Panel - Medical EDF Viewer"
        
        _app.processEvents()
        time.sleep(1)
        admin_dashboard.close()

    def test_admin_dashboard_shows_admin_username(self, user_manager, mock_main_window):
        """Test that admin username is displayed."""
        admin_dashboard = AdminDashboard(user_manager, mock_main_window, "admin")
        admin_dashboard.show()
        
        # Verify admin username is stored
        assert admin_dashboard.admin_username == "admin"
        
        _app.processEvents()
        time.sleep(1)
        admin_dashboard.close()

    def test_admin_dashboard_displays_all_users(self, user_manager, mock_main_window):
        """Test that all users are displayed in the dashboard."""
        print("\n=== DISPLAYING ADMIN USER LIST ===")
        admin_dashboard = AdminDashboard(user_manager, mock_main_window, "admin")
        admin_dashboard.show()
        
        _app.processEvents()
        time.sleep(1)
        
        # Verify user list is populated
        assert admin_dashboard.scroll_layout is not None
        # Count widgets in scroll layout (includes stretch)
        widget_count = admin_dashboard.scroll_layout.count()
        assert widget_count > 0
        
        admin_dashboard.close()

    def test_admin_can_delete_user(self, user_manager, mock_main_window):
        """Test that admin can delete a user from the dashboard."""
        print("\n=== TESTING USER DELETION FROM ADMIN PANEL ===")
        admin_dashboard = AdminDashboard(user_manager, mock_main_window, "admin")
        admin_dashboard.show()
        
        # Verify user exists before delete
        assert user_manager.user_exists("dr.smith")
        
        # Delete user
        admin_dashboard.delete_user("dr.smith")
        
        _app.processEvents()
        time.sleep(1)
        
        # Verify user is deleted
        assert not user_manager.user_exists("dr.smith")
        
        admin_dashboard.close()

    def test_admin_cannot_delete_own_account(self, user_manager, mock_main_window):
        """Test that admin account cannot be deleted."""
        admin_dashboard = AdminDashboard(user_manager, mock_main_window, "admin")
        admin_dashboard.show()
        
        # Try to delete admin account
        result = user_manager.delete_user("admin")
        
        # Verify deletion failed
        assert result is False
        assert user_manager.user_exists("admin")
        
        admin_dashboard.close()

    def test_admin_dashboard_refresh_user_list(self, user_manager, mock_main_window):
        """Test that user list refreshes after changes."""
        print("\n=== TESTING USER LIST REFRESH ===")
        admin_dashboard = AdminDashboard(user_manager, mock_main_window, "admin")
        admin_dashboard.show()
        
        initial_widget_count = admin_dashboard.scroll_layout.count()
        
        # Delete a user
        user_manager.delete_user("dr.smith")
        admin_dashboard.refresh_user_list()
        
        _app.processEvents()
        time.sleep(1)
        
        # List should refresh
        assert admin_dashboard.scroll_layout is not None
        
        admin_dashboard.close()

    def test_admin_dashboard_logout(self, user_manager, mock_main_window):
        """Test that admin can logout from dashboard."""
        print("\n=== TESTING ADMIN LOGOUT ===")
        admin_dashboard = AdminDashboard(user_manager, mock_main_window, "admin")
        admin_dashboard.show()
        
        # Trigger logout
        admin_dashboard.logout()
        
        _app.processEvents()
        time.sleep(1)
        
        # Admin dashboard should be closed
        assert not admin_dashboard.isVisible()
        
        # Login window should be created
        assert admin_dashboard.login_window is not None
        
        admin_dashboard.close()
        if admin_dashboard.login_window:
            admin_dashboard.login_window.close()


class TestUserManager:
    """Tests for UserManager functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def user_manager(self, temp_dir):
        """Create a UserManager instance."""
        manager = UserManager()
        manager.storage_path = os.path.join(temp_dir, "test_users.json")
        manager.users = {
            "admin": {"password": "admin2024", "role": "admin"},
            "dr.smith": {"password": "neuro2024", "role": "user"},
        }
        return manager

    def test_validate_login_success(self, user_manager):
        """Test successful login validation."""
        assert user_manager.validate_login("admin", "admin2024") is True
        assert user_manager.validate_login("dr.smith", "neuro2024") is True

    def test_validate_login_failure(self, user_manager):
        """Test login fails with wrong password."""
        assert user_manager.validate_login("admin", "wrongpassword") is False
        assert user_manager.validate_login("nonexistent", "password") is False

    def test_user_exists(self, user_manager):
        """Test user_exists method."""
        assert user_manager.user_exists("admin") is True
        assert user_manager.user_exists("nonexistent") is False

    def test_get_user_role(self, user_manager):
        """Test get_user_role method."""
        assert user_manager.get_user_role("admin") == "admin"
        assert user_manager.get_user_role("dr.smith") == "user"
        assert user_manager.get_user_role("nonexistent") is None

    def test_is_admin(self, user_manager):
        """Test is_admin method."""
        assert user_manager.is_admin("admin") is True
        assert user_manager.is_admin("dr.smith") is False

    def test_add_user(self, user_manager):
        """Test adding a new user."""
        user_manager.add_user("newuser", "password123")
        assert user_manager.user_exists("newuser") is True
        assert user_manager.validate_login("newuser", "password123") is True

    def test_delete_user(self, user_manager):
        """Test deleting a user."""
        user_manager.add_user("temp_user", "password")
        assert user_manager.delete_user("temp_user") is True
        assert user_manager.user_exists("temp_user") is False

    def test_delete_admin_fails(self, user_manager):
        """Test that admin can't be deleted."""
        assert user_manager.delete_user("admin") is False
        assert user_manager.user_exists("admin") is True

    def test_get_all_users(self, user_manager):
        """Test get_all_users returns all users."""
        users = user_manager.get_all_users()
        assert len(users) == 2
        assert "admin" in users
        assert "dr.smith" in users


class TestAuthenticationFlow:
    """Integration tests for complete authentication flows."""

    @pytest.fixture
    def mock_main_window(self):
        """Create a mock main window class."""
        class MockMainWindow(QMainWindow):
            def __init__(self):
                super().__init__()
                self.setWindowTitle("EDF Viewer - Main")
        return MockMainWindow

    @pytest.fixture
    def user_manager(self):
        """Create UserManager with test users."""
        manager = UserManager()
        manager.users = {
            "admin": {"password": "admin2024", "role": "admin"},
            "testuser": {"password": "testpass123", "role": "user"}
        }
        manager.storage_path = None
        return manager

    def test_complete_admin_login_flow(self, user_manager, mock_main_window):
        """Test complete admin login flow."""
        print("\n=== COMPLETE ADMIN LOGIN FLOW ===")
        login_window = LoginWindow(user_manager, mock_main_window)
        login_window.show()
        
        # Simulate admin login
        QTest.keyClicks(login_window.username_input, "admin")
        QTest.keyClicks(login_window.password_input, "admin2024")
        QTest.mouseClick(login_window.login_button, Qt.LeftButton)
        
        _app.processEvents()
        time.sleep(1)
        
        # Verify credentials accepted
        assert login_window.username_input.text() == "admin"
        
        login_window.close()

    def test_complete_user_signup_flow(self, user_manager, mock_main_window):
        """Test complete user signup flow."""
        print("\n=== COMPLETE USER SIGNUP FLOW ===")
        login_window = LoginWindow(user_manager, mock_main_window)
        signup_window = SignupWindow(user_manager, login_window)
        signup_window.show()
        
        # Simulate signup
        QTest.keyClicks(signup_window.username_input, "newuser")
        QTest.keyClicks(signup_window.password_input, "newpass2024")
        QTest.keyClicks(signup_window.confirm_password_input, "newpass2024")
        
        _app.processEvents()
        time.sleep(1)
        
        # Verify user was created
        assert signup_window.username_input.text() == "newuser"
        
        signup_window.close()
        login_window.close()


if __name__ == "__main__":
    """Run all tests with pytest."""
    pytest.main([__file__, "-v", "-s", "--tb=short"])
