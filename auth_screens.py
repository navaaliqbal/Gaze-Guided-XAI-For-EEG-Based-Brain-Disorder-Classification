import json
import os
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QRect
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QMessageBox, QFrame, QGraphicsDropShadowEffect
)
from PyQt5.QtGui import QFont, QColor


class ToastNotification(QWidget):
    """Small floating toast notification that appears in top center."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.timer = QTimer()
        self.timer.timeout.connect(self.hide_toast)
        self.animation = None
        self.init_ui()

    def init_ui(self):
        """Initialize toast UI."""
        layout = QHBoxLayout()
        layout.setContentsMargins(25, 16, 25, 16)
        layout.setSpacing(12)
        self.setLayout(layout)

        self.message_label = QLabel()
        self.message_label.setFont(QFont("Arial", 13, QFont.Bold))
        self.message_label.setWordWrap(True)
        layout.addWidget(self.message_label)

        self.setMinimumHeight(70)
        self.setMinimumWidth(380)
        self.setMaximumHeight(85)
        self.setMaximumWidth(550)

    def show_success(self, message, duration=3000):
        """Show success message (green toast)."""
        self.message_label.setText(message)
        self.setStyleSheet("""
            QWidget {
                background: #D4EDDA;
                border: 2px solid #28A745;
                border-radius: 8px;
            }
        """)
        self.message_label.setStyleSheet("color: #155724;")
        self.show_toast(duration)

    def show_error(self, message, duration=4000):
        """Show error message (red toast)."""
        self.message_label.setText(message)
        self.setStyleSheet("""
            QWidget {
                background: #F8D7DA;
                border: 2px solid #DC3545;
                border-radius: 8px;
            }
        """)
        self.message_label.setStyleSheet("color: #721C24;")
        self.show_toast(duration)

    def show_info(self, message, duration=3500):
        """Show info message (blue toast)."""
        self.message_label.setText(message)
        self.setStyleSheet("""
            QWidget {
                background: #D1ECF1;
                border: 2px solid #17A2B8;
                border-radius: 8px;
            }
        """)
        self.message_label.setStyleSheet("color: #0C5460;")
        self.show_toast(duration)

    def show_toast(self, duration):
        """Show the toast notification."""
        self.position_at_top_center()
        self.show()
        self.raise_()
        self.timer.start(duration)

    def position_at_top_center(self):
        """Position the toast at the top center of the parent window."""
        if self.parent():
            parent_rect = self.parent().geometry()
            toast_width = self.width()
            x = parent_rect.x() + (parent_rect.width() - toast_width) // 2
            y = parent_rect.y() + 30
            self.move(x, y)
        else:
            # If no parent, center on screen
            screen = QtWidgets.QApplication.desktop().screenGeometry()
            x = (screen.width() - self.width()) // 2
            y = 30
            self.move(x, y)

    def hide_toast(self):
        """Hide the toast notification and stop timer."""
        self.timer.stop()
        self.hide()


class UserManager:
    """Manages user authentication and storage with role-based access."""

    def __init__(self):
        self.storage_path = os.path.join(os.path.dirname(__file__), "users.json")
        self.users = {
            "admin": {"password": "admin2024", "role": "admin"},
            "dr.smith": {"password": "neuro2024", "role": "user"},
            "tech.jones": {"password": "eeg2024", "role": "user"}
        }
        self._load_users()

    def _load_users(self):
        if not os.path.exists(self.storage_path):
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as file_handle:
                data = json.load(file_handle)
            if isinstance(data, dict):
                # Merge stored users on top of defaults.
                self.users.update(data)
        except (OSError, json.JSONDecodeError):
            pass

    def _save_users(self):
        try:
            with open(self.storage_path, "w", encoding="utf-8") as file_handle:
                json.dump(self.users, file_handle, indent=2)
        except OSError:
            pass

    def validate_login(self, username, password):
        if username not in self.users:
            return False
        user_data = self.users[username]
        if isinstance(user_data, dict):
            return user_data.get("password") == password
        else:
            # Handle old format
            return user_data == password

    def user_exists(self, username):
        return username in self.users

    def get_user_role(self, username):
        """Get the role of a user (admin or user)."""
        if username not in self.users:
            return None
        user_data = self.users[username]
        if isinstance(user_data, dict):
            return user_data.get("role", "user")
        return "user"

    def is_admin(self, username):
        """Check if user is admin."""
        return self.get_user_role(username) == "admin"

    def add_user(self, username, password, role="user"):
        self.users[username] = {"password": password, "role": role}
        self._save_users()
        return True

    def delete_user(self, username):
        """Delete a user account."""
        if username in self.users and username != "admin":
            del self.users[username]
            self._save_users()
            return True
        return False

    def get_all_users(self):
        """Get all users."""
        return list(self.users.keys())


class LoginWindow(QMainWindow):
    """Login screen with username and password fields."""

    def __init__(self, user_manager, main_window_class):
        super().__init__()
        self.user_manager = user_manager
        self.main_window_class = main_window_class
        self.main_window = None
        self.signup_window = None
        self.toast = ToastNotification(self)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Login - Medical EDF Viewer")
        self.setGeometry(100, 100, 900, 700)

        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #4A90E2, stop:0.5 #2E5BBA, stop:1 #1E3A8A);
            }
        """)

        self.center_on_screen()
        self.setWindowState(Qt.WindowMaximized)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        central_widget.setStyleSheet("background: transparent;")

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(60, 60, 60, 60)
        main_layout.setSpacing(18)
        central_widget.setLayout(main_layout)

        panel = QFrame()
        panel.setStyleSheet("""
            QFrame {
                background: white;
                border-radius: 16px;
            }
        """)
        panel_layout = QVBoxLayout()
        panel_layout.setContentsMargins(40, 40, 40, 40)
        panel_layout.setSpacing(14)
        panel.setLayout(panel_layout)
        panel.setMaximumWidth(480)

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(26)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(0, 8)
        panel.setGraphicsEffect(shadow)

        title_label = QLabel("Medical EDF Viewer")
        title_label.setFont(QFont("Arial", 22, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2E5BBA;")
        panel_layout.addWidget(title_label)

        subtitle_label = QLabel("Sign in to continue")
        subtitle_label.setFont(QFont("Arial", 11))
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("color: #666;")
        panel_layout.addWidget(subtitle_label)

        panel_layout.addSpacing(8)

        username_label = QLabel("Username")
        username_label.setFont(QFont("Arial", 10, QFont.Bold))
        username_label.setStyleSheet("color: #333;")
        panel_layout.addWidget(username_label)

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Enter your username")
        self.username_input.setMinimumHeight(40)
        self.username_input.setStyleSheet("""
            QLineEdit {
                padding: 8px 10px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-size: 12px;
                background: #f8f9fa;
                color: #333;
            }
            QLineEdit:focus {
                border: 2px solid #4A90E2;
                background: white;
            }
        """)
        panel_layout.addWidget(self.username_input)

        password_label = QLabel("Password")
        password_label.setFont(QFont("Arial", 10, QFont.Bold))
        password_label.setStyleSheet("color: #333;")
        panel_layout.addWidget(password_label)

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Enter your password")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setMinimumHeight(40)
        self.password_input.setStyleSheet("""
            QLineEdit {
                padding: 8px 10px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-size: 12px;
                background: #f8f9fa;
                color: #333;
            }
            QLineEdit:focus {
                border: 2px solid #4A90E2;
                background: white;
            }
        """)
        panel_layout.addWidget(self.password_input)

        panel_layout.addSpacing(6)

        self.login_button = QPushButton("Login")
        self.login_button.setMinimumHeight(42)
        self.login_button.setCursor(Qt.PointingHandCursor)
        self.login_button.setStyleSheet("""
            QPushButton {
                background: #2E5BBA;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #1E4BA0;
            }
            QPushButton:pressed {
                background: #153B80;
            }
        """)
        self.login_button.clicked.connect(self.handle_login)
        panel_layout.addWidget(self.login_button)

        signup_layout = QHBoxLayout()
        signup_layout.setAlignment(Qt.AlignCenter)
        signup_text = QLabel("Don't have an account?")
        signup_text.setFont(QFont("Arial", 10))
        signup_text.setStyleSheet("color: #666;")
        signup_link = QLabel("<a href='#' style='color: #4A90E2; text-decoration: none; font-weight: bold;'>Sign Up</a>")
        signup_link.setFont(QFont("Arial", 10))
        signup_link.setTextFormat(Qt.RichText)
        signup_link.setOpenExternalLinks(False)
        signup_link.setCursor(Qt.PointingHandCursor)
        signup_link.linkActivated.connect(self.show_signup)
        signup_layout.addWidget(signup_text)
        signup_layout.addSpacing(5)
        signup_layout.addWidget(signup_link)
        panel_layout.addLayout(signup_layout)

        main_layout.addStretch()
        main_layout.addWidget(panel, 0, Qt.AlignHCenter)
        main_layout.addStretch()

        self.username_input.returnPressed.connect(self.handle_login)
        self.password_input.returnPressed.connect(self.handle_login)

    def center_on_screen(self):
        screen = QtWidgets.QApplication.desktop().screenGeometry()
        window_geometry = self.geometry()
        x = (screen.width() - window_geometry.width()) // 2
        y = (screen.height() - window_geometry.height()) // 2
        self.move(x, y)

    def handle_login(self):
        username = self.username_input.text().strip()
        password = self.password_input.text()

        if not username:
            self.toast.show_error("Please enter a username.")
            return

        if not password:
            self.toast.show_error("Please enter a password.")
            return

        if self.user_manager.validate_login(username, password):
            self.toast.show_success(f"Welcome, {username}!")
            # Check if user is admin
            if self.user_manager.is_admin(username):
                # Delay to show the toast
                QTimer.singleShot(1500, lambda: self.open_admin_dashboard(username))
            else:
                # Delay to show the toast
                QTimer.singleShot(1500, self.open_main_window)
        else:
            self.toast.show_error("Invalid username or password.")

    def open_main_window(self):
        self.main_window = self.main_window_class()
        self.main_window.show()
        self.close()

    def open_admin_dashboard(self, username):
        self.admin_dashboard = AdminDashboard(self.user_manager, self.main_window_class, username)
        self.admin_dashboard.show()
        self.close()

    def show_signup(self):
        self.signup_window = SignupWindow(self.user_manager, self)
        self.signup_window.show()
        self.hide()


class SignupWindow(QMainWindow):
    """Signup screen for new user registration."""

    def __init__(self, user_manager, login_window):
        super().__init__()
        self.user_manager = user_manager
        self.login_window = login_window
        self.toast = ToastNotification(self)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Sign Up - Medical EDF Viewer")
        self.setGeometry(100, 100, 900, 700)

        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #4A90E2, stop:0.5 #2E5BBA, stop:1 #1E3A8A);
            }
        """)

        self.center_on_screen()
        self.setWindowState(Qt.WindowMaximized)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        central_widget.setStyleSheet("background: transparent;")

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(60, 60, 60, 60)
        main_layout.setSpacing(18)
        central_widget.setLayout(main_layout)

        panel = QFrame()
        panel.setStyleSheet("""
            QFrame {
                background: white;
                border-radius: 16px;
            }
        """)
        panel_layout = QVBoxLayout()
        panel_layout.setContentsMargins(40, 40, 40, 40)
        panel_layout.setSpacing(14)
        panel.setLayout(panel_layout)
        panel.setMaximumWidth(480)

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(26)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(0, 8)
        panel.setGraphicsEffect(shadow)

        title_label = QLabel("Create Account")
        title_label.setFont(QFont("Arial", 22, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2E5BBA;")
        panel_layout.addWidget(title_label)

        subtitle_label = QLabel("Sign up to get started")
        subtitle_label.setFont(QFont("Arial", 11))
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("color: #666;")
        panel_layout.addWidget(subtitle_label)

        panel_layout.addSpacing(8)

        username_label = QLabel("Username")
        username_label.setFont(QFont("Arial", 10, QFont.Bold))
        username_label.setStyleSheet("color: #333;")
        panel_layout.addWidget(username_label)

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Choose a username")
        self.username_input.setMinimumHeight(40)
        self.username_input.setStyleSheet("""
            QLineEdit {
                padding: 8px 10px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-size: 12px;
                background: #f8f9fa;
                color: #333;
            }
            QLineEdit:focus {
                border: 2px solid #4A90E2;
                background: white;
            }
        """)
        panel_layout.addWidget(self.username_input)

        password_label = QLabel("Password")
        password_label.setFont(QFont("Arial", 10, QFont.Bold))
        password_label.setStyleSheet("color: #333;")
        panel_layout.addWidget(password_label)

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("At least 6 characters")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setMinimumHeight(40)
        self.password_input.setStyleSheet("""
            QLineEdit {
                padding: 8px 10px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-size: 12px;
                background: #f8f9fa;
                color: #333;
            }
            QLineEdit:focus {
                border: 2px solid #4A90E2;
                background: white;
            }
        """)
        panel_layout.addWidget(self.password_input)

        confirm_password_label = QLabel("Confirm Password")
        confirm_password_label.setFont(QFont("Arial", 10, QFont.Bold))
        confirm_password_label.setStyleSheet("color: #333;")
        panel_layout.addWidget(confirm_password_label)

        self.confirm_password_input = QLineEdit()
        self.confirm_password_input.setPlaceholderText("Re-enter your password")
        self.confirm_password_input.setEchoMode(QLineEdit.Password)
        self.confirm_password_input.setMinimumHeight(40)
        self.confirm_password_input.setStyleSheet("""
            QLineEdit {
                padding: 8px 10px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-size: 12px;
                background: #f8f9fa;
                color: #333;
            }
            QLineEdit:focus {
                border: 2px solid #4A90E2;
                background: white;
            }
        """)
        panel_layout.addWidget(self.confirm_password_input)

        panel_layout.addSpacing(6)

        self.signup_button = QPushButton("Create Account")
        self.signup_button.setMinimumHeight(42)
        self.signup_button.setCursor(Qt.PointingHandCursor)
        self.signup_button.setStyleSheet("""
            QPushButton {
                background: #2E5BBA;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #1E4BA0;
            }
            QPushButton:pressed {
                background: #153B80;
            }
        """)
        self.signup_button.clicked.connect(self.handle_signup)
        panel_layout.addWidget(self.signup_button)

        login_layout = QHBoxLayout()
        login_layout.setAlignment(Qt.AlignCenter)
        login_text = QLabel("Already have an account?")
        login_text.setFont(QFont("Arial", 10))
        login_text.setStyleSheet("color: #666;")
        login_link = QLabel("<a href='#' style='color: #4A90E2; text-decoration: none; font-weight: bold;'>Login</a>")
        login_link.setFont(QFont("Arial", 10))
        login_link.setTextFormat(Qt.RichText)
        login_link.setOpenExternalLinks(False)
        login_link.setCursor(Qt.PointingHandCursor)
        login_link.linkActivated.connect(self.back_to_login)
        login_layout.addWidget(login_text)
        login_layout.addSpacing(5)
        login_layout.addWidget(login_link)
        panel_layout.addLayout(login_layout)

        main_layout.addStretch()
        main_layout.addWidget(panel, 0, Qt.AlignHCenter)
        main_layout.addStretch()

        self.username_input.returnPressed.connect(self.handle_signup)
        self.password_input.returnPressed.connect(self.handle_signup)
        self.confirm_password_input.returnPressed.connect(self.handle_signup)

    def center_on_screen(self):
        screen = QtWidgets.QApplication.desktop().screenGeometry()
        window_geometry = self.geometry()
        x = (screen.width() - window_geometry.width()) // 2
        y = (screen.height() - window_geometry.height()) // 2
        self.move(x, y)

    def handle_signup(self):
        username = self.username_input.text().strip()
        password = self.password_input.text()
        confirm_password = self.confirm_password_input.text()

        if not username:
            self.toast.show_error("Username cannot be empty.")
            return

        if len(password) < 6:
            self.toast.show_error("Password must be at least 6 characters long.")
            return

        if password != confirm_password:
            self.toast.show_error("Passwords do not match.")
            return

        if self.user_manager.user_exists(username):
            self.toast.show_error("Username already exists.")
            return

        self.user_manager.add_user(username, password)
        self.toast.show_success("Account created successfully!")
        QTimer.singleShot(2000, self.back_to_login)

    def back_to_login(self):
        self.login_window.show()
        self.close()

    def closeEvent(self, event):
        self.login_window.show()
        event.accept()


class AdminDashboard(QMainWindow):
    """Admin panel for managing user accounts."""

    def __init__(self, user_manager, main_window_class, admin_username):
        super().__init__()
        self.user_manager = user_manager
        self.main_window_class = main_window_class
        self.admin_username = admin_username
        self.main_window = None
        self.login_window = None
        self.toast = ToastNotification(self)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Admin Panel - Medical EDF Viewer")
        self.setGeometry(100, 100, 1000, 700)

        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #4A90E2, stop:0.5 #2E5BBA, stop:1 #1E3A8A);
            }
        """)

        self.center_on_screen()
        self.setWindowState(Qt.WindowMaximized)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        central_widget.setStyleSheet("background: transparent;")

        # Main vertical layout with banner at top
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        central_widget.setLayout(main_layout)

        # Content layout (horizontal for panels)
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)

        # Left panel - Admin info and buttons
        left_panel = QFrame()
        left_panel.setStyleSheet("""
            QFrame {
                background: white;
                border-radius: 16px;
            }
        """)
        left_panel.setMaximumWidth(300)
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(30, 30, 30, 30)
        left_layout.setSpacing(20)
        left_panel.setLayout(left_layout)

        shadow_left = QGraphicsDropShadowEffect()
        shadow_left.setBlurRadius(26)
        shadow_left.setColor(QColor(0, 0, 0, 80))
        shadow_left.setOffset(0, 8)
        left_panel.setGraphicsEffect(shadow_left)

        admin_title = QLabel("Admin Panel")
        admin_title.setFont(QFont("Arial", 18, QFont.Bold))
        admin_title.setAlignment(Qt.AlignCenter)
        admin_title.setStyleSheet("color: #2E5BBA;")
        left_layout.addWidget(admin_title)

        admin_info = QLabel(f"Logged in as:\n{self.admin_username}")
        admin_info.setFont(QFont("Arial", 10))
        admin_info.setAlignment(Qt.AlignCenter)
        admin_info.setStyleSheet("color: #666; padding: 10px;")
        admin_info.setWordWrap(True)
        left_layout.addWidget(admin_info)

        left_layout.addSpacing(20)

        start_app_button = QPushButton("Start Application")
        start_app_button.setMinimumHeight(45)
        start_app_button.setCursor(Qt.PointingHandCursor)
        start_app_button.setStyleSheet("""
            QPushButton {
                background: #27AE60;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #229954;
            }
            QPushButton:pressed {
                background: #1E8449;
            }
        """)
        start_app_button.clicked.connect(self.start_application)
        left_layout.addWidget(start_app_button)

        logout_button = QPushButton("Logout")
        logout_button.setMinimumHeight(45)
        logout_button.setCursor(Qt.PointingHandCursor)
        logout_button.setStyleSheet("""
            QPushButton {
                background: #E74C3C;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #CB4335;
            }
            QPushButton:pressed {
                background: #A93226;
            }
        """)
        logout_button.clicked.connect(self.logout)
        left_layout.addWidget(logout_button)

        left_layout.addStretch()

        content_layout.addWidget(left_panel)

        # Right panel - User management
        right_panel = QFrame()
        right_panel.setStyleSheet("""
            QFrame {
                background: white;
                border-radius: 16px;
            }
        """)
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(30, 30, 30, 30)
        right_layout.setSpacing(15)
        right_panel.setLayout(right_layout)

        shadow_right = QGraphicsDropShadowEffect()
        shadow_right.setBlurRadius(26)
        shadow_right.setColor(QColor(0, 0, 0, 80))
        shadow_right.setOffset(0, 8)
        right_panel.setGraphicsEffect(shadow_right)

        users_title = QLabel("Manage Users")
        users_title.setFont(QFont("Arial", 16, QFont.Bold))
        users_title.setStyleSheet("color: #2E5BBA;")
        right_layout.addWidget(users_title)

        # Users scroll area
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: #f8f9fa;
                border-radius: 8px;
            }
            QScrollBar:vertical {
                background: #e0e0e0;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #4A90E2;
                border-radius: 4px;
            }
        """)

        scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout()
        self.scroll_layout.setSpacing(10)
        self.scroll_layout.setContentsMargins(10, 10, 10, 10)
        scroll_widget.setLayout(self.scroll_layout)
        scroll.setWidget(scroll_widget)

        right_layout.addWidget(scroll)

        self.refresh_user_list()

        content_layout.addWidget(right_panel, 1)
        main_layout.addLayout(content_layout)

    def refresh_user_list(self):
        """Refresh the list of users."""
        # Clear existing widgets
        while self.scroll_layout.count():
            child = self.scroll_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        users = self.user_manager.get_all_users()
        if not users:
            no_users = QLabel("No users found")
            no_users.setFont(QFont("Arial", 11))
            no_users.setStyleSheet("color: #999;")
            self.scroll_layout.addWidget(no_users)
            return

        for username in sorted(users):
            user_frame = self.create_user_item(username)
            self.scroll_layout.addWidget(user_frame)

        self.scroll_layout.addStretch()

    def create_user_item(self, username):
        """Create a user item widget."""
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background: #f8f9fa;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
            }
        """)
        layout = QHBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        frame.setLayout(layout)

        user_role = self.user_manager.get_user_role(username)
        role_text = "Admin" if user_role == "admin" else "User"
        role_color = "#E74C3C" if user_role == "admin" else "#27AE60"

        username_label = QLabel(username)
        username_label.setFont(QFont("Arial", 11, QFont.Bold))
        username_label.setStyleSheet("color: #333;")
        layout.addWidget(username_label)

        role_label = QLabel(role_text)
        role_label.setFont(QFont("Arial", 9, QFont.Bold))
        role_label.setStyleSheet(f"color: white; background: {role_color}; border-radius: 4px; padding: 4px 8px;")
        layout.addWidget(role_label)

        layout.addStretch()

        # Delete button (not for admin account)
        if username != "admin":
            delete_button = QPushButton("Delete")
            delete_button.setMaximumWidth(80)
            delete_button.setMinimumHeight(30)
            delete_button.setCursor(Qt.PointingHandCursor)
            delete_button.setStyleSheet("""
                QPushButton {
                    background: #E74C3C;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    font-size: 11px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background: #CB4335;
                }
                QPushButton:pressed {
                    background: #A93226;
                }
            """)
            delete_button.clicked.connect(lambda checked, u=username: self.delete_user(u))
            layout.addWidget(delete_button)

        return frame

    def delete_user(self, username):
        """Delete a user account."""
        if self.user_manager.delete_user(username):
            self.toast.show_success(f"User '{username}' has been deleted.")
            QTimer.singleShot(1500, self.refresh_user_list)
        else:
            self.toast.show_error(f"Cannot delete user '{username}'.")

    def start_application(self):
        """Start the main application."""
        self.main_window = self.main_window_class()
        self.main_window.show()
        self.close()

    def logout(self):
        """Logout and go back to login screen."""
        self.login_window = LoginWindow(self.user_manager, self.main_window_class)
        self.login_window.show()
        self.close()

    def center_on_screen(self):
        screen = QtWidgets.QApplication.desktop().screenGeometry()
        window_geometry = self.geometry()
        x = (screen.width() - window_geometry.width()) // 2
        y = (screen.height() - window_geometry.height()) // 2
        self.move(x, y)
