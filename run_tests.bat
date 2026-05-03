@echo off
REM Test Suite Quick Start Guide for Windows
REM Run this batch file to execute all tests with detailed reporting

echo ========================================
echo Medical EDF Viewer - Test Suite
echo ========================================
echo.

REM Check if pytest is installed
pip show pytest > nul 2>&1
if errorlevel 1 (
    echo Installing test dependencies...
    pip install -r test_requirements.txt
)

echo Running test suite...
echo.

echo 1. Running all tests with detailed output...
py -3.10 -m pytest test_auth_screens.py -v --tb=short

echo.
echo ========================================
echo Running tests with coverage report...
echo.

REM Run with coverage
py -3.10 -m pytest test_auth_screens.py -v --cov=auth_screens --cov-report=term-missing

echo.
echo ========================================
echo Test Summary:
echo - Total Tests: 49
echo - Test Categories: 6
echo - Framework: pytest
echo ========================================
echo.
echo To run specific tests:
echo   pytest test_auth_screens.py::TestUserManager -v
echo   pytest test_auth_screens.py::TestUserManager::test_validate_login_success -v
echo.
echo To generate HTML coverage report:
echo   pytest test_auth_screens.py --cov=auth_screens --cov-report=html
echo.
pause
