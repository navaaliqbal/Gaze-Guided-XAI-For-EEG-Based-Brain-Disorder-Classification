#!/bin/bash
# Test Suite Quick Start Guide
# Run this script to execute all tests with detailed reporting

echo "========================================"
echo "Medical EDF Viewer - Test Suite"
echo "========================================"
echo ""

# Check if pytest is installed
if ! python -m pip show pytest > /dev/null; then
    echo "Installing test dependencies..."
    python -m pip install -r test_requirements.txt
fi

echo "Running test suite..."
echo ""

# Run all tests with verbose output
echo "1. Running all tests with detailed output..."
python -m pytest test_auth_screens.py -v --tb=short

echo ""
echo "========================================"
echo "Running tests with coverage report..."
echo ""

# Run with coverage
python -m pytest test_auth_screens.py -v --cov=auth_screens --cov-report=term-missing

echo ""
echo "========================================"
echo "Test Summary:"
echo "- Total Tests: 49"
echo "- Test Categories: 6"
echo "- Framework: pytest"
echo "========================================"
echo ""
echo "To run specific tests:"
echo "  pytest test_auth_screens.py::TestUserManager -v"
echo "  pytest test_auth_screens.py::TestUserManager::test_validate_login_success -v"
echo ""
echo "To generate HTML coverage report:"
echo "  pytest test_auth_screens.py --cov=auth_screens --cov-report=html"
echo ""
