#!/bin/bash
# Run tests for Snail Trails GPU simulation

echo "========================================="
echo "  Snail Trails - Test Suite"
echo "========================================="
echo

# Check if pytest is installed
if ! python -c "import pytest" 2>/dev/null; then
    echo "Installing test dependencies..."
    pip install pytest pytest-cov -q
fi

echo "Running tests..."
echo

# Run tests with coverage
python -m pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

echo
echo "========================================="
echo "Test suite complete!"
echo "========================================="
