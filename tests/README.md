# Testing Guide

## Test Organization

```
tests/
├── unit/              # Unit tests for individual components
├── integration/       # Integration tests for module interactions
└── e2e/              # End-to-end tests for complete flows
```

## Running Tests

```bash
# All tests
poetry run pytest

# Unit tests only
poetry run pytest tests/unit/

# Integration tests
poetry run pytest tests/integration/

# E2E tests
poetry run pytest tests/e2e/

# With coverage
poetry run pytest --cov=src --cov-report=html

# Specific test file
poetry run pytest tests/unit/test_compression.py

# Specific test function
poetry run pytest tests/unit/test_compression.py::test_text_compression

# Verbose output
poetry run pytest -v

# Fail fast
poetry run pytest -x
```

## Test Categories

### Unit Tests
- Individual function and class testing
- Mocked dependencies
- Fast execution (<1s per test)

### Integration Tests
- Multiple component interactions
- Real dependencies where practical
- Moderate execution time (1-5s)

### E2E Tests
- Complete pipeline testing
- Real audio files and models
- Slower execution (>5s)

## Writing New Tests

1. Create test file: `test_<module>.py`
2. Use fixtures from `conftest.py`
3. Follow AAA pattern (Arrange, Act, Assert)
4. Add docstrings explaining test purpose
5. Mark slow tests with `@pytest.mark.slow`

## Fixtures Available

- `sample_audio`: 1-second test audio
- `sample_audio_16k`: 16kHz resampled audio
- `test_config`: Minimal mode configuration
- `stt_model`: Mocked STT model
- `tts_model`: Mocked TTS model

## Code Coverage Goals

- Overall: >80%
- Critical paths (compression, network): >90%
- UI/CLI: >60%
