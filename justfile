test:
    uv run --extra test pytest -vvv tests

test-embedding:
    MBED_USE_REAL_EMBEDDINGS=1 @just test
    