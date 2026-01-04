"""
API Design Decisions:

1. Preprocessing: API scales Amount internally using saved scaler
2. Input validation: Pydantic model enforces 30 features
3. Response format: prediction + probability + threshold + version
4. Error handling: Return 422 for validation errors with details
5. Model loading: On startup, not per-request
6. Threshold: Default 0.5, configurable via environment variable
"""

