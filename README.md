# fraud-detection-api
Building a simple end-to-end fraud detection model pipeline to demonstrate key skills.


## Running with Docker

To build the image, run:
```bash
docker build -t fraud-detection-api .
```

To run the container:
```bash
docker run -p 8000:8000 fraud-detection-api
```

To test the API:
```bash
curl http://localhost:8000/
```

Make a prediction: Fraudulent transaction
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @example_fraud_transaction.json
```
Make a prediction: Non-fraudulent transaction
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @example_nonfraud_transaction.json
```


## Running Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run with output (see print statements)
python -m pytest tests/ -v -s
```

### Test Coverage

- Input validation (negative amounts, missing features)
- API response format and types
- Health check endpoint
- End-to-end prediction flow