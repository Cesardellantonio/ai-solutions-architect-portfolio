# Project 2: API Integration

A REST API serving an ML model for sentiment analysis, demonstrating model deployment, request handling, and scaling patterns.

## Architecture

```
[Client Request] → [Flask REST API] → [ML Model (Sentiment)] → [JSON Response]
                         │                                            │
                    Rate limiting                              Confidence score
                    Input validation                           + prediction
```

> Full architecture diagram: `diagrams/api-architecture.png`

## Business Value

**Scenario:** SaaS customer feedback analyzer — real-time sentiment classification of support tickets to prioritize responses and improve customer satisfaction.

## Setup

```bash
pip install -r requirements.txt
python app.py
# API available at http://localhost:5000
```

## Key Concepts

- REST API design for ML model serving
- Request/response patterns and error handling
- Scaling strategies (load balancing, caching)
- Governance and data privacy in API design

## Status

🔲 Not Started
