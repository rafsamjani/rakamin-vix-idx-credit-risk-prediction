# Deployment Guide and Production Setup

## Executive Summary

This guide provides comprehensive instructions for deploying the Credit Risk Prediction System to production environments, including local development, cloud deployment, and operational considerations for maintaining a reliable and scalable system.

## Deployment Architecture

### System Overview
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API   │    │   Database      │
│   (Streamlit)   │◄──►│   (FastAPI)     │◄──►│   (PostgreSQL)  │
│                 │    │                 │    │                 │
│ - User Interface│    │ - Model API     │    │ - User Data     │
│ - Visualizations│    │ - Predictions   │    │ - Audit Logs    │
│ - Forms         │    │ - Authentication│    │ - Metrics       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   ML Model      │
                    │   (XGBoost)     │
                    │                 │
                    │ - Predictions   │
                    │ - Feature Eng.  │
                    │ - Monitoring    │
                    └─────────────────┘
```

### Technology Stack
- **Frontend**: Streamlit Dashboard
- **Backend**: FastAPI for REST API
- **Database**: PostgreSQL for data persistence
- **Machine Learning**: XGBoost model with Python
- **Containerization**: Docker & Docker Compose
- **Orchestration**: Kubernetes (optional)
- **Monitoring**: Prometheus + Grafana
- **CI/CD**: GitHub Actions

## Local Development Setup

### Prerequisites
```bash
# Required software
- Python 3.9+
- Docker & Docker Compose
- Git
- PostgreSQL client tools
```

### Step 1: Clone Repository
```bash
git clone https://github.com/rafsamjani/rakamin-vix-idx-credit-risk-prediction.git
cd rakamin-vix-idx-credit-risk-prediction
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Setup Database
```bash
# Create Docker Compose file for local development
cat > docker-compose.yml << 'EOF'
version: '3.8'
services:
  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: credit_risk_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
EOF

# Start database
docker-compose up -d postgres

# Run database migrations
python -m alembic upgrade head
```

### Step 4: Configure Environment
```bash
# Create environment file
cat > .env << 'EOF'
# Database Configuration
DATABASE_URL=postgresql://postgres:password@localhost:5432/credit_risk_db

# Redis Configuration
REDIS_URL=redis://localhost:6379

# Model Configuration
MODEL_PATH=models/best_model.pkl
MODEL_VERSION=1.0.0

# Security
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-key-here

# Application Settings
DEBUG=True
LOG_LEVEL=INFO

# External APIs (if needed)
RATE_LIMIT_API=https://api.example.com/rates
CREDIT_BUREAU_API=https://api.example.com/credit
EOF
```

### Step 5: Load Sample Data
```bash
# Download and load sample dataset
python scripts/download_data.py

# Process and load data into database
python scripts/load_data.py

# Train initial model
python src/train_model.py
```

### Step 6: Run Local Development Server
```bash
# Start backend API server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# In another terminal, start Streamlit dashboard
streamlit run dashboard/app.py --server.port 8501
```

## Production Deployment Options

### Option 1: Streamlit Cloud (Recommended for Simplicity)

#### Step 1: Prepare for Deployment
```python
# requirements.txt (Streamlit specific)
streamlit==1.29.0
plotly==5.17.0
pandas==2.1.4
numpy==1.24.3
scikit-learn==1.3.2
joblib==1.4.0
```

#### Step 2: Create Streamlit App Configuration
```python
# .streamlit/config.toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[server]
headless = true
port = 8501

[browser]
gatherUsageStats = false
```

#### Step 3: Deploy to Streamlit Cloud
1. Push code to GitHub repository
2. Connect Streamlit Cloud to GitHub
3. Configure environment variables in Streamlit Cloud
4. Deploy application

### Option 2: Docker Container Deployment

#### Step 1: Create Dockerfile
```dockerfile
# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 streamlit
USER streamlit

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Start Streamlit app
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0"]
```

#### Step 2: Create Docker Compose for Production
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - MODEL_PATH=/app/models/best_model.pkl
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    depends_on:
      - postgres
      - redis
    restart: unless-stopped

  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:6-alpine
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app
    restart: unless-stopped

volumes:
  postgres_data:
```

#### Step 3: Deploy with Docker Compose
```bash
# Set environment variables
export DATABASE_URL="postgresql://user:pass@postgres:5432/credit_risk_db"
export POSTGRES_DB="credit_risk_db"
export POSTGRES_USER="your_username"
export POSTGRES_PASSWORD="your_password"

# Deploy production stack
docker-compose -f docker-compose.prod.yml up -d
```

### Option 3: Kubernetes Deployment

#### Step 1: Create Kubernetes Manifests
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: credit-risk-system
---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: credit-risk-system
data:
  MODEL_PATH: "/app/models/best_model.pkl"
  LOG_LEVEL: "INFO"
---
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
  namespace: credit-risk-system
type: Opaque
data:
  DATABASE_URL: <base64-encoded-db-url>
  SECRET_KEY: <base64-encoded-secret>
---
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: credit-risk-app
  namespace: credit-risk-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: credit-risk-app
  template:
    metadata:
      labels:
        app: credit-risk-app
    spec:
      containers:
      - name: app
        image: your-registry/credit-risk-app:latest
        ports:
        - containerPort: 8501
        envFrom:
        - configMapRef:
            name: app-config
        - secretRef:
            name: app-secrets
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 5
          periodSeconds: 5
---
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: credit-risk-service
  namespace: credit-risk-system
spec:
  selector:
    app: credit-risk-app
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8501
  type: ClusterIP
---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: credit-risk-ingress
  namespace: credit-risk-system
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - credit-risk.yourdomain.com
    secretName: credit-risk-tls
  rules:
  - host: credit-risk.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: credit-risk-service
            port:
              number: 80
```

#### Step 2: Deploy to Kubernetes
```bash
# Apply all manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n credit-risk-system
kubectl get services -n credit-risk-system
kubectl get ingress -n credit-risk-system
```

## Monitoring and Observability

### Application Monitoring

#### 1. Prometheus Metrics
```python
# src/monitoring.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define metrics
prediction_counter = Counter('predictions_total', 'Total predictions', ['result'])
prediction_latency = Histogram('prediction_duration_seconds', 'Prediction duration')
active_users = Gauge('active_users_current', 'Current active users')

def record_prediction(result, duration):
    """Record prediction metrics"""
    prediction_counter.labels(result=result).inc()
    prediction_latency.observe(duration)

def start_metrics_server(port=8001):
    """Start Prometheus metrics server"""
    start_http_server(port)
```

#### 2. Health Checks
```python
# src/health.py
from fastapi import FastAPI
import psutil
import joblib

def health_check():
    """Comprehensive health check"""
    checks = {
        "model_loaded": check_model_availability(),
        "database_connected": check_database_connection(),
        "memory_usage": check_memory_usage(),
        "disk_space": check_disk_space()
    }

    all_healthy = all(checks.values())
    status_code = 200 if all_healthy else 503

    return {"status": "healthy" if all_healthy else "unhealthy", "checks": checks}

def check_model_availability():
    """Check if model file exists and is loadable"""
    try:
        joblib.load("models/best_model.pkl")
        return True
    except:
        return False
```

#### 3. Logging Configuration
```python
# src/logging_config.py
import logging
import sys
from pythonjsonlogger import jsonlogger

def setup_logging():
    """Setup structured JSON logging"""

    logHandler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s'
    )
    logHandler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logHandler)

    return logger
```

### Model Performance Monitoring

#### 1. Prediction Quality Monitoring
```python
# src/model_monitoring.py
import numpy as np
from sklearn.metrics import roc_auc_score

class ModelMonitor:
    """Monitor model performance and data drift"""

    def __init__(self):
        self.predictions = []
        self.actuals = []
        self.feature_stats = {}

    def log_prediction(self, features, prediction, confidence):
        """Log prediction for monitoring"""
        self.predictions.append({
            'features': features,
            'prediction': prediction,
            'confidence': confidence,
            'timestamp': datetime.now()
        })

        # Update feature statistics
        self.update_feature_stats(features)

    def log_actual(self, actual):
        """Log actual outcome when available"""
        if self.predictions:
            self.predictions[-1]['actual'] = actual
            self.actuals.append(actual)

    def check_drift(self, threshold=0.1):
        """Check for data drift"""
        if len(self.feature_stats) < 100:
            return False

        # Compare recent feature distributions to baseline
        current_stats = self.calculate_current_stats()
        drift_score = self.calculate_drift_score(current_stats)

        return drift_score > threshold

    def generate_alerts(self):
        """Generate alerts for model issues"""
        alerts = []

        # Check for performance degradation
        if len(self.actuals) >= 50:
            recent_auc = self.calculate_recent_auc()
            if recent_auc < 0.75:
                alerts.append({
                    'type': 'performance_degradation',
                    'message': f'Model AUC dropped to {recent_auc:.3f}',
                    'severity': 'high'
                })

        # Check for data drift
        if self.check_drift():
            alerts.append({
                'type': 'data_drift',
                'message': 'Significant data drift detected',
                'severity': 'medium'
            })

        return alerts
```

#### 2. Automated Model Retraining
```python
# src/auto_retrain.py
import schedule
import time

def should_retrain():
    """Determine if model should be retrained"""
    monitor = ModelMonitor()

    # Check performance degradation
    alerts = monitor.generate_alerts()
    performance_alerts = [a for a in alerts if a['type'] == 'performance_degradation']

    # Check data freshness
    last_training = get_last_training_date()
    days_since_training = (datetime.now() - last_training).days

    return len(performance_alerts) > 0 or days_since_training > 30

def retrain_model():
    """Execute model retraining pipeline"""
    try:
        logger.info("Starting model retraining...")

        # Load new data
        new_data = load_recent_data()

        # Train new model
        new_model = train_model_pipeline(new_data)

        # Validate new model
        validation_score = validate_model(new_model)

        if validation_score > 0.80:
            # Backup old model
            backup_current_model()

            # Deploy new model
            deploy_model(new_model)

            # Update monitoring
            reset_model_monitor()

            logger.info("Model retraining completed successfully")
            return True
        else:
            logger.warning(f"New model validation failed: {validation_score:.3f}")
            return False

    except Exception as e:
        logger.error(f"Model retraining failed: {str(e)}")
        return False

# Schedule retraining checks
schedule.every(1).days.do(should_retrain)

def start_retraining_scheduler():
    """Start the retraining scheduler"""
    while True:
        schedule.run_pending()
        time.sleep(3600)  # Check every hour
```

## Security Configuration

### 1. Application Security
```python
# src/security.py
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timedelta

security = HTTPBearer()

def create_access_token(data: dict, expires_delta: timedelta = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def rate_limiter(max_requests: int = 100, window_seconds: int = 3600):
    """Rate limiting decorator"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Implement rate limiting logic
            # Use Redis for distributed rate limiting
            pass
        return wrapper
    return decorator
```

### 2. Data Protection
```python
# src/data_protection.py
import hashlib
import hmac

def anonymize_data(df, sensitive_columns):
    """Anonymize sensitive data for logging/analytics"""
    df_anonymized = df.copy()

    for column in sensitive_columns:
        if column in df_anonymized.columns:
            # Hash sensitive columns
            df_anonymized[column] = df_anonymized[column].apply(
                lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:16]
            )

    return df_anonymized

def encrypt_sensitive_data(data, key):
    """Encrypt sensitive data at rest"""
    # Implement encryption logic
    pass

def mask_pii(text):
    """Mask personally identifiable information"""
    # Simple masking - implement based on requirements
    import re

    # Mask email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                  lambda m: f"{m.group()[0]}***@{m.group().split('@')[1]}", text)

    # Mask phone numbers
    text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '***-***-****', text)

    return text
```

## CI/CD Pipeline Setup

### GitHub Actions Workflow
```yaml
# .github/workflows/deploy.yml
name: Build and Deploy

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run tests
      run: |
        pytest tests/ --cov=src --cov-report=xml

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  build-and-deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
    - uses: actions/checkout@v3

    - name: Build Docker image
      run: |
        docker build -t credit-risk-app:${{ github.sha }} .
        docker tag credit-risk-app:${{ github.sha }} credit-risk-app:latest

    - name: Deploy to production
      run: |
        # Deploy commands based on your infrastructure
        echo "Deploying to production..."
        # kubectl apply -f k8s/
        # or docker-compose -f docker-compose.prod.yml up -d
```

## Performance Optimization

### 1. Model Optimization
```python
# src/model_optimization.py
import pickle
import gzip

def optimize_model_for_production(model_path):
    """Optimize model for production deployment"""

    # Load model
    model = joblib.load(model_path)

    # Optimize prediction function
    def predict_optimized(features):
        # Use optimized data structures
        if isinstance(features, dict):
            features_array = np.array([list(features.values())])
        else:
            features_array = features

        return model.predict_proba(features_array)[:, 1]

    # Save optimized model
    optimized_model = {
        'predictor': predict_optimized,
        'feature_names': model.feature_names_in_,
        'model_version': '1.0.0'
    }

    # Compress model file
    with gzip.open('models/model_optimized.pkl.gz', 'wb') as f:
        pickle.dump(optimized_model, f)
```

### 2. Caching Strategy
```python
# src/caching.py
import redis
import json
from functools import wraps

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_predictions(expiration=3600):
    """Cache predictions for identical inputs"""
    def decorator(func):
        @wraps(func)
        def wrapper(features):
            # Create cache key from features
            cache_key = f"prediction:{hash(json.dumps(sorted(features.items())))}"

            # Try to get from cache
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)

            # Compute and cache result
            result = func(features)
            redis_client.setex(cache_key, expiration, json.dumps(result))

            return result
        return wrapper
    return decorator
```

## Disaster Recovery and Backup

### 1. Database Backup Strategy
```bash
#!/bin/bash
# scripts/backup_database.sh

# Backup database
BACKUP_DIR="/backups/postgres"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="credit_risk_backup_${DATE}.sql"

# Create backup
pg_dump -h localhost -U postgres -d credit_risk_db > "${BACKUP_DIR}/${BACKUP_FILE}"

# Compress backup
gzip "${BACKUP_DIR}/${BACKUP_FILE}"

# Delete old backups (keep last 30 days)
find ${BACKUP_DIR} -name "*.sql.gz" -mtime +30 -delete

# Upload to cloud storage (AWS S3 example)
aws s3 cp "${BACKUP_DIR}/${BACKUP_FILE}.gz" "s3://credit-risk-backups/database/"
```

### 2. Model Backup Strategy
```bash
#!/bin/bash
# scripts/backup_models.sh

MODEL_DIR="/app/models"
BACKUP_DIR="/backups/models"
DATE=$(date +%Y%m%d_%H%M%S)

# Create model backup
tar -czf "${BACKUP_DIR}/models_backup_${DATE}.tar.gz" -C "${MODEL_DIR}" .

# Upload to cloud storage
aws s3 cp "${BACKUP_DIR}/models_backup_${DATE}.tar.gz" "s3://credit-risk-backups/models/"
```

This comprehensive deployment guide ensures that the Credit Risk Prediction System can be reliably deployed, monitored, and maintained in production environments while meeting security, performance, and scalability requirements.