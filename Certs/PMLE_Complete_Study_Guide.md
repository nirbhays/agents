# Google Professional Machine Learning Engineer - Complete Study Guide

**Last Updated:** December 28, 2025  
**Exam Format:** 60 questions in 120 minutes  
**Passing Score:** ~70-75% (42-45 correct answers)  
**Cost:** $200 USD  
**Validity:** 2 years from certification date

---

## 🎯 EXAM OVERVIEW

### What Changed in 2025

**Major Updates:**
1. **Vertex AI Unified Platform** - All AI Platform questions now use Vertex AI terminology
2. **Generative AI Integration** - 15-20% of exam covers LLMs, RAG, vector databases
3. **MLOps Maturity** - Stronger focus on production pipelines, monitoring, automation
4. **Cost Optimization** - More scenario-based questions on resource management
5. **Responsible AI** - Increased coverage of fairness, explainability, privacy

**Exam Domains (Official Breakdown):**
1. **Framing ML Problems** - 10-15%
2. **Architecting ML Solutions** - 20-25%
3. **Data Preparation & Processing** - 15-20%
4. **ML Model Development** - 20-25%
5. **ML Pipeline Automation & Orchestration** - 10-15%
6. **Monitoring, Optimizing & Maintaining ML Solutions** - 15-20%

---

## 📚 DOMAIN 1: FRAMING ML PROBLEMS (10-15%)

### 1.1 Translating Business Objectives into ML Use Cases

**Key Concepts:**
- **Supervised vs Unsupervised vs Reinforcement Learning**
- **Classification vs Regression vs Clustering**
- **When ML is NOT appropriate** (rule-based systems might be better)
- **ML Problem Types:**
  - Binary classification (fraud detection, spam)
  - Multi-class classification (image recognition)
  - Multi-label classification (tagging)
  - Regression (price prediction, forecasting)
  - Clustering (customer segmentation)
  - Anomaly detection (system monitoring)
  - Recommendation systems (collaborative filtering)
  - Time series forecasting (demand prediction)

**Common Exam Scenarios:**
1. **Retail:** Product recommendation, demand forecasting, inventory optimization
2. **Healthcare:** Patient outcome prediction, disease classification, drug response
3. **Finance:** Fraud detection, credit scoring, algorithmic trading
4. **Manufacturing:** Predictive maintenance, quality control, defect detection
5. **Marketing:** Customer churn prediction, lifetime value, segmentation

**Decision Framework:**
```
Business Problem → Define Success Metrics → Check Data Availability → 
Choose ML Approach → Evaluate Feasibility → Estimate ROI
```

### 1.2 Defining ML Success Metrics

**Classification Metrics:**
- **Accuracy** - When classes are balanced
- **Precision** - When false positives are costly (fraud detection)
- **Recall** - When false negatives are costly (cancer detection)
- **F1 Score** - Harmonic mean of precision and recall
- **AUC-ROC** - Overall model performance across thresholds
- **Confusion Matrix** - Detailed breakdown of predictions

**Regression Metrics:**
- **MAE (Mean Absolute Error)** - Average absolute difference
- **RMSE (Root Mean Squared Error)** - Penalizes large errors more
- **MAPE (Mean Absolute Percentage Error)** - Percentage-based error
- **R² Score** - Proportion of variance explained

**Business Metrics vs Model Metrics:**
- Model metric: AUC-ROC = 0.95
- Business metric: Saved $2M in fraud losses
- **Always align model metrics with business KPIs**

**Exam Tip:** Questions often present scenarios where you must choose between optimizing for precision vs recall based on business cost.

### 1.3 Identifying ML Problem Types

**Pattern Recognition Table:**

| Business Goal | ML Problem Type | Example |
|---------------|-----------------|---------|
| Will customer churn? | Binary Classification | Yes/No prediction |
| Which product to recommend? | Multi-class Classification | Choose from N products |
| Predict sales revenue | Regression | Continuous value |
| Group similar customers | Clustering | K-means, DBSCAN |
| Detect unusual transactions | Anomaly Detection | Isolation Forest |
| Optimize ad bidding | Reinforcement Learning | Maximize reward |
| Forecast next 7 days | Time Series | ARIMA, LSTM |

**Common Traps in Exam:**
- ❌ Using classification for continuous predictions (should be regression)
- ❌ Using supervised learning when you don't have labels (should be unsupervised)
- ❌ Treating multi-label as multi-class (different problem types)

---

## 📚 DOMAIN 2: ARCHITECTING ML SOLUTIONS (20-25%)

### 2.1 Vertex AI Platform Components (CRITICAL - 40% of this domain)

**Vertex AI Unified Platform:**
```
Vertex AI
├── AutoML (Automated ML training)
│   ├── AutoML Tables (structured data)
│   ├── AutoML Vision (image classification, object detection)
│   ├── AutoML Natural Language (text classification, entity extraction)
│   └── AutoML Video (video classification, action recognition)
├── Custom Training (Bring your own model)
│   ├── Pre-built containers (TensorFlow, PyTorch, Scikit-learn)
│   ├── Custom containers (any framework)
│   └── Hyperparameter Tuning (Vertex AI Vizier)
├── Model Registry (Centralized model management)
├── Endpoints (Model serving)
│   ├── Online predictions (real-time)
│   └── Batch predictions (offline)
├── Pipelines (Kubeflow Pipelines integration)
├── Feature Store (Feature management and serving)
├── Model Monitoring (Skew and drift detection)
├── Explainable AI (Model interpretability)
└── Generative AI (New in 2024-2025)
    ├── Model Garden (Pre-trained models)
    ├── Vertex AI Search (RAG-based search)
    └── Generative AI Studio (Prompt design & tuning)
```

**When to Use What:**

| Scenario | Recommendation | Why |
|----------|----------------|-----|
| Small dataset (<10K rows), no ML expertise | AutoML Tables | Automated feature engineering |
| Large tabular data, need full control | Custom Training + BigQuery ML | Flexibility + scale |
| Image classification with <1K images | AutoML Vision | Transfer learning built-in |
| Object detection, need custom anchors | Custom Training (TensorFlow) | Fine-grained control |
| Text classification, 5 classes | AutoML Natural Language | Quick setup, good accuracy |
| LLM fine-tuning for domain-specific tasks | Vertex AI Generative AI Studio | Managed LLM infrastructure |
| Real-time predictions (<100ms latency) | Vertex AI Endpoints (online) | Low-latency serving |
| Batch scoring 10M records nightly | Vertex AI Batch Prediction | Cost-effective for batch |
| Need feature consistency train/serve | Vertex AI Feature Store | Prevents training-serving skew |

### 2.2 Compute Options (GCP Infrastructure)

**Training Infrastructure:**

1. **Vertex AI Managed Training**
   - Pre-built containers (TensorFlow, PyTorch, XGBoost)
   - Custom containers (Docker)
   - **Machine Types:**
     - n1-standard-4 (15 GB RAM, 4 vCPUs) - Small models
     - n1-highmem-8 (52 GB RAM, 8 vCPUs) - Medium models
     - a2-highgpu-1g (85 GB RAM, 12 vCPUs, 1x A100) - Deep learning
     - a2-megagpu-16g (1360 GB RAM, 96 vCPUs, 16x A100) - Large-scale training
   - **Accelerators:**
     - NVIDIA T4 ($0.35/hr) - Training small models
     - NVIDIA V100 ($2.48/hr) - Standard deep learning
     - NVIDIA A100 ($3.67/hr) - Large models, faster training
     - TPU v3 ($8.00/hr per pod) - TensorFlow at scale

2. **AI Platform Training (Legacy - know for older questions)**
   - Still appears in some exam questions
   - Being migrated to Vertex AI

3. **GKE (Google Kubernetes Engine)**
   - Use when: Need full control of infrastructure
   - Use when: Multi-tenancy required
   - Use when: Custom orchestration logic

4. **Dataproc (Managed Spark/Hadoop)**
   - Use when: Spark MLlib workloads
   - Use when: Existing Spark pipelines
   - Use when: Processing TB+ of data for feature engineering

**Cost Optimization Strategies (High-Frequency Exam Topic):**

1. **Preemptible VMs**
   - 80% cheaper than regular VMs
   - Can be terminated anytime (24-hour max runtime)
   - **Use for:** Fault-tolerant training (checkpointing enabled)
   - **Don't use for:** Critical inference serving

2. **Spot VMs** (Newer than Preemptible)
   - Similar to Preemptible but more stable
   - No 24-hour limit
   - Slightly higher price than Preemptible

3. **Committed Use Discounts**
   - 1-year commitment: 25-37% discount
   - 3-year commitment: 52-57% discount
   - **Use for:** Production inference endpoints

4. **GPU Right-Sizing**
   - Start with T4 (cheapest), move to V100/A100 only if needed
   - Monitor GPU utilization (should be >70%)

5. **Batch Prediction vs Online**
   - Batch: $0.08 per node hour
   - Online: Always-on endpoint costs
   - **Use batch when:** Predictions can wait (daily/weekly scoring)

**Exam Pattern:** Expect 3-5 questions on "Which machine type should you choose for this scenario?"

### 2.3 Storage Options for ML

**Data Storage Decision Matrix:**

| Data Type | Volume | Access Pattern | Recommended Storage | Why |
|-----------|--------|----------------|---------------------|-----|
| Structured (tables) | <10 GB | SQL queries | Cloud SQL | Traditional RDBMS |
| Structured (tables) | 10 GB - 10 TB | Analytics | BigQuery | Serverless, fast queries |
| Structured (tables) | >10 TB | Analytics + ML | BigQuery | Federated queries, BQML |
| Semi-structured (JSON) | Any | Document queries | Firestore | NoSQL, real-time |
| Unstructured (images) | <1 TB | Training data | Cloud Storage (Standard) | Cost-effective |
| Unstructured (images) | >1 TB | Frequent access | Cloud Storage (Standard) | High throughput |
| Unstructured (images) | Archive | Rare access | Cloud Storage (Coldline/Archive) | 10x cheaper |
| Streaming data | Real-time | Event processing | Pub/Sub → Dataflow | Real-time pipelines |
| Time-series data | IoT sensors | Fast writes/reads | Bigtable | Low-latency, scalable |
| Feature vectors | ML features | Low-latency serving | Vertex AI Feature Store | Prevents skew |

**BigQuery ML Integration (Heavily Tested):**

**When to Use BigQuery ML:**
- ✅ Data already in BigQuery (avoid data movement)
- ✅ Linear regression, logistic regression, K-means, matrix factorization
- ✅ Time series forecasting (ARIMA)
- ✅ Data analysts want to build models (SQL interface)
- ✅ Quick prototyping (no Python code needed)

**When NOT to Use BigQuery ML:**
- ❌ Deep learning (use Vertex AI Custom Training)
- ❌ Complex model architectures (use TensorFlow/PyTorch)
- ❌ Reinforcement learning
- ❌ Real-time streaming predictions (use Dataflow)

**Example BigQuery ML Workflow:**
```sql
-- Create model
CREATE OR REPLACE MODEL `project.dataset.customer_churn_model`
OPTIONS(
  model_type='LOGISTIC_REG',
  input_label_cols=['churned']
) AS
SELECT * FROM `project.dataset.training_data`;

-- Evaluate model
SELECT * FROM ML.EVALUATE(MODEL `project.dataset.customer_churn_model`);

-- Make predictions
SELECT * FROM ML.PREDICT(MODEL `project.dataset.customer_churn_model`, 
  TABLE `project.dataset.new_customers`);
```

### 2.4 Vertex AI Feature Store (Growing Importance)

**Problem it Solves:**
- **Training-serving skew:** Features computed differently in training vs serving
- **Feature reusability:** Share features across multiple models
- **Point-in-time correctness:** Avoid data leakage in training

**Architecture:**
```
Raw Data → Feature Engineering → Feature Store → 
├─→ Training (Batch serving)
└─→ Inference (Online serving)
```

**When to Use:**
- ✅ Multiple models share features (e.g., user_age, product_category)
- ✅ Real-time predictions need low latency (<100ms)
- ✅ Features are expensive to compute (cache in Feature Store)
- ✅ Need feature versioning and monitoring

**When NOT to Use:**
- ❌ Batch-only predictions (overkill, use BigQuery)
- ❌ Simple models with <10 features (overhead not justified)
- ❌ Features change every request (can't cache)

**Exam Tip:** If question mentions "training-serving skew" or "feature consistency," answer is likely Vertex AI Feature Store.

---

## 📚 DOMAIN 3: DATA PREPARATION & PROCESSING (15-20%)

### 3.1 Data Processing Tools

**Tool Selection Matrix:**

| Tool | Best For | Scale | Programming | Use Case |
|------|----------|-------|-------------|----------|
| **BigQuery** | SQL analytics | Petabytes | SQL | Data exploration, BQML |
| **Dataflow** | Stream/batch ETL | Terabytes | Apache Beam (Python/Java) | Real-time pipelines |
| **Dataproc** | Spark/Hadoop | Terabytes | Spark (Scala/Python) | Existing Spark code |
| **Dataprep** | Visual ETL | Gigabytes | No-code (Trifacta) | Data analysts, cleaning |
| **Cloud Functions** | Simple transforms | Small files | Python/Node.js | Lightweight triggers |
| **Cloud Composer** | Orchestration | Any | Airflow (Python) | Complex DAGs |

**Decision Tree:**
```
Do you need real-time processing?
├─ YES → Dataflow (streaming)
└─ NO → Is data already in BigQuery?
    ├─ YES → Use BigQuery for transformations
    └─ NO → Do you have existing Spark code?
        ├─ YES → Dataproc
        └─ NO → New pipeline → Dataflow (batch)
```

### 3.2 Feature Engineering Techniques

**Common Transformations:**

1. **Numerical Features:**
   - **Scaling:** Min-max normalization, Z-score standardization
   - **Binning:** Convert continuous → categorical (age → age_group)
   - **Log transformation:** Handle skewed distributions
   - **Polynomial features:** x, x², x³ for non-linear relationships

2. **Categorical Features:**
   - **One-hot encoding:** Create binary column per category (use for <50 categories)
   - **Label encoding:** Convert to integers (use for ordinal data)
   - **Target encoding:** Replace with target mean (risk of overfitting)
   - **Embedding:** Learn dense representations (use for high cardinality)

3. **Text Features:**
   - **TF-IDF:** Term frequency × inverse document frequency
   - **Word embeddings:** Word2Vec, GloVe, BERT
   - **N-grams:** Capture phrase patterns ("not good" vs "good")

4. **Image Features:**
   - **Transfer learning:** Use pre-trained models (ResNet, EfficientNet)
   - **Data augmentation:** Flip, rotate, crop, color jitter
   - **Normalization:** Scale pixels to [0,1] or [-1,1]

5. **Time Series Features:**
   - **Lag features:** Previous values (t-1, t-7, t-30)
   - **Rolling statistics:** Moving average, moving std dev
   - **Date features:** Day of week, month, holiday indicator

**AutoML Feature Engineering:**
- Vertex AI AutoML automatically handles:
  - ✅ Scaling and normalization
  - ✅ One-hot encoding
  - ✅ Missing value imputation
  - ✅ Feature crosses (interaction features)
- **Exam Tip:** If question says "no ML expertise," choose AutoML (auto feature engineering)

### 3.3 Handling Imbalanced Data

**Problem:** 99% class A, 1% class B → Model predicts all A, gets 99% accuracy but fails business goal

**Solutions:**

1. **Resampling:**
   - **Oversampling minority class:** Duplicate rare examples (risk: overfitting)
   - **Undersampling majority class:** Remove common examples (risk: lose information)
   - **SMOTE:** Synthetic Minority Over-sampling Technique (create synthetic examples)

2. **Class Weights:**
   - Penalize misclassifying minority class more
   - TensorFlow: `class_weight={0: 1, 1: 99}` (class 1 is 99x more important)
   - Automatically balances loss function

3. **Threshold Adjustment:**
   - Default threshold: 0.5 (predict class 1 if P > 0.5)
   - Adjust threshold: 0.1 (predict class 1 if P > 0.1) → Higher recall

4. **Anomaly Detection Algorithms:**
   - If imbalance >99:1, treat as anomaly detection
   - Isolation Forest, One-Class SVM, Autoencoders

**Exam Pattern:** Expect 2-3 questions on imbalanced data (fraud detection, rare disease)

### 3.4 Data Validation & Monitoring

**TensorFlow Data Validation (TFDV):**
- Generates statistics and schema from data
- Detects anomalies (new values, missing features, type changes)
- **Use in Vertex AI Pipelines for data quality checks**

**Common Data Issues:**

| Issue | Detection | Solution |
|-------|-----------|----------|
| Missing values | TFDV statistics | Imputation (mean/median/mode) or drop |
| Duplicate records | BigQuery `DISTINCT` | Deduplication logic |
| Outliers | Z-score > 3 | Cap values or remove |
| Data type mismatch | TFDV schema validation | Cast to correct type |
| Label leakage | Manual inspection | Remove leaky features |
| Drift | TFDV compare train/serve | Retrain model |

**Exam Tip:** If question mentions "data quality checks in pipeline," answer involves TFDV or Vertex AI Pipelines validation step.

---

## 📚 DOMAIN 4: ML MODEL DEVELOPMENT (20-25%)

### 4.1 Model Selection

**Algorithm Cheat Sheet:**

**Classification:**
- **Logistic Regression:** Linear decision boundary, interpretable, fast
  - Use when: Need interpretability (regulatory), baseline model
- **Decision Trees/Random Forest:** Non-linear, handles mixed data types
  - Use when: Mixed numerical + categorical, need feature importance
- **Gradient Boosting (XGBoost/LightGBM):** High accuracy, slow training
  - Use when: Structured data, Kaggle-style competitions
- **Neural Networks:** Universal approximators, needs large data
  - Use when: Complex patterns, >100K training examples
- **SVM:** Effective in high dimensions, slow on large datasets
  - Use when: Text classification, kernel tricks needed

**Regression:**
- **Linear Regression:** Baseline, interpretable
- **Random Forest Regressor:** Robust to outliers
- **XGBoost Regressor:** Typically best for structured data
- **Neural Networks:** When non-linear relationships are complex

**Clustering:**
- **K-Means:** Fast, needs K specified
- **DBSCAN:** Doesn't need K, finds arbitrary shapes
- **Hierarchical:** Creates dendrogram, good for visualization

**Time Series:**
- **ARIMA:** Classical statistical approach
- **Prophet (Facebook):** Handles holidays and seasonality
- **LSTM/GRU:** Deep learning for complex patterns

### 4.2 TensorFlow on Vertex AI

**Training Script Structure:**
```python
import tensorflow as tf
from tensorflow import keras

def create_model():
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model(train_data, val_data, epochs=10):
    model = create_model()
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint('checkpoints/'),
            tf.keras.callbacks.TensorBoard(log_dir='logs/'),
            tf.keras.callbacks.EarlyStopping(patience=3)
        ]
    )
    return model

# Save model for Vertex AI
model.save('gs://bucket/model/')  # SavedModel format
```

**Vertex AI Custom Training Job:**
```python
from google.cloud import aiplatform

aiplatform.init(project='my-project', location='us-central1')

job = aiplatform.CustomTrainingJob(
    display_name='my-training-job',
    script_path='trainer/task.py',
    container_uri='gcr.io/cloud-aiplatform/training/tf-cpu.2-12:latest',
    requirements=['pandas==1.5.0', 'scikit-learn==1.2.0'],
    model_serving_container_image_uri='gcr.io/cloud-aiplatform/prediction/tf2-cpu.2-12:latest'
)

model = job.run(
    replica_count=1,
    machine_type='n1-standard-4',
    accelerator_type='NVIDIA_TESLA_T4',
    accelerator_count=1,
    args=['--epochs', '50', '--batch-size', '32']
)
```

### 4.3 Hyperparameter Tuning (Vertex AI Vizier)

**What to Tune:**
- **Learning rate:** 0.0001 to 0.1 (log scale)
- **Batch size:** 16, 32, 64, 128
- **Number of layers:** 2 to 5
- **Hidden units:** 64, 128, 256, 512
- **Dropout rate:** 0.0 to 0.5
- **Regularization (L2):** 0.0001 to 0.1

**Vertex AI Hyperparameter Tuning:**
```python
from google.cloud import aiplatform

hpt_job = aiplatform.HyperparameterTuningJob(
    display_name='hpt-job',
    custom_job=job,
    metric_spec={
        'accuracy': 'maximize'  # Optimization goal
    },
    parameter_spec={
        'learning_rate': aiplatform.hyperparameter_tuning.DoubleParameterSpec(
            min=0.0001, max=0.1, scale='log'
        ),
        'batch_size': aiplatform.hyperparameter_tuning.DiscreteParameterSpec(
            values=[16, 32, 64, 128]
        ),
        'hidden_units': aiplatform.hyperparameter_tuning.IntegerParameterSpec(
            min=64, max=512, scale='linear'
        )
    },
    max_trial_count=20,
    parallel_trial_count=4,
    search_algorithm='ALGORITHM_UNSPECIFIED'  # Bayesian optimization
)

hpt_job.run()
```

**Search Algorithms:**
- **Grid Search:** Try all combinations (exhaustive, slow)
- **Random Search:** Try random combinations (better than grid)
- **Bayesian Optimization:** Smart search using past results (best for expensive training)

**Exam Tip:** If question mentions "optimize hyperparameters," answer is Vertex AI Vizier (managed hyperparameter tuning).

### 4.4 Transfer Learning

**Pre-trained Models Available:**
- **Vision:** ResNet, EfficientNet, Vision Transformer (ViT)
- **NLP:** BERT, T5, GPT (via Vertex AI Model Garden)
- **Audio:** Speech-to-Text models

**Transfer Learning Strategy:**
```
1. Load pre-trained model (trained on ImageNet/Wikipedia)
2. Freeze early layers (keep learned features)
3. Replace final layer (match your classes)
4. Fine-tune on your data (small learning rate)
```

**When to Use Transfer Learning:**
- ✅ Small dataset (<10K examples)
- ✅ Task similar to pre-training task (images → images, text → text)
- ✅ Want to reduce training time

**When to Train from Scratch:**
- ❌ Very large dataset (>1M examples)
- ❌ Task very different from pre-training (medical images)
- ❌ Need full control of architecture

---

## 📚 DOMAIN 5: ML PIPELINE AUTOMATION & ORCHESTRATION (10-15%)

### 5.1 Vertex AI Pipelines (Kubeflow Pipelines)

**Pipeline Components:**
```python
from kfp.v2 import dsl
from kfp.v2.dsl import component, Input, Output, Dataset, Model

@component(base_image='python:3.9')
def preprocess_data(
    input_data: Input[Dataset],
    output_data: Output[Dataset]
):
    import pandas as pd
    df = pd.read_csv(input_data.path)
    # Preprocessing logic
    df.to_csv(output_data.path, index=False)

@component(base_image='gcr.io/cloud-aiplatform/training/tf-cpu.2-12:latest')
def train_model(
    training_data: Input[Dataset],
    model: Output[Model],
    epochs: int = 10
):
    import tensorflow as tf
    # Training logic
    model.save(model.path)

@dsl.pipeline(name='ml-pipeline')
def pipeline(project_id: str, dataset_id: str):
    preprocess_task = preprocess_data(...)
    train_task = train_model(
        training_data=preprocess_task.outputs['output_data'],
        epochs=50
    )
```

**Compile and Run:**
```python
from kfp.v2 import compiler
from google.cloud import aiplatform

compiler.Compiler().compile(
    pipeline_func=pipeline,
    package_path='pipeline.json'
)

aiplatform.init(project='my-project', location='us-central1')

job = aiplatform.PipelineJob(
    display_name='ml-pipeline',
    template_path='pipeline.json',
    parameter_values={'project_id': 'my-project', 'dataset_id': 'my-dataset'}
)

job.run()
```

**Pipeline Best Practices:**
- ✅ Use components for reusability
- ✅ Parameterize pipelines (avoid hardcoding)
- ✅ Add data validation steps (TFDV)
- ✅ Version control pipeline definitions
- ✅ Monitor pipeline execution (metrics, logs)

### 5.2 Cloud Composer (Apache Airflow)

**When to Use Composer vs Vertex AI Pipelines:**

| Use Case | Tool | Why |
|----------|------|-----|
| ML training pipelines | Vertex AI Pipelines | Native Vertex AI integration |
| Multi-cloud orchestration | Composer (Airflow) | Cloud-agnostic |
| Complex DAG with 100+ tasks | Composer | Better for large DAGs |
| Need Python operators | Composer | Flexible operators |
| ETL + ML in one workflow | Composer | Mix data + ML tasks |
| Simple ML pipeline | Vertex AI Pipelines | Simpler, fully managed |

**Airflow DAG Example:**
```python
from airflow import DAG
from airflow.providers.google.cloud.operators.bigquery import BigQueryExecuteQueryOperator
from airflow.providers.google.cloud.operators.vertex_ai import VertexAITrainingJobRunOperator

with DAG('ml_dag', schedule_interval='@daily') as dag:
    extract_data = BigQueryExecuteQueryOperator(
        task_id='extract',
        sql='SELECT * FROM dataset.table WHERE date = {{ ds }}'
    )
    
    train_model = VertexAITrainingJobRunOperator(
        task_id='train',
        custom_job=custom_job_spec
    )
    
    extract_data >> train_model
```

### 5.3 CI/CD for ML

**Continuous Training Pipeline:**
```
Code Change → Git Commit → Cloud Build Trigger →
Build Container → Run Vertex AI Training Job →
Evaluate Model → (if better) Deploy to Endpoint
```

**Cloud Build Configuration:**
```yaml
steps:
# Build training container
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '-t', 'gcr.io/$PROJECT_ID/trainer:$SHORT_SHA', '.']

# Push container
- name: 'gcr.io/cloud-builders/docker'
  args: ['push', 'gcr.io/$PROJECT_ID/trainer:$SHORT_SHA']

# Run Vertex AI training
- name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
  entrypoint: 'gcloud'
  args:
    - 'ai'
    - 'custom-jobs'
    - 'create'
    - '--region=us-central1'
    - '--display-name=training-job'
    - '--worker-pool-spec=machine-type=n1-standard-4,replica-count=1,container-image-uri=gcr.io/$PROJECT_ID/trainer:$SHORT_SHA'
```

**Model Deployment Workflow:**
1. Train model → Save to Cloud Storage
2. Register model in Vertex AI Model Registry
3. Create Vertex AI Endpoint
4. Deploy model to endpoint (version 1)
5. Test with canary traffic (10%)
6. Gradually increase traffic (50%, 100%)
7. Monitor metrics (latency, error rate, accuracy)

---

## 📚 DOMAIN 6: MONITORING, OPTIMIZING & MAINTAINING ML SOLUTIONS (15-20%)

### 6.1 Model Monitoring (HIGH IMPORTANCE)

**Types of Monitoring:**

1. **Input Data Skew**
   - **Definition:** Training data distribution ≠ Serving data distribution
   - **Example:** Model trained on users age 18-35, but production users are age 50+
   - **Detection:** Compare feature distributions (training vs production)
   - **Solution:** Retrain on recent data, apply domain adaptation

2. **Prediction Drift**
   - **Definition:** Model outputs change over time (even with same inputs)
   - **Example:** Fraud detection model flags more transactions as fraud over time
   - **Detection:** Monitor prediction distribution, alert on shifts
   - **Solution:** Investigate root cause, retrain if needed

3. **Training-Serving Skew**
   - **Definition:** Features computed differently in training vs serving
   - **Example:** Training uses SQL AVERAGE(), serving uses Python mean()
   - **Detection:** Compare feature values (training vs serving)
   - **Solution:** Use Vertex AI Feature Store for consistency

4. **Concept Drift**
   - **Definition:** Relationship between features and target changes
   - **Example:** Customer preferences change (COVID-19 impact on retail)
   - **Detection:** Monitor model accuracy on labeled production data
   - **Solution:** Retrain model with recent data

**Vertex AI Model Monitoring:**
```python
from google.cloud import aiplatform

monitoring_job = aiplatform.ModelDeploymentMonitoringJob.create(
    display_name='monitoring-job',
    endpoint=endpoint,
    logging_sampling_strategy=aiplatform.model_monitoring.RandomSampleConfig(
        sample_rate=0.8
    ),
    schedule_config=aiplatform.model_monitoring.ScheduleConfig(
        monitor_interval=3600  # 1 hour
    ),
    predict_instance_schema_uri='gs://bucket/schema.yaml',
    analysis_instance_schema_uri='gs://bucket/schema.yaml',
    skew_configs=[
        aiplatform.model_monitoring.SkewDetectionConfig(
            data_source='gs://bucket/training_data/',
            skew_thresholds={
                'feature1': 0.3,
                'feature2': 0.5
            },
            target_field='label'
        )
    ],
    drift_configs=[
        aiplatform.model_monitoring.DriftDetectionConfig(
            drift_thresholds={
                'feature1': 0.3,
                'feature2': 0.5
            }
        )
    ]
)
```

**Alerting & Response:**
- Set up Cloud Monitoring alerts (email, PagerDuty, Slack)
- Define alert thresholds (e.g., accuracy drops >5%, drift >0.3)
- Automated response: Trigger retraining pipeline

### 6.2 Model Evaluation

**Offline Evaluation (Before Deployment):**
- **Holdout Test Set:** 80% train, 20% test
- **K-Fold Cross-Validation:** 5-fold or 10-fold
- **Metrics:** Accuracy, precision, recall, F1, AUC-ROC (classification); MAE, RMSE, R² (regression)

**Online Evaluation (After Deployment):**
- **A/B Testing:** Split traffic 50/50 between old model and new model
- **Multi-Armed Bandit:** Dynamically allocate traffic based on performance
- **Shadow Deployment:** Run new model alongside old model (don't serve predictions)

**Vertex AI Model Evaluation:**
```python
# Get evaluation metrics
evaluation = model.evaluate(
    test_dataset=test_data,
    metrics=['accuracy', 'precision', 'recall']
)

print(f"Accuracy: {evaluation['accuracy']}")
print(f"Precision: {evaluation['precision']}")
print(f"Recall: {evaluation['recall']}")

# Confusion matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
plt.show()
```

**When to Retrain:**
- ✅ Model accuracy drops >5% from baseline
- ✅ Data drift detected (distribution shift)
- ✅ New data available (e.g., 6 months of new data)
- ✅ Scheduled retraining (e.g., monthly)
- ❌ Don't retrain: Overfitting on recent outliers

### 6.3 Model Explainability

**Vertex AI Explainable AI:**
- **Feature Attributions:** Which features contributed most to prediction?
- **Methods:**
  - **Integrated Gradients:** For neural networks
  - **XRAI:** For image models (region importance)
  - **Sampled Shapley:** For any model (slower but universal)

**Enable Explainability:**
```python
# During model upload
model = aiplatform.Model.upload(
    display_name='my-model',
    artifact_uri='gs://bucket/model/',
    serving_container_image_uri='gcr.io/cloud-aiplatform/prediction/tf2-cpu.2-12:latest',
    explanation_metadata=aiplatform.explain.ExplanationMetadata(
        inputs={
            'features': aiplatform.explain.ExplanationMetadata.InputMetadata(
                input_tensor_name='features'
            )
        },
        outputs={
            'prediction': aiplatform.explain.ExplanationMetadata.OutputMetadata(
                output_tensor_name='prediction'
            )
        }
    ),
    explanation_parameters=aiplatform.explain.ExplanationParameters(
        integrated_gradients=aiplatform.explain.IntegratedGradientsAttribution(
            step_count=50
        )
    )
)

# Get explanations
explanation = endpoint.explain(instances=[...])
attributions = explanation.attributions[0].feature_attributions
```

**When Explainability is Required:**
- ✅ Regulatory compliance (finance, healthcare)
- ✅ High-stakes decisions (loan approval, medical diagnosis)
- ✅ Debugging model behavior
- ✅ Building trust with stakeholders

### 6.4 Model Optimization

**Latency Optimization:**

1. **Model Quantization**
   - Convert float32 → int8 (4x smaller, faster)
   - **TensorFlow Lite:** For mobile/edge deployment
   ```python
   converter = tf.lite.TFLiteConverter.from_saved_model('model/')
   converter.optimizations = [tf.lite.Optimize.DEFAULT]
   tflite_model = converter.convert()
   ```

2. **Model Pruning**
   - Remove unimportant weights (sparsity)
   - 50-90% size reduction with minimal accuracy loss

3. **Knowledge Distillation**
   - Train small "student" model to mimic large "teacher" model
   - Use when: Large model too slow for production

4. **Hardware Acceleration**
   - **GPU:** 10-100x faster than CPU for inference
   - **TPU:** Optimized for TensorFlow models
   - **TensorRT:** NVIDIA optimization library

**Throughput Optimization:**
- **Batching:** Process multiple requests together
  ```python
  endpoint = model.deploy(
      machine_type='n1-standard-4',
      min_replica_count=1,
      max_replica_count=10,
      batch_size=32,  # Process 32 requests per batch
      max_batch_size=64
  )
  ```
- **Auto-scaling:** Scale replicas based on traffic
  ```python
  endpoint.update(
      min_replica_count=2,
      max_replica_count=20,
      autoscaling_metric='aiplatform.googleapis.com/prediction/online/cpu/utilization',
      autoscaling_target=70  # Scale when CPU >70%
  )
  ```

**Cost Optimization:**
- Use batch predictions for non-time-sensitive workloads
- Use preemptible VMs for training
- Right-size machine types (don't over-provision)
- Set min_replica_count=0 for low-traffic endpoints (scale to zero)

---

## 🎯 GENERATIVE AI & LLMs (NEW in 2025 - 15-20% of Exam)

### 7.1 Vertex AI Generative AI Studio

**Available Models (Model Garden):**
- **PaLM 2 for Text (text-bison):** Text generation, summarization, Q&A
- **PaLM 2 for Chat (chat-bison):** Multi-turn conversations
- **Codey:** Code generation, code completion, code explanation
- **Imagen:** Text-to-image generation
- **Chirp:** Speech-to-text (multilingual)

**Prompt Design:**
```python
from vertexai.language_models import TextGenerationModel

model = TextGenerationModel.from_pretrained('text-bison@002')

response = model.predict(
    prompt="""
    You are a medical expert. Based on the following symptoms, suggest possible diagnoses:
    
    Symptoms: Fever, cough, difficulty breathing
    
    Diagnosis:
    """,
    temperature=0.2,  # Lower = more deterministic
    max_output_tokens=256,
    top_k=40,
    top_p=0.8
)

print(response.text)
```

**Key Parameters:**
- **Temperature (0-1):** 
  - 0 = Deterministic (same output every time)
  - 1 = Creative (varied outputs)
  - Use 0.2 for factual tasks, 0.8 for creative tasks
- **Max Output Tokens:** Limit response length (cost control)
- **Top-k:** Sample from top k probable tokens
- **Top-p (Nucleus Sampling):** Sample from tokens with cumulative probability p

### 7.2 Model Tuning & Fine-Tuning

**When to Tune:**
- ✅ Domain-specific terminology (legal, medical)
- ✅ Specific output format required (JSON, SQL)
- ✅ Consistent style/tone needed (brand voice)
- ❌ Don't tune if prompt engineering works (cheaper, faster)

**Tuning vs Fine-Tuning:**
- **Tuning (Adapter Tuning):** Fast (minutes), small dataset (100-1K examples)
- **Fine-Tuning:** Slower (hours), larger dataset (1K-100K examples), more customization

**Adapter Tuning Example:**
```python
from vertexai.language_models import TextGenerationModel

base_model = TextGenerationModel.from_pretrained('text-bison@002')

tuning_job = base_model.tune_model(
    training_data='gs://bucket/training_data.jsonl',  # JSONL format
    train_steps=100,
    tuning_job_location='us-central1',
    tuned_model_location='us-central1'
)

# Wait for completion
tuned_model = tuning_job.result()

# Use tuned model
response = tuned_model.predict('Your prompt here')
```

**Training Data Format (JSONL):**
```json
{"input_text": "Summarize this medical report: ...", "output_text": "Patient has ..."}
{"input_text": "Translate to medical code: ...", "output_text": "ICD-10: E11.9"}
```

### 7.3 Retrieval-Augmented Generation (RAG)

**Architecture:**
```
User Query → Embedding Model → Vector Search (Matching Engine) →
Retrieved Documents → LLM (with context) → Response
```

**Vertex AI Matching Engine (Vector Database):**
```python
from google.cloud import aiplatform

# Create index
index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name='document-embeddings',
    contents_delta_uri='gs://bucket/embeddings/',
    dimensions=768,  # Embedding dimension
    approximate_neighbors_count=10,
    distance_measure_type='DOT_PRODUCT_DISTANCE'
)

# Deploy index
index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name='document-search',
    public_endpoint_enabled=True
)

index_endpoint.deploy_index(
    index=index,
    deployed_index_id='doc-index-v1'
)

# Search
query_embedding = embedding_model.get_embeddings(['user query'])[0]
response = index_endpoint.find_neighbors(
    deployed_index_id='doc-index-v1',
    queries=[query_embedding],
    num_neighbors=5
)
```

**RAG Pipeline:**
```python
from vertexai.language_models import TextGenerationModel, TextEmbeddingModel

# Step 1: Get embedding of user query
embedding_model = TextEmbeddingModel.from_pretrained('textembedding-gecko@001')
query_embedding = embedding_model.get_embeddings(['What is mitochondria?'])[0]

# Step 2: Find similar documents
similar_docs = index_endpoint.find_neighbors(
    queries=[query_embedding.values],
    num_neighbors=3
)

# Step 3: Retrieve document text
context = '\n\n'.join([get_document_text(doc_id) for doc_id in similar_docs])

# Step 4: Generate response with context
generation_model = TextGenerationModel.from_pretrained('text-bison@002')
prompt = f"""
Answer the question based on the following context:

Context:
{context}

Question: What is mitochondria?

Answer:
"""

response = generation_model.predict(prompt, temperature=0.2)
print(response.text)
```

**When to Use RAG:**
- ✅ Need up-to-date information (LLMs have knowledge cutoff date)
- ✅ Domain-specific knowledge (company docs, research papers)
- ✅ Reduce hallucinations (ground responses in facts)
- ✅ Provide source citations (show which document was used)

### 7.4 Responsible AI for LLMs

**Key Concerns:**
1. **Hallucinations:** LLM generates false information
   - Mitigation: Use RAG, validate facts, add disclaimers
2. **Bias:** LLM reflects biases in training data
   - Mitigation: Test on diverse inputs, apply fairness constraints
3. **Toxicity:** LLM generates harmful content
   - Mitigation: Use content filters, safety classifiers
4. **Privacy:** LLM leaks training data (PII, confidential info)
   - Mitigation: Filter training data, don't include sensitive info
5. **Copyright:** LLM reproduces copyrighted text
   - Mitigation: Check for verbatim copying, add attribution

**Vertex AI Safety Attributes:**
```python
from vertexai.language_models import TextGenerationModel

model = TextGenerationModel.from_pretrained('text-bison@002')

response = model.predict(
    prompt='Your prompt here',
    temperature=0.8
)

# Check safety attributes
print(response.safety_attributes)
# Output: {
#   'categories': ['Politics', 'Finance'],
#   'blocked': False,
#   'scores': [0.2, 0.1]  # Low scores = safe
# }
```

**Best Practices:**
- ✅ Always show confidence scores or uncertainty
- ✅ Provide source citations (RAG)
- ✅ Add disclaimers for medical/legal/financial advice
- ✅ Test on diverse demographics (fairness)
- ✅ Implement human-in-the-loop for high-stakes decisions
- ✅ Monitor outputs for toxicity, bias

**Exam Tip:** Expect 2-3 questions on responsible AI, especially bias detection and mitigation strategies.

---

## 🔥 HIGH-FREQUENCY EXAM TOPICS (Must Know Cold)

1. **Vertex AI vs Legacy AI Platform** (Know new terminology)
2. **When to use AutoML vs Custom Training** (Dataset size, expertise)
3. **BigQuery ML use cases** (SQL interface, when appropriate)
4. **Preemptible VMs for training** (Cost optimization)
5. **Vertex AI Feature Store** (Prevents training-serving skew)
6. **Model monitoring (skew, drift)** (Detection and mitigation)
7. **Imbalanced data handling** (Class weights, resampling)
8. **Hyperparameter tuning** (Vertex AI Vizier)
9. **Transfer learning** (When to use, how to implement)
10. **RAG architecture** (LLM + vector database)
11. **Model explainability** (Regulatory requirements, techniques)
12. **Pipeline orchestration** (Vertex AI Pipelines vs Composer)
13. **Data processing tools** (BigQuery vs Dataflow vs Dataproc)
14. **Responsible AI** (Bias, fairness, explainability)
15. **Cost optimization strategies** (Batch vs online, preemptible, right-sizing)

---

## 📝 EXAM STRATEGY

### Time Management
- **60 questions in 120 minutes = 2 minutes per question**
- **First pass (60 min):** Answer all questions you're confident about, flag uncertain ones
- **Second pass (40 min):** Review flagged questions, use elimination
- **Final pass (20 min):** Double-check marked questions, review for careless errors

### Elimination Strategy
- Read question stem carefully (identify key requirements)
- Eliminate obviously wrong answers (violates a requirement)
- Between 2 similar answers: Choose the "GCP-managed" option (less operational overhead)
- If stuck: Choose the simplest solution that meets requirements

### Common Traps
- ❌ Overcomplicating (exam favors managed services over DIY)
- ❌ Ignoring cost constraints (if question mentions "minimize cost," it's important)
- ❌ Missing "need low latency" requirement (changes architecture)
- ❌ Confusing terminology (AI Platform vs Vertex AI)

### Question Types

**1. Scenario-Based (80% of exam):**
> "You work for a retail company. You have 10M customer records. You need to predict customer churn with <100ms latency. What should you do?"

**Approach:**
- Extract requirements: 10M records (large), <100ms (real-time), classification
- Eliminate options that don't meet requirements
- Choose most managed solution

**2. Technical Implementation (15%):**
> "Which hyperparameter tuning algorithm is best for expensive training jobs?"

**Approach:**
- Direct knowledge question
- Answer: Bayesian optimization (Vertex AI Vizier default)

**3. Conceptual (5%):**
> "What is training-serving skew?"

**Approach:**
- Definition-based question
- Answer: Features computed differently in training vs serving

---

## ✅ FINAL CHECKLIST (Week Before Exam)

**Technical Skills:**
- [ ] Can explain Vertex AI architecture (AutoML, Custom Training, Pipelines, etc.)
- [ ] Know when to use BigQuery ML vs Vertex AI
- [ ] Understand Feature Store purpose and use cases
- [ ] Can differentiate skew vs drift vs concept drift
- [ ] Know model evaluation metrics (precision, recall, F1, AUC-ROC)
- [ ] Understand imbalanced data strategies
- [ ] Can explain hyperparameter tuning process
- [ ] Know transfer learning workflow
- [ ] Understand RAG architecture for LLMs
- [ ] Can explain responsible AI concerns (bias, fairness, explainability)

**Operational Knowledge:**
- [ ] Know cost optimization strategies (preemptible VMs, batch predictions)
- [ ] Understand when to use Dataflow vs Dataproc vs BigQuery
- [ ] Can explain Vertex AI Pipelines workflow
- [ ] Know model monitoring setup (skew detection, drift detection)
- [ ] Understand model deployment strategies (A/B testing, canary)

**Exam Mechanics:**
- [ ] Registered for exam on Webassessor
- [ ] Tested computer and internet (if remote)
- [ ] Reviewed exam policies (no phone, no notes)
- [ ] Planned time (arrive 15 min early if in-person)

---

## 🎓 STUDY PLAN (4-Week Intensive)

**Week 1: Foundations**
- Read official exam guide
- Review Vertex AI documentation
- Practice with BigQuery ML (5 sample models)
- Study: Framing ML problems, architecting solutions

**Week 2: Deep Dive**
- Hands-on: Build Vertex AI Custom Training job
- Hands-on: Deploy model to endpoint
- Hands-on: Set up model monitoring
- Study: Data preparation, model development

**Week 3: Advanced Topics**
- Hands-on: Build Vertex AI Pipeline
- Hands-on: Implement RAG with Matching Engine
- Study: Generative AI, LLMs, responsible AI
- Study: Pipeline orchestration, optimization

**Week 4: Practice & Review**
- Practice exams (ExamTopics, Whizlabs)
- Review weak areas
- Memorize high-frequency topics
- Final review day before exam

---

## 📖 ADDITIONAL RESOURCES

**Official Documentation:**
- Vertex AI Documentation: https://cloud.google.com/vertex-ai/docs
- BigQuery ML: https://cloud.google.com/bigquery-ml/docs
- Generative AI Studio: https://cloud.google.com/vertex-ai/docs/generative-ai/learn/overview

**Practice Questions:**
- ExamTopics: https://www.examtopics.com/exams/google/professional-machine-learning-engineer/
- Whizlabs: https://www.whizlabs.com/google-cloud-certified-professional-machine-learning-engineer/
- Official Sample: Search "Google PMLE sample questions"

**Hands-On Labs:**
- Google Cloud Skills Boost: https://www.cloudskillsboost.google/paths/17
- Coursera: "Machine Learning with TensorFlow on Google Cloud"

**Books:**
- "Machine Learning Design Patterns" by Lakshmanan, Robinson, Munn
- "Machine Learning System Design" by Chip Huyen

---

**Good luck on your exam! With focused preparation, you can pass in 4 weeks.** 🚀
