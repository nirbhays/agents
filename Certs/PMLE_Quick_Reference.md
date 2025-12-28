# PMLE Quick Reference Cheat Sheet

**Exam:** Google Professional Machine Learning Engineer  
**Format:** 60 questions, 120 minutes  
**Pass:** ~70-75% (42-45 correct)

---

## 🎯 DECISION TREES

### Should I Use AutoML or Custom Training?

```
Do you have ML expertise?
├─ NO → Use AutoML
└─ YES → Do you need full control?
    ├─ YES → Custom Training
    └─ NO → Dataset size?
        ├─ <10K rows → AutoML (better with small data)
        └─ >10K rows → Custom Training (more cost-effective at scale)
```

### Which Storage for ML Data?

```
Data type?
├─ Structured (tables) → Volume?
│   ├─ <10 GB → Cloud SQL
│   └─ >10 GB → BigQuery
├─ Unstructured (images/files) → Cloud Storage
├─ Real-time streams → Pub/Sub → Dataflow
└─ Low-latency key-value → Bigtable
```

### Which Data Processing Tool?

```
Processing type?
├─ SQL analytics → BigQuery
├─ Real-time streaming → Dataflow
├─ Existing Spark code → Dataproc
├─ Simple transforms → Cloud Functions
└─ Orchestration → Cloud Composer (Airflow)
```

### Online vs Batch Predictions?

```
Latency requirement?
├─ <1 second (real-time) → Online prediction (endpoint)
└─ Can wait (hours/days) → Batch prediction (cheaper)
```

---

## 📊 VERTEX AI COMPONENTS (CRITICAL)

### AutoML Products

| Product | Use Case | Max Dataset Size |
|---------|----------|------------------|
| **AutoML Tables** | Structured data (regression, classification) | 100 GB |
| **AutoML Vision** | Image classification, object detection | 10K-1M images |
| **AutoML Natural Language** | Text classification, entity extraction | 1M documents |
| **AutoML Video** | Video classification, action recognition | 100K hours |

### Custom Training

```python
from google.cloud import aiplatform

job = aiplatform.CustomTrainingJob(
    display_name='training-job',
    script_path='trainer/task.py',
    container_uri='gcr.io/cloud-aiplatform/training/tf-cpu.2-12:latest',
    requirements=['pandas', 'scikit-learn']
)

model = job.run(
    replica_count=1,
    machine_type='n1-standard-4',
    accelerator_type='NVIDIA_TESLA_T4',  # GPU
    accelerator_count=1
)
```

### Machine Types (Memorize Common Ones)

| Type | vCPUs | RAM | Use Case | Cost/hr |
|------|-------|-----|----------|---------|
| **n1-standard-4** | 4 | 15 GB | Small models | $0.19 |
| **n1-highmem-8** | 8 | 52 GB | Medium models | $0.47 |
| **a2-highgpu-1g** | 12 | 85 GB + 1 A100 | Deep learning | $4.29 |

### Accelerators

| Accelerator | Use Case | Cost/hr |
|-------------|----------|---------|
| **T4** | Training small models | $0.35 |
| **V100** | Standard deep learning | $2.48 |
| **A100** | Large models, faster training | $3.67 |
| **TPU v3** | TensorFlow at scale | $8.00 |

---

## 💰 COST OPTIMIZATION (High-Frequency Topic)

### Strategies

| Strategy | Savings | Use When |
|----------|---------|----------|
| **Preemptible VMs** | 80% | Fault-tolerant training (with checkpointing) |
| **Spot VMs** | 70-80% | More stable than preemptible |
| **Batch predictions** | 90% vs online | Predictions can wait (daily/weekly) |
| **Right-size machines** | 30-50% | Match resources to workload |
| **Committed use (1yr)** | 25-37% | Production endpoints |
| **Committed use (3yr)** | 52-57% | Long-term production |

### Cost Formula

**Training Cost:**
```
Cost = (Machine cost/hr + GPU cost/hr) × Training hours × Replica count
```

**Online Prediction Cost:**
```
Cost = Machine cost/hr × 24 hrs × 30 days × Replica count
```

**Batch Prediction Cost:**
```
Cost = Machine cost/hr × Processing hours
```

---

## 📈 MODEL EVALUATION METRICS

### Classification

| Metric | Formula | Use When |
|--------|---------|----------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Balanced classes |
| **Precision** | TP/(TP+FP) | False positives costly (fraud) |
| **Recall** | TP/(TP+FN) | False negatives costly (cancer) |
| **F1 Score** | 2×(Precision×Recall)/(Precision+Recall) | Balance precision & recall |
| **AUC-ROC** | Area under ROC curve | Overall performance |

**Confusion Matrix:**
```
                Predicted
              Positive  Negative
Actual  Pos     TP        FN
        Neg     FP        TN
```

### Regression

| Metric | What It Measures | Lower = Better |
|--------|------------------|----------------|
| **MAE** | Average absolute error | ✅ |
| **RMSE** | Root mean squared error (penalizes large errors) | ✅ |
| **MAPE** | Mean absolute percentage error | ✅ |
| **R² Score** | Variance explained (0-1) | Higher = Better |

---

## 🔧 FEATURE ENGINEERING

### Numerical Features

| Technique | When to Use |
|-----------|-------------|
| **Min-max scaling** | Neural networks (scale to [0,1]) |
| **Standardization (Z-score)** | Linear models, features with different scales |
| **Log transformation** | Skewed distributions (e.g., income) |
| **Binning** | Convert continuous → categorical |

### Categorical Features

| Technique | When to Use | Example |
|-----------|-------------|---------|
| **One-hot encoding** | <50 categories | Color: Red, Blue → [1,0], [0,1] |
| **Label encoding** | Ordinal data | Rating: Low, Med, High → 0, 1, 2 |
| **Target encoding** | High cardinality (>50) | Replace with target mean |
| **Embedding** | Very high cardinality (>1000) | User ID, Product ID |

### Text Features

| Technique | What It Does |
|-----------|--------------|
| **TF-IDF** | Weight words by importance |
| **Word2Vec** | Dense word representations |
| **BERT** | Contextual embeddings |

---

## 🚨 IMBALANCED DATA

### Problem
99% class A, 1% class B → Model predicts all A, gets 99% accuracy but fails

### Solutions

| Technique | How It Works | Pros | Cons |
|-----------|--------------|------|------|
| **Oversampling** | Duplicate minority class | Simple | Overfitting risk |
| **Undersampling** | Remove majority class | Fast | Lose information |
| **SMOTE** | Synthetic minority examples | Better than oversampling | Computational cost |
| **Class weights** | Penalize minority errors more | No data change needed | Need to tune weights |

**Class Weights Example:**
```python
# If 99% class 0, 1% class 1
class_weight = {0: 1, 1: 99}  # Class 1 is 99x more important

model.fit(X, y, class_weight=class_weight)
```

---

## 🔍 MODEL MONITORING (Very Important)

### Types of Drift

| Type | Definition | Example | Detection |
|------|------------|---------|-----------|
| **Input Skew** | Training dist ≠ Serving dist | Trained on age 18-35, serving 50+ | Compare feature distributions |
| **Training-Serving Skew** | Features computed differently | SQL AVG() vs Python mean() | Use Feature Store |
| **Prediction Drift** | Output distribution changes | More fraud predictions over time | Monitor prediction dist |
| **Concept Drift** | Feature-target relationship changes | Customer preferences shift | Monitor accuracy on labeled data |

### Vertex AI Model Monitoring

```python
monitoring_job = aiplatform.ModelDeploymentMonitoringJob.create(
    endpoint=endpoint,
    schedule_config=aiplatform.model_monitoring.ScheduleConfig(
        monitor_interval=3600  # Check every hour
    ),
    skew_configs=[
        aiplatform.model_monitoring.SkewDetectionConfig(
            data_source='gs://bucket/training_data/',
            skew_thresholds={'feature1': 0.3}  # Alert if >0.3 skew
        )
    ],
    drift_configs=[
        aiplatform.model_monitoring.DriftDetectionConfig(
            drift_thresholds={'feature1': 0.3}  # Alert if >0.3 drift
        )
    ]
)
```

---

## 🤖 GENERATIVE AI (15-20% of Exam)

### Vertex AI Models

| Model | Use Case | Input | Output |
|-------|----------|-------|--------|
| **text-bison** | Text generation, Q&A | Text prompt | Text |
| **chat-bison** | Multi-turn chat | Conversation | Text |
| **Codey** | Code generation | Text description | Code |
| **Imagen** | Image generation | Text prompt | Image |

### Prompt Engineering

**Temperature:**
- **0-0.3:** Deterministic (factual tasks, code)
- **0.4-0.7:** Balanced (general tasks)
- **0.8-1.0:** Creative (stories, brainstorming)

```python
from vertexai.language_models import TextGenerationModel

model = TextGenerationModel.from_pretrained('text-bison@002')

response = model.predict(
    prompt='Explain quantum computing in simple terms:',
    temperature=0.2,  # More deterministic
    max_output_tokens=256
)
```

### RAG (Retrieval-Augmented Generation)

**Architecture:**
```
User Query → Embedding → Vector Search → Retrieved Docs → LLM → Response
```

**When to Use RAG:**
- ✅ Need up-to-date information
- ✅ Domain-specific knowledge (company docs)
- ✅ Reduce hallucinations
- ✅ Provide source citations

**Vertex AI Matching Engine (Vector DB):**
```python
# Create index
index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name='embeddings',
    contents_delta_uri='gs://bucket/embeddings/',
    dimensions=768
)

# Search
response = index_endpoint.find_neighbors(
    queries=[query_embedding],
    num_neighbors=5
)
```

---

## 🔄 ML PIPELINES

### Vertex AI Pipelines (Kubeflow)

```python
from kfp.v2 import dsl
from kfp.v2.dsl import component

@component
def preprocess_data(input_data: Input[Dataset], output_data: Output[Dataset]):
    # Preprocessing logic
    pass

@component
def train_model(training_data: Input[Dataset], model: Output[Model]):
    # Training logic
    pass

@dsl.pipeline(name='ml-pipeline')
def pipeline():
    preprocess_task = preprocess_data(...)
    train_task = train_model(training_data=preprocess_task.outputs['output_data'])
```

### Vertex AI Pipelines vs Cloud Composer

| Use Case | Tool | Why |
|----------|------|-----|
| ML training pipelines | Vertex AI Pipelines | Native Vertex AI integration |
| Multi-cloud orchestration | Cloud Composer | Cloud-agnostic |
| Complex DAG (100+ tasks) | Cloud Composer | Better for large DAGs |
| Simple ML pipeline | Vertex AI Pipelines | Simpler, fully managed |

---

## 🎓 HYPERPARAMETER TUNING

### What to Tune

| Hyperparameter | Typical Range | Scale |
|----------------|---------------|-------|
| **Learning rate** | 0.0001 - 0.1 | Log |
| **Batch size** | 16, 32, 64, 128 | Discrete |
| **Hidden units** | 64 - 512 | Linear |
| **Dropout rate** | 0.0 - 0.5 | Linear |
| **L2 regularization** | 0.0001 - 0.1 | Log |

### Search Algorithms

| Algorithm | Speed | Quality | Use When |
|-----------|-------|---------|----------|
| **Grid search** | Slowest | Good | Few hyperparameters |
| **Random search** | Fast | Good | Many hyperparameters |
| **Bayesian optimization** | Medium | Best | Expensive training |

**Vertex AI Vizier (default = Bayesian):**
```python
hpt_job = aiplatform.HyperparameterTuningJob(
    custom_job=job,
    metric_spec={'accuracy': 'maximize'},
    parameter_spec={
        'learning_rate': aiplatform.hyperparameter_tuning.DoubleParameterSpec(
            min=0.0001, max=0.1, scale='log'
        )
    },
    max_trial_count=20,
    parallel_trial_count=4
)
```

---

## 🛡️ RESPONSIBLE AI

### Key Concerns

| Concern | Definition | Mitigation |
|---------|------------|------------|
| **Bias** | Unfair treatment of groups | Test on diverse data, fairness constraints |
| **Explainability** | Understanding predictions | SHAP, LIME, Integrated Gradients |
| **Privacy** | Leaking sensitive data | Differential privacy, federated learning |
| **Hallucinations** | Generating false info | RAG, fact-checking, disclaimers |
| **Toxicity** | Harmful content | Content filters, safety classifiers |

### Vertex AI Explainable AI

**Methods:**
- **Integrated Gradients:** For neural networks
- **XRAI:** For image models
- **Sampled Shapley:** Universal (slower)

```python
model = aiplatform.Model.upload(
    explanation_parameters=aiplatform.explain.ExplanationParameters(
        integrated_gradients=aiplatform.explain.IntegratedGradientsAttribution(
            step_count=50
        )
    )
)

# Get explanations
explanation = endpoint.explain(instances=[...])
```

---

## 🎯 EXAM TIPS

### Time Management
- **2 minutes per question** (60 questions, 120 minutes)
- **First pass (60 min):** Answer confident questions
- **Second pass (40 min):** Review flagged questions
- **Final pass (20 min):** Double-check

### Elimination Strategy
1. Read requirements carefully
2. Eliminate obviously wrong answers
3. Between 2 similar answers: Choose managed service
4. If stuck: Choose simplest solution

### Common Traps
- ❌ Overcomplicating (favor managed services)
- ❌ Ignoring cost constraints
- ❌ Missing latency requirements
- ❌ Confusing AI Platform (old) vs Vertex AI (new)

### High-Frequency Topics
1. Vertex AI components (AutoML, Custom Training, Pipelines)
2. Cost optimization (preemptible VMs, batch predictions)
3. Feature Store (training-serving skew)
4. Model monitoring (skew, drift)
5. Imbalanced data (class weights, SMOTE)
6. Hyperparameter tuning (Vertex AI Vizier)
7. BigQuery ML (when to use)
8. RAG architecture (LLM + vector DB)
9. Responsible AI (bias, explainability)
10. Data processing tools (BigQuery vs Dataflow vs Dataproc)

---

## 🔑 MEMORIZE THESE

**Vertex AI Terminology (2025):**
- ✅ Vertex AI (new) ❌ AI Platform (legacy)
- ✅ Vertex AI Endpoints ❌ AI Platform Predictions
- ✅ Vertex AI Pipelines ❌ AI Platform Pipelines

**When to Retrain:**
- ✅ Accuracy drops >5%
- ✅ Data drift detected
- ✅ New data available (6+ months)
- ❌ Don't retrain on outliers

**Feature Store Benefits:**
- Prevents training-serving skew
- Feature reusability
- Point-in-time correctness
- Low-latency serving

**Model Explainability Required:**
- Regulatory compliance (finance, healthcare)
- High-stakes decisions (loans, medical)
- Debugging model behavior
- Building stakeholder trust

---

**Print this and review the night before exam!** 🚀
