# Cloud Platforms for AI

Cloud platforms provide managed infrastructure for training, deploying, and scaling AI workloads. In 2025, all three major clouds offer competitive AI-specific services.

---

## Platform Comparison

| Feature | AWS SageMaker | Google Vertex AI | Azure ML |
|---------|--------------|-----------------|---------|
| Managed training | ✅ | ✅ | ✅ |
| Pretrained model hub | SageMaker JumpStart | Model Garden | Azure AI Foundry |
| AutoML | Autopilot | AutoML | Automated ML |
| Feature store | Feature Store | Feature Store | Feature Store |
| MLflow support | ✅ | ✅ | ✅ (native) |
| Spot instance training | ✅ | Preemptible | Spot VMs |
| Best for | AWS-heavy orgs | GCP/TPU workloads | Microsoft/enterprise |

---

## GPU Instance Comparison (2025)

| Instance | GPU | VRAM | Cost/hr | Best For |
|---------|-----|------|---------|---------|
| `p4d.24xlarge` (AWS) | 8× A100 80GB | 640GB | ~$32 | LLM training |
| `p5.48xlarge` (AWS) | 8× H100 80GB | 640GB | ~$98 | State-of-the-art |
| `g5.xlarge` (AWS) | A10G 24GB | 24GB | ~$1.0 | Inference, fine-tune |
| `g4dn.xlarge` (AWS) | T4 16GB | 16GB | ~$0.53 | Cost-efficient inference |
| `a2-highgpu-1g` (GCP) | A100 40GB | 40GB | ~$3.7 | Training |
| `Standard_NC6s_v3` (Azure) | V100 16GB | 16GB | ~$3.0 | General training |

**Cost optimization tips:**
- Use **spot/preemptible** instances for fault-tolerant training: 60-90% cheaper
- Use **g5.xlarge** for most fine-tuning (A10G is great value)
- Use **p4d** only when you need > 80GB VRAM

---

## AWS SageMaker

```python
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, IntegerParameter

sess = sagemaker.Session()
role = sagemaker.get_execution_role()
bucket = sess.default_bucket()

# ── Training Job ───────────────────────────────────────────────────
estimator = PyTorch(
    entry_point="train.py",
    source_dir="./src",
    role=role,
    instance_type="ml.g5.2xlarge",   # A10G GPU
    instance_count=1,
    framework_version="2.2",
    py_version="py310",
    hyperparameters={
        "epochs": 30,
        "batch-size": 64,
        "learning-rate": 0.001,
    },
    use_spot_instances=True,          # Save 70%
    max_wait=7200,                    # Max wait for spot
    checkpoint_s3_uri=f"s3://{bucket}/checkpoints",
    environment={"TRANSFORMERS_CACHE": "/tmp/hf_cache"},
)

estimator.fit({"train": f"s3://{bucket}/data/train",
               "val":   f"s3://{bucket}/data/val"})

# ── Hyperparameter Tuning ──────────────────────────────────────────
tuner = HyperparameterTuner(
    estimator=estimator,
    objective_metric_name="val:accuracy",
    objective_type="Maximize",
    hyperparameter_ranges={
        "learning-rate": ContinuousParameter(1e-4, 1e-1, scaling_type="Logarithmic"),
        "batch-size":    IntegerParameter(16, 128),
    },
    max_jobs=20,
    max_parallel_jobs=4,
    strategy="Bayesian",
)
tuner.fit({"train": f"s3://{bucket}/data/train"})

# ── Deploy Endpoint ───────────────────────────────────────────────
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.xlarge",
    endpoint_name="my-model-endpoint",
)

# Auto-scaling
import boto3
aas = boto3.client("application-autoscaling")
aas.register_scalable_target(
    ServiceNamespace="sagemaker",
    ResourceId=f"endpoint/my-model-endpoint/variant/AllTraffic",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    MinCapacity=1,
    MaxCapacity=10,
)
```

### SageMaker JumpStart (Foundation Models)
```python
from sagemaker.jumpstart.model import JumpStartModel

# Deploy Llama 3 in one line
model = JumpStartModel(model_id="meta-textgeneration-llama-3-70b-instruct")
predictor = model.deploy(
    accept_eula=True,
    instance_type="ml.g5.48xlarge",
)

response = predictor.predict({"inputs": "Explain RAG in simple terms"})
```

---

## Google Vertex AI

```python
from google.cloud import aiplatform
from google.cloud.aiplatform import gapic

aiplatform.init(project="my-project", location="us-central1")

# ── Custom Training ───────────────────────────────────────────────
job = aiplatform.CustomTrainingJob(
    display_name="pytorch-classifier",
    script_path="train.py",
    container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-2:latest",
    requirements=["transformers", "datasets", "accelerate"],
    model_serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.2-2:latest",
)

model = job.run(
    dataset=None,
    replica_count=1,
    machine_type="n1-standard-8",
    accelerator_type="NVIDIA_TESLA_A100",
    accelerator_count=1,
    args=["--epochs=50", "--batch-size=64"],
    base_output_dir=f"gs://my-bucket/output",
)

# ── Vertex AI Pipelines (Kubeflow) ────────────────────────────────
from kfp import dsl
from kfp.registry import RegistryClient

@dsl.component(base_image="python:3.11", packages_to_install=["scikit-learn"])
def train_component(data_path: str, model_output: dsl.Output[dsl.Model]):
    from sklearn.ensemble import RandomForestClassifier
    import joblib, pandas as pd
    df = pd.read_csv(data_path)
    X, y = df.drop("label", axis=1), df["label"]
    clf = RandomForestClassifier(n_estimators=200)
    clf.fit(X, y)
    joblib.dump(clf, model_output.path)

@dsl.pipeline(name="ml-pipeline")
def my_pipeline(data_gcs_path: str):
    train_task = train_component(data_path=data_gcs_path)

# Deploy pipeline
from google.cloud.aiplatform import PipelineJob
PipelineJob(
    display_name="ml-pipeline",
    template_path="pipeline.yaml",
    parameter_values={"data_gcs_path": "gs://my-bucket/data.csv"},
).run()
```

### Vertex AI Model Garden
```python
# Access Gemini, Claude, Llama, and more
import vertexai
from vertexai.generative_models import GenerativeModel

vertexai.init(project="my-project", location="us-central1")

model = GenerativeModel("gemini-2.0-flash-001")
response = model.generate_content("Summarize the latest AI research trends")
print(response.text)
```

---

## Azure Machine Learning

```python
from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment, AmlCompute
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="...",
    resource_group_name="...",
    workspace_name="my-workspace",
)

# Create GPU compute cluster
compute = AmlCompute(
    name="gpu-cluster",
    type="amlcompute",
    size="Standard_NC6s_v3",   # V100
    min_instances=0,            # Scale to zero when idle
    max_instances=4,
    tier="Dedicated",
    # tier="LowPriority",       # Spot: 60-80% cheaper
)
ml_client.begin_create_or_update(compute).result()

# Submit training job
job = command(
    code="./src",
    command="python train.py --epochs ${{inputs.epochs}} --lr ${{inputs.lr}}",
    inputs={"epochs": 50, "lr": 0.001},
    environment="azureml:AzureML-PyTorch-2.0-GPU:1",
    compute="gpu-cluster",
    display_name="pytorch-training",
    experiment_name="image-classification",
)

returned_job = ml_client.jobs.create_or_update(job)
ml_client.jobs.stream(returned_job.name)  # Stream logs

# Deploy model
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment

endpoint = ManagedOnlineEndpoint(name="my-endpoint", auth_mode="key")
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name="my-endpoint",
    model=returned_job.outputs.model,
    instance_type="Standard_DS3_v2",
    instance_count=1,
)
ml_client.online_deployments.begin_create_or_update(deployment).result()
```

---

## Storage for ML Artifacts

```python
# AWS S3
import boto3
s3 = boto3.client("s3")

# Upload dataset
s3.upload_file("data.parquet", "my-bucket", "datasets/data.parquet")

# Stream large dataset without downloading
import pandas as pd
df = pd.read_parquet("s3://my-bucket/datasets/data.parquet",
                     storage_options={"key": "...", "secret": "..."})

# Google Cloud Storage
from google.cloud import storage
gcs = storage.Client()
bucket = gcs.bucket("my-bucket")
blob = bucket.blob("models/resnet50.pt")
blob.upload_from_filename("resnet50.pt")

# Azure Blob
from azure.storage.blob import BlobServiceClient
client = BlobServiceClient.from_connection_string("...")
blob_client = client.get_blob_client("my-container", "data.parquet")
with open("data.parquet", "rb") as f:
    blob_client.upload_blob(f, overwrite=True)
```

---

## Cost Optimization Tips

| Strategy | Savings | Notes |
|---------|---------|-------|
| Spot/Preemptible instances | 60-90% | Implement checkpointing |
| Right-size instances | 30-50% | Profile actual GPU utilization |
| fp16/bf16 training | Memory → smaller instance | Use `torch.amp` |
| Gradient checkpointing | Memory → smaller instance | `model.gradient_checkpointing_enable()` |
| Batch inference | 10-20x throughput | Vs. single-sample inference |
| Model quantization | Smaller instance | INT8 serving with vLLM |
| Auto-scaling to zero | ~100% for idle | Set `min_instances=0` |

---

## Related Topics

- [MLOps Overview](index.md)
- [Containerization with Docker](index.md#containerization-with-docker)
- [Serving Models](index.md#serving-models)
- [Fine-Tuning LLMs](../llm/fine-tuning.md)
