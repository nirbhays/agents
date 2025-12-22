# GCP Developer + DevOps — Consolidated Cheat Sheet

Use this as a rapid refresher during practice and right before the exams.

---

## Service Selection
- Compute:
  - Cloud Run: container, HTTP, autoscale to zero, traffic splitting
  - App Engine: PaaS, traffic splitting, versions, standard vs flexible
  - GKE: full K8s control, node/pod autoscaling, BinAuthZ
  - Cloud Functions: event-driven, lightweight handlers
- Data:
  - Cloud SQL (relational), Firestore (NoSQL doc), Spanner (global SQL), Memorystore (Redis)
- Integration:
  - Pub/Sub (async events, DLQ, ordering keys), API Gateway/Endpoints
- Observability:
  - Logging, Monitoring, Error Reporting, Trace, Profiler, Debugger

---

## CI/CD Essentials
- Artifact Registry: docker, maven, npm; vulnerability scanning
- Cloud Build: steps in `cloudbuild.yaml`; private pools for VPC build
- Cloud Deploy: delivery pipelines with canary/blue-green; approvals
- Provenance/SLSA: generate attestations; Binary Authorization on GKE

Common commands:
```bash
# Build & push
gcloud artifacts repositories create repo --repository-format=docker --location=$REGION
gcloud builds submit --tag $REGION-docker.pkg.dev/$PROJECT/repo/app:latest

# Deploy Cloud Run
gcloud run deploy app --image=$REGION-docker.pkg.dev/$PROJECT/repo/app:latest --region=$REGION --allow-unauthenticated

# Split traffic
gcloud run services update-traffic app --to-revisions REV1=90,REV2=10 --region=$REGION

# Pub/Sub with DLQ
gcloud pubsub topics create topic
gcloud pubsub topics create dlq
gcloud pubsub subscriptions create sub --topic=topic --dead-letter-topic=dlq --max-delivery-attempts=5
```

---

## Security Quick Hits
- IAM: least privilege; use service accounts per service
- Secret Manager: no plaintext env secrets; grant accessor to service identity
- Workload Identity: avoid long-lived keys
- CMEK: enable where required (Spanner/BigQuery/GCS options)
- API Gateway: JWT/OIDC auth; per-route auth

---

## SRE Metrics & Alerts
- SLIs: availability (%), latency (p50/p95/p99), error rate, saturation
- SLO example: 99.9% availability over 30 days
- Error budget: 0.1% of total time in window
- Burn-rate alert (example): 5m@14x + 1h@4x → page; 6h@2x → ticket

---

## Troubleshooting Playbook
- High error rate: check logs for 5xx, recent rollouts; rollback; examine dependencies
- Latency spike: check p95/p99, trace waterfall; hot pods, DB contention, throttling
- Pub/Sub lag: inspect subscription backlog, consumer throughput; add partitions/scale workers
- Cloud Run cold starts: set min instances; reduce image size; tune concurrency
- DB saturation: add read replicas, caching, connection pool sizing; review slow queries

---

## Data Store Selection
- Cloud SQL: relational joins/transactions, regional HA, up to multi-zone
- Firestore: flexible schema, global multi-region, strong/eventual consistency modes
- Spanner: global scale, strong consistency, horizontal scaling
- Memorystore: low-latency cache; avoid as source of truth

---

## Networking
- VPC: subnets, firewall rules, routes; Shared VPC for centralized networking
- Private access: Serverless VPC Access (Cloud Run → Cloud SQL/hosted services)
- Load balancing: Global HTTP(S) LB for web; internal LBs for east-west
- Private Service Connect: consume producer services privately

---

## Cost & Performance
- Right-size: Cloud Run concurrency, GKE requests/limits; min instances for latency
- Caching/CDN: Memorystore, Cloud CDN for static and cacheable dynamic content
- Storage class lifecycle: Standard → Nearline → Coldline/Archive
- Budgets/alerts: set per project; labels for cost allocation

---

## Exam Tactics
- Read the question twice; highlight constraints (region, compliance, RTO/RPO)
- Prefer managed serverless (Cloud Run/Functions) unless explicit K8s control needed
- Choose least-privilege + secret manager + workload identity over keys
- For reliability, mention retries + timeouts + idempotency + DLQ + rollbacks
- Tie answers to SLOs, error budgets, and observability
