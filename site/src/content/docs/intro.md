---
title: Lakebase Cookbook
sidebar_label: Introduction
sidebar_position: 1
description: Examples and guides to accelerate your Databricks Lakebase projects.
---

# Lakebase Cookbook

Examples and guides to accelerate your **Databricks Lakebase** projects.

Lakebase is Databricks' fully-managed, Postgres-compatible OLTP database, built
for transactional workloads that sit alongside your lakehouse.

- **Announcement blog:** [databricks.com/product/lakebase](https://www.databricks.com/product/lakebase)
- **Official documentation:** [docs.databricks.com/aws/en/oltp](https://docs.databricks.com/aws/en/oltp/)

## Examples in this cookbook

| Example                                                  | Description                                                                                                                                                   |
|----------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [GraphRAG](/docs/examples/graphrag/)                     | Knowledge-graph-augmented RAG with pgvector seeding and recursive-CTE graph traversal                                                                         |
| [Branching CI/CD](/docs/examples/branching-cicd/)        | Validate schema changes on an isolated Lakebase branch in GitHub CI/CD, with an AI impact report per PR                                                       |
| [Consort](/docs/examples/consort/)                       | Spec-first, test-driven agentic development where every green is a real test on a live Lakebase branch, held by a deterministic state machine and human gates |
| [SCM Extension](/docs/examples/lakebase-scm-extension/)  | A VS Code / Cursor extension that pairs each code branch with a Lakebase database branch                                                                      |
| [SCM Utils](/docs/examples/lakebase-scm-utils/)          | The portable engine behind the extension and Consort: branching, the paired-branch SCM state machine, credentials, and migrations                             |
| [AI Memory](/docs/examples/ai-memory/)                   | Short-term, long-term, and semantic (pgvector) agent memory on one Lakebase store                                                                             |
| [FastAPI App](/docs/examples/lakebase-fastapi/)          | A Databricks App serving a Lakebase synced table through a FastAPI REST API                                                                                   |
| [Genie Caching](/docs/examples/lakebase-genie-caching/)  | A pgvector semantic cache in front of Genie Spaces, so reworded questions reuse SQL instead of regenerating it                                                |
| [Feature Store](/docs/examples/feature-store/)           | Feature store integration                                                                                                                                     |
| [Reverse ETL](/docs/examples/reverse-etl/)               | Reverse ETL patterns                                                                                                                                          |
| [Zerobus Ingestion](/docs/examples/zerobus-to-lakebase/) | Land data from event sources via Zerobus and sync to Lakebase                                                                                                 |
| [Kafka Ingestion](/docs/examples/kafka-rt-to-lakebase/)  | Ingest directly from Kafka w/real-time mode                                                                                                                   |   

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

Guidelines:

- Resources related to your example must be deployable via Databricks Asset Bundles
- Keep database configurations small to reduce spend
- Use `uv` as the package manager
- Use `ruff` as the linter
