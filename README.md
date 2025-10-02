# Project Setup Guide

This guide will help you set up the environment, run the databases, ingest data, and start the backend and frontend applications.

---

## Prerequisites

Before you begin, make sure you have the following installed:

- Python 3.12
- pip
- Docker
- Docker Compose

Install the required Python dependencies:

```bash
pip install uv
uv pip install -r requirements.txt
```

---

## 1. Prepare the Environment Variables

Create a `.env` file inside the `src/` and the `notebook/`  folder:

```
src/
  └── .env
notebook/
  └── .env
```

Add the following variables to your `.env` file:

```env
GROQ_API_KEY=<your_api_key>
GROQ_API_URL=https://api.groq.com/openai/v1
POSTGRES_URL=postgresql://myuser:mypassword@localhost:5432/postgres  # Local PostgreSQL
```

> 🔑 Replace `<your_api_key>` and database credentials with your actual values.

---

## 2. Run the Databases

Start the databases using Docker Compose:

```bash
cd docker
docker compose up
```

This will spin up both Qdrant and PostgreSQL services.

---

## 3. Ingest the Data

1. Navigate to the **notebook** folder.
2. Run the following Jupyter notebooks in order:
   - `sql_data_ingestion.ipynb` → for relational database ingestion.
   - `train_test_to_sql.ipynb` → for vector database ingestion.  
     ⚠️ Make sure to **uncomment the `vanna train` cell** before running.

---
