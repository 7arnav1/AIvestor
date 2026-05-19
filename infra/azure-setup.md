# Azure deployment

AIvestor runs as a single Container App: FastAPI + built TypeScript UI in one image.

## Option A — local Azure CLI (fastest)

```bash
az login
az extension add --name containerapp --upgrade
chmod +x scripts/deploy-azure.sh
./scripts/deploy-azure.sh
```

First request may take a few minutes while the container downloads market data.

## Option B — GitHub Actions

1. Create an Azure service principal and add secret `AZURE_CREDENTIALS` (JSON from `az ad sp create-for-rbac`).
2. In GitHub: Actions → **Deploy to Azure** → Run workflow.

## Include a trained PPO model

The image does not ship with `ppo_portfolio.zip`. After you train locally:

```bash
# Train and cache data locally
python -m aivestor.scripts.fetch_data --tickers SPY AGG GLD --start 2015-01-01
python -m aivestor.scripts.train_rl --timesteps 50000 --data-source cache

# Build image with data + model baked in (demo only)
docker build -t aivestor:local .
docker run --rm -p 8000:8000 \
  -v "$(pwd)/data/cache:/app/data/cache" \
  -v "$(pwd)/data/models:/app/data/models" \
  aivestor:local
```

For Azure Files mount on Container Apps, attach a storage account volume to `/app/data/models` and `/app/data/cache`.

## Postgres (optional)

Use `docker compose up -d` locally, or Azure Database for PostgreSQL with:

```
DATABASE_URL=postgresql+psycopg://user:pass@host:5432/aivestor
```

Set that env var on the Container App if you persist OHLCV in the database instead of CSV cache.
