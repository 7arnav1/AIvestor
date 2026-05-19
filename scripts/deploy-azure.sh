#!/usr/bin/env bash
# Deploy AIvestor to Azure Container Apps (builds Dockerfile in the cloud).
#
# Prerequisites:
#   az login
#   az extension add --name containerapp --upgrade
#
# Usage:
#   ./scripts/deploy-azure.sh
#   AZURE_RESOURCE_GROUP=my-rg AZURE_APP_NAME=my-app ./scripts/deploy-azure.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RG="${AZURE_RESOURCE_GROUP:-aivestor-rg}"
APP="${AZURE_APP_NAME:-aivestor-api}"
LOCATION="${AZURE_LOCATION:-eastus}"
DATA_START="${DATA_START:-2015-01-01}"

echo "Resource group: $RG"
echo "Container App:  $APP"
echo "Region:         $LOCATION"

az group create --name "$RG" --location "$LOCATION"

az containerapp up \
  --name "$APP" \
  --resource-group "$RG" \
  --location "$LOCATION" \
  --source . \
  --ingress external \
  --target-port 8000 \
  --env-vars "DATA_START=${DATA_START}"

FQDN=$(az containerapp show \
  --name "$APP" \
  --resource-group "$RG" \
  --query properties.configuration.ingress.fqdn -o tsv)

echo ""
echo "Deployed. Open: https://${FQDN}"
echo ""
echo "Note: PPO needs a trained model under data/models/. Mount Azure Files or"
echo "train locally and rebuild, or run evaluate without PPO until a model exists."
