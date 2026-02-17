# Security & Secrets Management

**Status:** 📝 Planned  
**Last updated:** 2026-02-13

This document will cover:

## API Key Management

- Never commit API keys to git
- Use environment variables in Lambda
- Local development with `.env` files
- Rotation procedures

## AWS IAM Policies

- Lambda execution roles
- S3 bucket policies
- Principle of least privilege

## Secrets in Config

Currently API keys are in `config/config.yaml` which is `.gitignore`d.

**Best practice:**
- Development: Use `.env` file or export env vars
- Lambda: Use AWS Systems Manager Parameter Store or Secrets Manager
- Never hardcode secrets in code

---

**To be written in:** Later phase as needed
