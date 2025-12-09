# 🚀 AWS Automation Setup - TL;DR

## What You're Building

```
┌─────────────────────────────────────────────────────────────┐
│                    YOUR WORKFLOW                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  😴 Sleep peacefully                                        │
│       ↓                                                     │
│  ⏰ 6:00 AM CT / 7:00 AM ET                                │
│       ↓                                                     │
│  ☁️  AWS Lambda runs automatically                          │
│       ↓                                                     │
│  📊 Fetches latest NBA props                                │
│       ↓                                                     │
│  🔍 Finds arbitrage opportunities                           │
│       ↓                                                     │
│  📤 Pushes to GitHub                                        │
│       ↓                                                     │
│  🌐 Streamlit Cloud auto-deploys                            │
│       ↓                                                     │
│  ☕ Wake up, check dashboard over coffee                    │
│       ↓                                                     │
│  💰 Place bets (optional!)                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 📋 Checklist (Simplified)

### Phase 1: Preparation (10 min)
- [ ] Get GitHub Personal Access Token
- [ ] Get Odds API Key
- [ ] Create AWS account (if needed)

### Phase 2: AWS Setup (60 min)
- [ ] Store secrets in AWS Secrets Manager
- [ ] Create Lambda function
- [ ] Upload Lambda code from `docs/lambda_function.py`
- [ ] Configure Lambda (15 min timeout, 512 MB memory)
- [ ] Add environment variables
- [ ] Grant Secrets Manager permissions

### Phase 3: Dependencies (30 min)
- [ ] Create Lambda layer with Python packages
- [ ] Attach layer to function

### Phase 4: Testing (15 min)
- [ ] Test Lambda function manually
- [ ] Verify GitHub commit
- [ ] Verify Streamlit dashboard updates

### Phase 5: Scheduling (15 min)
- [ ] Create EventBridge rule
- [ ] Set cron: `0 12 * * ? *` (7 AM EST)
- [ ] Verify schedule

### Phase 6: Monitoring (15 min)
- [ ] Set up CloudWatch alerts (optional)
- [ ] Configure SNS email notifications (optional)

## 💰 Cost

**~$0.40/month** (mostly Secrets Manager)

Lambda is FREE (well within free tier limits)

## 📚 Full Documentation

1. **Step-by-step guide:** `docs/AWS_AUTOMATION_CHECKLIST.md`
2. **Quick reference:** `docs/AWS_QUICK_REFERENCE.md`
3. **Lambda code:** `docs/lambda_function.py`
4. **Dependencies:** `docs/lambda_requirements.txt`

## 🆘 Need Help?

### Common Issues

**Lambda times out:**
→ Increase timeout to 15 minutes

**Out of memory:**
→ Increase memory to 1024 MB

**Can't access secrets:**
→ Attach `SecretsManagerReadWrite` policy to Lambda role

**Git push fails:**
→ Check GitHub token is valid

### Support Resources

- AWS Lambda Docs: https://docs.aws.amazon.com/lambda/
- Streamlit Community: https://discuss.streamlit.io/
- GitHub Issues: https://github.com/MylesThomas/betting/issues

## ✅ Success Criteria

**You're done when:**
- Wake up at 6 AM CT
- Open https://tqs-props-dashboard.streamlit.app
- See today's fresh data
- You didn't run any commands
- Profit! 💰

---

**Total Setup Time:** 2-3 hours  
**Ongoing Effort:** 0 minutes/day  
**ROI:** Infinite (zero effort for daily updates!) ♾️

