# Tech Debt Tracker

**Last updated:** 2026-02-13

This document tracks known issues, technical debt, and improvement opportunities.

## High Priority

### Lambda Functions Scattered Across Directories

**Issue:** Lambda functions are in `scripts/`, `docs/`, and `lambda/`  
**Impact:** Hard to find, inconsistent structure, confusing  
**Target:** All lambda functions in `lambda/<name>/lambda_function.py`  
**Plan:** See IMPLEMENTATION_PLAN.md Phase 0.4  
**Status:** 📝 Planned

### No Architectural Boundary Tests

**Issue:** Import rules not enforced mechanically  
**Impact:** Easy to create tangled dependencies  
**Target:** `tests/test_architecture.py` with boundary checks  
**Plan:** See IMPLEMENTATION_PLAN.md Phase 3.3  
**Status:** 📝 Planned

### Unstructured Logging

**Issue:** Mix of print statements and logging, no consistent format  
**Impact:** Hard to debug, can't query logs  
**Target:** Structured JSON logging throughout  
**Plan:** See IMPLEMENTATION_PLAN.md Phase 4.1  
**Status:** 📝 Planned

---

## Medium Priority

### No Data Validation Harnesses

**Issue:** No automated way to check props data quality  
**Impact:** Bad data can slip through unnoticed  
**Target:** `scripts/validate_*.py` harnesses  
**Plan:** See IMPLEMENTATION_PLAN.md Phase 4.2  
**Status:** 📝 Planned

### Scattered Documentation

**Issue:** README files, docs/, books/, tmp/ all have overlapping content  
**Impact:** Hard to find information, docs get stale  
**Target:** Consolidated docs/ structure  
**Plan:** See IMPLEMENTATION_PLAN.md Phase 1.3  
**Status:** 🚧 In Progress

### Hardcoded Thresholds

**Issue:** Magic numbers scattered in analysis code  
**Impact:** Hard to tune, can't track why value was chosen  
**Target:** All thresholds in config YAML files  
**Status:** 📝 Planned

---

## Low Priority

### Test Coverage Gaps

**Issue:** Many scripts have no tests  
**Impact:** Changes risk breaking things  
**Target:** 80% coverage across critical paths  
**Status:** 📝 Planned

### No Automated Dependency Graph

**Issue:** Can't visualize module relationships  
**Impact:** Hard to understand system  
**Target:** `docs/generated/dependency-graph.png`  
**Plan:** See IMPLEMENTATION_PLAN.md Phase 3.4  
**Status:** 📝 Planned

---

## Resolved (Archive)

*None yet - this section will track completed items*

---

## Template for New Items

```markdown
### [Title]

**Issue:** What's wrong?  
**Impact:** Why does it matter?  
**Target:** What's the desired state?  
**Plan:** How do we fix it?  
**Status:** 📝 Planned / 🚧 In Progress / ✅ Done
```

---

**Legend:**
- 📝 Planned - Not started
- 🚧 In Progress - Being worked on
- ✅ Done - Completed (move to Resolved section)
- ⚠️ Blocked - Can't proceed until something else is done

---

**Maintained by:** Daily quality checker + human review  
**Review cycle:** Weekly
