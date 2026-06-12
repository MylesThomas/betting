# NBA OSS Repo Assessments

Assessed 2026-06-10 using `/assess-repo` skill. Rubric: 9 factors × 5pts = 45 max.
Source list: `ref/free-nba-ml-repos.md`
**v2: scores based on actual source code, not README claims.**

---

## Ranked Summary

| Rank | Repo | Score | Verdict | Key delta from README-only |
|------|------|-------|---------|---------------------------|
| 1 | [nba_api](https://github.com/swar/nba_api) | 30/45 | Mine for parts | Confirmed: live odds endpoint exists; no rate limiting in http.py |
| 2 | [NBA-Machine-Learning-Sports-Betting](https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting) | 23/45 | Mine for parts | TimeSeriesSplit confirmed; sbrscrape is fragile unmaintained scraper |
| 3 | [NBA_AI](https://github.com/NBA-Betting/NBA_AI) | 21/45 | Reference only | **Downgraded** — no bet-sizing layer; shuffle=True on sequential data; coaching/travel features hardcoded as zeros |
| 4 | [NBA-Game-Outcome-Prediction](https://github.com/Pirkn/NBA-Game-Outcome-Prediction) | 16/45 | Reference only | TimeSeriesSplit imported but never used; Selenium + manual input() calls |
| 5 | [nba-prediction](https://github.com/cmunch1/nba-prediction) | 16/45 | Reference only | **Downgraded** — requires Hopsworks/Neptune/ScrapingAnt credentials; no odds code anywhere |
| 6 | [basketball-analysis](https://github.com/zostaff/basketball-analysis) | 12/45 | Skip | Model weights not in repo; homography math is real but never betting-relevant |
| 7 | [NBA-Prediction-Modeling](https://github.com/luke-lite/NBA-Prediction-Modeling) | 12/45 | Reference only | Random hold-out split; likely leakage from rolling averages computed pre-split |
| 8 | [basketball-analysis-system](https://github.com/CyrilleAD/basketball-analysis-system) | 12/45 | Skip | Speed calc uses hardcoded magic constant (×0.05), no court homography |
| 9 | [AI-Basketball-Shot-Detection-Tracker](https://github.com/avishah3/AI-Basketball-Shot-Detection-Tracker) | 8/45 | Skip | **Downgraded** — hardcoded local video path; cv2.imshow couples to desktop display; zero odds logic |

**Key takeaway from code review:** The README pass was materially too generous on 3 repos. `NBA_AI` looked sophisticated on paper but has no bet-sizing layer and uses shuffle=True on time-series data. `cmunch1` requires 4 paid/external service credentials to run. `avishah3` is a pure hobbyist demo. Only `nba_api` and `kyleskom` hold up under code scrutiny — and both are for game-level markets, not props.

---

## Individual Assessments (code-verified)

---

### [nba_api](https://github.com/swar/nba_api)

**One-liner:** A Python client that wraps every publicly accessible NBA.com stats and live-data endpoint into typed classes with pandas DataFrame output.

**Stars:** 3.7k | **Last commit:** 2026-02 | **Language:** Python 99.4%

| Factor | Score | Evidence |
|--------|-------|----------|
| Alpha potential | 3/5 | `playerdashptreb.py` (contested/uncontested, rebound distance splits), `playerdashptshots.py` (shot-clock range, defender distance, touch-time), `hustlestatsboxscore.py` (box-outs, deflections, screen assists) — prop-relevant endpoints exist. No edge framing built in. |
| Novel insight | 4/5 | `boxscoreplayertrackv3.py` exposes speed, distance traveled, contested shots. `leaguedashptstats.py` covers passing, pull-up shooting, defense at rim. Well beyond box scores. |
| Model sophistication | 2/5 | Data-access library only. Clean `get_data_frames()` API. No retry/rate-limit logic in `library/http.py` — callers must handle 429s themselves. |
| Odds integration | 2/5 | `live/nba/endpoints/odds.py` exposes opening/current spread and moneyline per book. Game-level only, today's games only, no player props odds, no historical lines. CLV approximation possible but requires custom code. |
| Data pipeline | 3/5 | `leaguegamefinder.py` accepts `date_from_nullable`/`date_to_nullable` + 80 filter params. No automated scheduling, no S3, no schema versioning. Debug cache via MD5-hashed files only. |
| Backtesting rigor | 2/5 | Historical logs back to 1946. No walk-forward tooling, no ROI calc, no vig handling. Data depth is good; rigor infrastructure absent. |
| Stack adaptability | 5/5 | `pip install nba_api`, stateless class instantiation, `get_request=False` option. Lambda-friendly. No DB or Docker required. |
| Maintenance burden | 4/5 | v1.11.4 released 2026-02. CircleCI + GitHub Actions. `pyproject.toml` + `poetry.lock`. 3.7k stars, 718 forks. `CHANGELOG.MD` present. |
| Out-of-box / free impact | 5/5 | `pip install nba_api` → one import → DataFrames. Zero auth, zero cost. Tracking data, hustle stats, live odds all immediately accessible. |
| **Total** | **30/45** | |

**Verdict:** Mine for parts

**Best component to steal:** `playerdashptreb.py` + `playerdashptshots.py` + `hustlestatsboxscore.py` trio — contested/uncontested splits, shot-clock pressure, box-outs, defender distance — exact secondary features that price rebounds and assist props better than box scores alone.

**Gaps / risks:** No player-prop odds data; no rate-limit handling in `http.py` (will 429 in production without a throttle wrapper); historical lines entirely absent.

---

### [NBA-Machine-Learning-Sports-Betting](https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting)

**One-liner:** Trains XGBoost and neural network models on team box-score averages and sportsbook lines to predict game moneylines and totals, then outputs EV and Kelly Criterion stake sizes from the command line.

**Stars:** 1.7k | **Last commit:** 2026-01 | **Language:** Python 74.7%

| Factor | Score | Evidence |
|--------|-------|----------|
| Alpha potential | 2/5 | Moneylines and totals only — zero player props. `Expected_Value.py` computes EV as a one-shot snapshot; no CLV tracking. |
| Novel insight | 1/5 | `Create_Games.py` builds features from season-average box scores in `TeamData.sqlite` + `Days_Rest_Home/Away`. No tracking, injury load, court factors. Pure public data recycling. |
| Model sophistication | 3/5 | `XGBoost_Model_ML.py` uses `TimeSeriesSplit` with 100 random-search trials over eta, depth, lambda, alpha, colsample. Optional probability calibration. NN is separate with no ensembling at inference. |
| Odds integration | 3/5 | `SbrOddsProvider.py` pulls live lines from 7 books via `sbrscrape`. `Kelly_Criterion.py` implements full Kelly. EV calculated. Missing CLV, no vig decomposition, no line-movement history. |
| Data pipeline | 2/5 | `Get_Data.py` auto-ingests daily stats to SQLite with backfill. `sbrscrape` is a fragile unmaintained PyPI scraper — single point of failure for live odds. No historical line storage. |
| Backtesting rigor | 2/5 | `TimeSeriesSplit` in training is commendable. But no walk-forward P&L simulation, no net-of-vig ROI curve, no honest sample size caveat anywhere in the code. |
| Stack adaptability | 3/5 | Clean module structure (`src/Predict`, `src/Train-Models`, `src/Utils`). `Kelly_Criterion.py` and `XGBoost_Runner.py` are importable standalone. SQLite needs replacement for Lambda/S3. |
| Maintenance burden | 4/5 | Jan 2026 commit, `config.toml` for season config, readable code. `# TODO: Add tests` throughout `Get_Data.py`. `sbrscrape` dependency is unmaintained. |
| Out-of-box / free impact | 3/5 | `python3 main.py -xgb -odds=fanduel` is genuinely one command. Pre-trained models ship in `Models/`. Risk: `sbrscrape` breaks periodically — no guarantee odds fetch works on a given day. |
| **Total** | **23/45** | |

**Verdict:** Mine for parts

**Best component to steal:** `XGBoost_Model_ML.py` training loop — `TimeSeriesSplit` walk-forward CV with 100-trial random search, early stopping, and optional probability calibration — directly portable as training backbone for any game-level or prop-level model.

**Gaps / risks:** Feature set is season-average box scores only — no signal on anything books don't already price; `sbrscrape` odds dependency is the single point of failure and is unsupported.

---

### [NBA_AI](https://github.com/NBA-Betting/NBA_AI)

**One-liner:** A hierarchical prediction system (L1 player embeddings → L2 synergy → L3 team → L4 game) that generates spread/total forecasts but has no bet-sizing, CLV, or profitability layer.

**Stars:** 112 | **Last commit:** 2026-04 | **Language:** Python 98.2%

| Factor | Score | Evidence |
|--------|-------|----------|
| Alpha potential | 2/5 | Game spreads and totals only (`pred_spread`, `pred_total` in `phase5_predictor.py`). No player props. Predictions never compared to market for value extraction. |
| Novel insight | 3/5 | `team_features.py` includes schedule density (`games_in_last_7`), back-to-back flags, rest days, arena altitude, eFG/turnover/ORB rolling windows. But coaching and travel/timezone fields are hardcoded as zeros — not yet populated. |
| Model sophistication | 3/5 | L1→L2(`PlayerSynergyNetwork`)→L3(`TeamModel`)→L4(`GamePredictor`) + 42M-param transformer. But `train_phase5_b.py` uses `shuffle=True` on sequential data — explicit time-series leakage. No walk-forward CV. |
| Odds integration | 2/5 | `betting.py` collects ESPN opening/current/closing lines + Covers.com scraping with UPSERT storage. Lines stored but never consumed by any predictor. No CLV, no Kelly, no EV anywhere. |
| Data pipeline | 3/5 | `orchestrator.py` runs at 1:30am + 4pm daily. SQLite backend with `Betting` table and `lines_finalized` flag. Historical lines stored. Not serverless-native but structured for porting. |
| Backtesting rigor | 1/5 | `train_legacy_models.py` reports only MAE and win accuracy. Zero ROI, Sharpe, drawdown, or bet-level P&L. No walk-forward CV in any training script. |
| Stack adaptability | 3/5 | Pure Python, modular `src/pipeline/`, `src/predictions/`. SQLite needs replacing for Lambda. PyTorch `.pt` + `.joblib` files load cleanly. |
| Maintenance burden | 3/5 | Config-driven, logging, health checks, dry-run mode present. Multiple partially-implemented features signal active but incomplete development. Committed through April 2026. |
| Out-of-box / free impact | 1/5 | Requires pre-built model checkpoints (not in repo), populated SQLite DB, manual orchestrator wiring. No inference path without significant setup. Output has no decision layer. |
| **Total** | **21/45** | |

**Verdict:** Reference only

**Best component to steal:** `betting.py` 3-tier odds collection pipeline (ESPN opening/closing + Covers.com backfill) — production-quality historical lines ingestion module portable to your own system.

**Gaps / risks:** The value-capture layer is entirely absent — predicts spreads but has no mechanism to compare to market, calculate edge, or size bets; `shuffle=True` on sequential training data is a data leakage red flag.

---

### [NBA-Game-Outcome-Prediction](https://github.com/Pirkn/NBA-Game-Outcome-Prediction)

**One-liner:** Undergraduate moneyline classifier using Selenium scraping, MOV-adjusted Elo, rolling averages, and Ridge/XGBoost/NN ensemble — no odds data anywhere.

**Stars:** 9 | **Last commit:** 2024 | **Language:** Jupyter Notebook 100%

| Factor | Score | Evidence |
|--------|-------|----------|
| Alpha potential | 1/5 | Binary `target = won.shift(-1)` only. No props, no totals, no spread. Word "odds" does not appear in any notebook source cell. |
| Novel insight | 2/5 | MOV-scaled Elo K-factor (`k = (20*(mov+3)**0.8)/(7.5+0.006*elo_diff)`) and season-regression blend are mildly interesting. Everything else is vanilla basketball-reference box scores. |
| Model sophistication | 3/5 | Optuna HPO (300 trials each), `TimeSeriesSplit` imported, season-walk-forward backtest loop, Keras NN with BatchNorm/Dropout. But `TimeSeriesSplit` is imported and never actually used in the backtest — manual season loop is used instead. |
| Odds integration | 1/5 | Zero. No sportsbook lines, CLV, vig, or Kelly anywhere. |
| Data pipeline | 2/5 | Selenium + `time.sleep(5)` basketball-reference scraping. `input()` prompts throughout. Hardcoded local paths. No automation. |
| Backtesting rigor | 2/5 | Walk-forward by season is structurally sound. Accuracy reported (65–67%) but no ROI, no confidence intervals. Target filled with `2` for future games creates a three-class target risk. |
| Stack adaptability | 2/5 | Pure Python/pandas/sklearn, no external API deps. But Selenium + `input()` calls + hardcoded paths make Lambda non-trivial. |
| Maintenance burden | 2/5 | `requirements.txt` present. `game_season = 2024` hardcoded. Basketball-reference Selenium scraper is fragile. |
| Out-of-box / free impact | 1/5 | Requires Chrome + hours of scraping + manual `input()` per prediction. Output has no line comparison. |
| **Total** | **16/45** | |

**Verdict:** Reference only

**Best component to steal:** The MOV-adjusted Elo implementation in `elo_handler()` — K-factor formula with season-regression blend is clean and could slot into a team-strength feature for a spread or total model.

**Gaps / risks:** No odds data at any stage; Selenium + `input()` pipeline is entirely unsuitable for automation.

---

### [nba-prediction](https://github.com/cmunch1/nba-prediction)

**One-liner:** Automated daily NBA win-probability pipeline on XGBoost/LightGBM with GitHub Actions — but requires 4 paid/external service credentials to run and has zero betting-market logic.

**Stars:** 307 | **Last commit:** 2026-04 | **Language:** Jupyter Notebook 98.2%, Python 1.8%

| Factor | Score | Evidence |
|--------|-------|----------|
| Alpha potential | 1/5 | Predicts only `HOME_TEAM_WINS` binary (`constants.py`, `data_processing.py::add_TARGET()`). No props, no totals. Best model AUC ~0.64 on moneylines — least mispriced market. |
| Novel insight | 1/5 | `feature_engineering.py` — pure NBA.com box-score rolling averages (3/7/10 game windows). No tracking, injury, rest, or travel data. |
| Model sophistication | 3/5 | Optuna 150-trial tuning (`optuna_objectives.py`), `TimeSeriesSplit` available, SHAP in `07_model_testing.ipynb`, explicit data leakage prevention. No ensemble stacking. |
| Odds integration | 1/5 | Zero odds code anywhere in `src/` or notebooks. |
| Data pipeline | 2/5 | GitHub Actions cron at 8am UTC scrapes NBA.com via Selenium/ScrapingAnt. CSVs committed to repo. No historical lines, no odds DB. |
| Backtesting rigor | 1/5 | Static train/test split only (`data_processing.py::split_train_test()`). No walk-forward, no net ROI, no sample size discussion. |
| Stack adaptability | 3/5 | Pure Python, modular `src/` layout. GitHub Actions pattern is extractable. Hopsworks feature store is a coupling risk. |
| Maintenance burden | 3/5 | Clean, well-organized modules. Last commit April 2026. Selenium scraper is fragile. Hopsworks/Neptune/Google Drive secrets add operational overhead. |
| Out-of-box / free impact | 1/5 | Requires Hopsworks, Neptune.ai, ScrapingAnt, and Google Drive credentials in `.github/workflows/`. Cannot run without 4 external service accounts. Produces win probability with no bet-sizing logic. |
| **Total** | **16/45** | |

**Verdict:** Reference only

**Best component to steal:** `optuna_objectives.py` Optuna tuning loop with `TimeSeriesSplit` — clean, directly transplantable hyperparameter search scaffold for a props pipeline.

**Gaps / risks:** 4 paid/external service dependencies make this impossible to run out-of-the-box; zero betting-market logic means it's a win-probability toy with no decision layer.

---

### [basketball-analysis](https://github.com/zostaff/basketball-analysis)

**One-liner:** YOLO + ByteTrack CV pipeline that annotates broadcast video with player positions, team assignments, and a tactical overhead map — zero connection to betting or statistical modeling.

**Stars:** 14 | **Last commit:** 2025 | **Language:** Jupyter Notebook 69%, Python 31%

| Factor | Score | Evidence |
|--------|-------|----------|
| Alpha potential | 1/5 | `main.py` outputs annotated video files. No props, no lines, no betting framing anywhere. |
| Novel insight | 2/5 | `tactical_view_converter.py` does real homography-based court mapping with NBA court dimensions (28m×15m) and derives pixel-to-meter positions and speeds. Mildly novel raw signal, never converted to a betting-relevant feature. |
| Model sophistication | 1/5 | Pure inference wrappers over pretrained YOLO + ByteTrack. `team_assigner.py` uses FashionCLIP zero-shot. No training, no regression, no CV. |
| Odds integration | 1/5 | Zero. No odds import, no line logic, no Kelly anywhere. |
| Data pipeline | 1/5 | Input: local video file via CLI. Output: annotated `.mp4`. Stub/pickle caching for frames exists but no historical storage, no S3. |
| Backtesting rigor | 1/5 | None. `training_notebooks/` contains YOLO fine-tuning notebooks only. |
| Stack adaptability | 2/5 | Python + Docker. `requirements.txt` has conflicting opencv pins (`4.9.0.80` and `4.8.0.74`). Heavy GPU dependency; not Lambda-friendly. |
| Maintenance burden | 2/5 | 14 stars, 1 commit, no tests, `sys.path.append('../')` hacks in every module, conflicting deps. |
| Out-of-box / free impact | 1/5 | Custom YOLO weights not in repo. GPU required. Produces zero betting-relevant output. |
| **Total** | **12/45** | |

**Verdict:** Skip

**Best component to steal:** `tactical_view_converter/homography.py` court-to-meter coordinate math — useful only if building a spatial features layer from broadcast video (a multi-month build).

**Gaps / risks:** Model weights not shipped; conflicting dependency pins; adapting toward betting alpha would require building essentially everything from scratch.

---

### [NBA-Prediction-Modeling](https://github.com/luke-lite/NBA-Prediction-Modeling)

**One-liner:** Academic comparison of 7 models on 11,979 games concluding ELO beats all ML approaches — but uses a random hold-out split with likely data leakage.

**Stars:** 51 | **Last commit:** 2024-08 | **Language:** Jupyter Notebook 99.9%

| Factor | Score | Evidence |
|--------|-------|----------|
| Alpha potential | 1/5 | Game win/loss only. No betting markets, no EV. |
| Novel insight | 1/5 | Four factors (eFG%, TOV%, ORB%, FTr) from public box scores. No tracking, schedule density, or injury load. ELO seeds from a FiveThirtyEight CSV. |
| Model sophistication | 2/5 | LR, RF, KNN, GNB, SVC, Keras MLP, GridSearchCV. But train/test is a single random hold-out (seed=99). Rolling averages computed on full dataset before splitting — strong leakage risk. |
| Odds integration | 1/5 | Zero. No lines, no vig, no Kelly. |
| Data pipeline | 1/5 | Static CSVs committed to repo. `elo_calculator.py` reads a hardcoded GitHub raw URL. No automation. |
| Backtesting rigor | 1/5 | `elo-testing.ipynb` bins accuracy by win-probability bucket only. No ROI, no walk-forward, no bet sizing simulation. `tester.ipynb` is mostly empty scaffolding. |
| Stack adaptability | 2/5 | `ELOModelBuilder` class in `model_builder.py` is self-contained and importable. No Lambda entry points, no config injection. |
| Maintenance burden | 2/5 | `nbaenv.yml` pins conda env. Row-by-row `.loc` loop in `create_new_model()` is O(n) slow. Aug 2024, no tests, no CI. |
| Out-of-box / free impact | 1/5 | No line ingestion, no edge threshold, no Kelly. Running gives a win probability with no context for whether it beats closing line. |
| **Total** | **12/45** | |

**Verdict:** Reference only

**Best component to steal:** `ELOModelBuilder.calc_K()` + `new_season_elo_adj()` in `model_builder.py` — MOV-adjusted K-factor with season-regression is clean and correct for use as a team-strength prior.

**Gaps / risks:** Random hold-out split means all accuracy figures are likely inflated by leakage; abandoned August 2024 with no path to current data.

---

### [basketball-analysis-system](https://github.com/CyrilleAD/basketball-analysis-system)

**One-liner:** Modular YOLO + ByteTracker CV pipeline for basketball video annotation with player tracking, team assignment, and speed metrics — no predictive modeling or betting logic.

**Stars:** 8 | **Last commit:** 2025-07 | **Language:** Python 100%

| Factor | Score | Evidence |
|--------|-------|----------|
| Alpha potential | 1/5 | `main.py` produces annotated video. Zero betting signals anywhere in codebase. |
| Novel insight | 2/5 | `speed_and_distance_calculator.py` computes per-player speed/km-h and cumulative distance from pixel tracking. But pixel-to-meter conversion uses a hardcoded magic constant (`* 0.05`) with no court homography — accuracy is unknown. |
| Model sophistication | 1/5 | `team_assigner.py` uses k-means (k=2) on jersey colors. YOLO + ByteTracker for detection. No regression, no ensemble, no CV. |
| Odds integration | 1/5 | Entirely absent from every file. |
| Data pipeline | 1/5 | Local video file via `argparse`. Stub caching as `.pkl`. No historical ingestion, no S3. |
| Backtesting rigor | 1/5 | None. |
| Stack adaptability | 2/5 | Clean Python module structure, modular `__init__.py` pattern. But requires YOLO weights + video input — not Lambda-friendly. |
| Maintenance burden | 2/5 | Minimal, concrete `requirements.txt`. Hardcoded pixel calibration constant, no tests. |
| Out-of-box / free impact | 1/5 | YOLO model weights not included. Produces only annotated video — zero betting value today. |
| **Total** | **12/45** | |

**Verdict:** Skip

**Best component to steal:** `team_assigner.py` k-means jersey-color classification — reusable for any broadcast-to-data pipeline, but adds nothing to a box-score or API-based betting stack.

**Gaps / risks:** Speed metric uses magic constant with no calibration; nearly identical scope to `zostaff/basketball-analysis` with even fewer stars.

---

### [AI-Basketball-Shot-Detection-Tracker](https://github.com/avishah3/AI-Basketball-Shot-Detection-Tracker)

**One-liner:** YOLOv8 + OpenCV hobbyist tool that counts makes/misses from a hardcoded local video file — zero connection to betting markets or any predictive modeling.

**Stars:** 258 | **Last commit:** 2024 | **Language:** Python 100%

| Factor | Score | Evidence |
|--------|-------|----------|
| Alpha potential | 1/5 | `shot_detector.py` outputs only `makes / attempts` counters on video frame. No market, no prop, no line framing. |
| Novel insight | 1/5 | `utils.py` detects shot success via linear interpolation through hoop bounding box. No player identity, no game context. |
| Model sophistication | 1/5 | `main.py` runs `model.train(data='config.yaml', epochs=100, imgsz=640)` — single YOLOv8n, no ensembling, no CV, no hyperparameter search. |
| Odds integration | 0/5 | Zero references to odds, lines, CLV, Kelly, or vig across all source files. |
| Data pipeline | 0/5 | `self.cap = cv2.VideoCapture("video_test_5.mp4")` — hardcoded local video. No ingestion, no storage. |
| Backtesting rigor | 0/5 | None. |
| Stack adaptability | 2/5 | Pure Python, modular class structure. But `cv2.imshow` / `cv2.waitKey` hard-couples to a desktop display loop — not Lambda/headless friendly. |
| Maintenance burden | 2/5 | Ships `best.pt`. `opencv-python==4.5.4.60` and `ultralytics==8.0.26` are stale pins. `cvzone==1.5.6` extra dependency. |
| Out-of-box / free impact | 1/5 | Runs without training (ships `best.pt`), but output is a live video window with make/miss counts — zero betting signal. |
| **Total** | **8/45** | |

**Verdict:** Skip

**Best component to steal:** `score()` + `clean_ball_pos()` trajectory logic in `utils.py` — could theoretically measure shot-arc quality from broadcast video, but requires building ~95% of a pipeline on top.

**Gaps / risks:** Hardcoded local video path, desktop-display coupling, stale deps, zero betting logic — repurposing for alpha generation would require rebuilding from scratch.
