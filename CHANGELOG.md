# Changelog

## [1.0.0] - 2026-11-23

### Added
- `POST /predict` and `POST /predict/batch` (up to 500 customers)
- `GET /health` and `GET /model/info`
- Per-customer explanations (`top_risk_factors`) from logistic regression coefficients
- Optional API-key auth via the `API_KEY` environment variable
- Request logging middleware with request IDs and response times
- Business-rule validation (internet and phone fields must be consistent)
- Dockerfile and GitHub Actions CI (lint, test on Python 3.11 and 3.12, Docker smoke test)

### Fixed (found during development)
- **Risk factors had no direction.** Numeric factors showed as just `tenure`, which could mean high or low tenure. They now read `tenure is low` / `TotalCharges is high`.
- **Low-risk customers got "risk factors" too.** A customer at 0.6% risk was listed with three factors, which was misleading. Factors now need a meaningful contribution (≥ 0.25 log-odds) to be reported.
- **Possible silent mismatch in batch results.** `zip()` over customers and predictions would quietly drop rows if the lists ever differed in length; now uses `zip(..., strict=True)` so it fails loudly instead.
- **Inflated probabilities (carried over from the analysis project).** Training with `class_weight="balanced"` pushed probabilities up and overstated revenue at risk. The model is now trained unweighted and uses a tuned decision threshold (0.309) instead.
