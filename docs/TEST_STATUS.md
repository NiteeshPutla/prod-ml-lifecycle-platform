# Test status at initial publication

Local validation on 2026-11-08, Python 3.9.6:

- 14 tests passed.
- 20 tests failed.
- Command: `python -m pytest -q --tb=short`.

This is a work-in-progress project. Current failures include missing convenience methods in the online store, feature pipeline, guardrail monitor and simulator; test/API mismatches for feature enums, lineage nodes, model registry and sequential testers; an embedding aggregation error; and an invalid test configuration constructor. The full failing suite is preserved to make the remaining work visible. These failures predate repository publication.

Runtime databases, trained models, MLflow runs, environments and local credentials are excluded from Git. Recreate runtime state locally through the project workflows.
