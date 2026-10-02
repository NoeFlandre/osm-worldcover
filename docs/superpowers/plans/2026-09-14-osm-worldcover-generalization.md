# osm-worldcover Generalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the project to `osm-worldcover`. Make one reproducible pipeline produce the existing Wikidata WorldCover dataset, a description-tag dataset and a website-tag dataset.

**Architecture:** Source recipes normalize each repository into the existing canonical region tables. The pure domain logic and the WorldCover processing stay shared. The adapters and the configuration keep the source-specific I/O, the geometry decoding, the text extraction, the card metadata and the HF publication settings.

**Tech Stack:** uv, Python 3.12, pandas/GeoPandas/Shapely/PyArrow, Typer, pytest/pytest-bdd/Hypothesis, Ruff, ty, import-linter, CRAP, mutmut, Docker, MkDocs, Hugging Face Hub.

---

### Task 1: Establish source contracts with tests

**Files:**
- Create: `tests/unit/test_sources.py`
- Modify: `tests/unit/test_hub.py`
- Modify: `tests/unit/test_config.py`

- [x] Add failing tests for the three source recipes, the source-specific region paths, the description WKB normalization, the localized descriptions, and the website/contact documents.
- [x] Run the focused tests. Confirm that they fail because the generic source contract does not exist.

### Task 2: Implement the generic source registry and adapters

**Files:**
- Create: `src/osm_worldcover/sources.py`
- Modify: `src/osm_worldcover/config.py`
- Modify: `src/osm_worldcover/adapters/source.py`
- Modify: `src/osm_worldcover/adapters/hub.py`
- Modify: `src/osm_worldcover/build.py`
- Modify: `src/osm_worldcover/pipeline.py`

- [x] Add named recipes with source dataset IDs and derived output IDs.
- [x] Normalize each source into canonical polygon/link/document tables.
- [x] Make the region discovery, the download, the cleanup and the build orchestration recipe-aware.
- [x] Run the focused tests. Confirm that they pass.

### Task 3: Generalize cards and publication metadata

**Files:**
- Modify: `src/osm_worldcover/domain/card.py`
- Modify: `src/osm_worldcover/adapters/publish.py`
- Modify: `src/osm_worldcover/domain/manifest.py`
- Modify: `tests/unit/test_card.py`
- Modify: `tests/unit/test_publish.py`

- [x] Generate the source-specific titles, load instructions, provenance links, tags and licensing caveats. Keep the map and the deterministic named examples.
- [x] Test all three card recipes and the generic publication commit message.

### Task 4: Rename the repository/package and user-facing surfaces

**Files:**
- Rename: `src/osm_worldcover/` to `src/osm_worldcover/`
- Modify: `pyproject.toml`, `uv.lock`, `.importlinter`, `README.md`, `docs/`, `Dockerfile`, `.github/workflows/ci.yml`, and all tests/imports.

- [x] Rename the package, the distribution, the CLI, the Docker image, the docs, the badges and the GitHub links to `osm-worldcover`. Keep the historical Wikidata HF dataset ID only where it is necessary.
- [x] Regenerate the lockfile. Run the packaging and import smoke tests.

### Task 5: Full verification and publication

- [ ] Run these checks in order: baseline, Ruff, ty, unit/property/acceptance tests, architecture, CRAP, mutation, Docker smoke, diff review.
- [ ] Build representative deterministic releases for all three recipes. Inspect the cards, the maps and the Parquet schemas.
- [ ] Rename the GitHub repository. Update the remote.
- [ ] Create and publish the two new public HF dataset repositories.
- [ ] Verify the Dataset Viewer config, split, rows and schema of each repository. Verify that the existing Wikidata HF dataset has no change.
