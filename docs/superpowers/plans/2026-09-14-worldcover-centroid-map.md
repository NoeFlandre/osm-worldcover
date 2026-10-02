# WorldCover centroid coverage map Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a deterministic centroid map of all ESA-labelled polygons in the release. Keep the Hugging Face Dataset Viewer healthy for all three splits.

**Architecture:** Add a small adapter. The adapter aggregates the exact release Parquets with DuckDB. It validates one stable ESA label and one finite centroid for each polygon. It renders a headless Matplotlib PNG over a Natural Earth land outline. The finalization manifest also stores one deterministic named polygon for each represented ESA class for the examples table of the card. Generate that asset inside `publish_dataset`. Render the card sections that link to it and show the examples. Upload the unchanged tabular files and the PNG. Verify the public Dataset Viewer API after the upload.

**Tech Stack:** Python 3.12, DuckDB, PyArrow/Parquet, GeoPandas, Matplotlib Agg, pytest, Ruff, `ty`, Hugging Face Hub CLI/API, Dataset Viewer HTTP API.

---

## File map

- Create `src/osm_worldcover/adapters/coverage_map.py`: aggregation of the release-Parquet centroids, validation, Natural Earth loading and PNG rendering.
- Modify `src/osm_worldcover/domain/card.py`: add the static map section. Explain that the points are centroids and not polygon outlines.
- Modify `src/osm_worldcover/domain/card.py`: add the static map section and the table of representative polygon/class examples.
- Modify `src/osm_worldcover/domain/manifest.py` and `src/osm_worldcover/finalize.py`: record the deterministic named examples from the exact kept rows.
- Modify `src/osm_worldcover/adapters/publish.py`: generate `worldcover_centroids.png` before the README is regenerated and before the build folder is uploaded.
- Modify `src/osm_worldcover/adapters/writer.py`: write Viewer-safe Parquet row groups with page indexes for future builds.
- Modify `pyproject.toml` and `uv.lock`: add Matplotlib as the runtime renderer dependency.
- Create `tests/unit/test_coverage_map.py`: tests for local Parquet aggregation, validation, deterministic palette and isolated PNG rendering.
- Modify `tests/unit/test_card.py`: assert the map link and the centroid wording.
- Modify `tests/unit/test_card.py`, `tests/unit/test_manifest.py` and `tests/unit/test_finalize_streaming.py`: assert that the example names and classes go into the card manifest.
- Modify `tests/unit/test_publish.py`: assert that map generation is part of publication. Do not make the unit tests download Natural Earth.
- Modify `tests/unit/test_writer.py`: assert that the writer splits large input batches into bounded row groups.
- Create `docs/superpowers/specs/2026-09-14-worldcover-centroid-map-design.md`: the approved design record (already committed).
- Create `docs/superpowers/plans/2026-09-14-worldcover-centroid-map.md`: this implementation plan.

## Task 1: Add the failing map and card tests

**Files:**
- Create: `tests/unit/test_coverage_map.py`
- Modify: `tests/unit/test_card.py`

- [ ] **Step 1: Add small Parquet fixtures and aggregation assertions.**

Use a helper that writes `train.parquet`, `validation.parquet` and `test.parquet` with the five map columns. Include repeated rows for one polygon and one polygon in another split. Then assert that `centroids_from_build()` returns one row for each polygon, sorted by `polygon_id`, with the expected code and label.

```python
def _write_build(root: Path, rows: dict[str, list[dict[str, object]]]) -> Path:
    root.mkdir()
    for split in ("train", "validation", "test"):
        pd.DataFrame(rows.get(split, [])).to_parquet(root / f"{split}.parquet", index=False)
    return root


def test_centroids_are_deduplicated_across_article_rows_and_splits(tmp_path: Path) -> None:
    build = _write_build(
        tmp_path / "build",
        {
            "train": [
                {
                    "polygon_id": "p2",
                    "lat": 48.8,
                    "lon": 2.3,
                    "worldcover_code": 50,
                    "worldcover_label": "Built-up",
                },
                {
                    "polygon_id": "p1",
                    "lat": 51.5,
                    "lon": -0.1,
                    "worldcover_code": 10,
                    "worldcover_label": "Tree cover",
                },
                {
                    "polygon_id": "p1",
                    "lat": 51.5,
                    "lon": -0.1,
                    "worldcover_code": 10,
                    "worldcover_label": "Tree cover",
                },
            ],
            "validation": [
                {
                    "polygon_id": "p2",
                    "lat": 48.8,
                    "lon": 2.3,
                    "worldcover_code": 50,
                    "worldcover_label": "Built-up",
                },
            ],
        },
    )

    result = centroids_from_build(build)

    assert result["polygon_id"].tolist() == ["p1", "p2"]
    assert result[["lat", "lon", "worldcover_code"]].to_dict("records") == [
        {"lat": 51.5, "lon": -0.1, "worldcover_code": 10},
        {"lat": 48.8, "lon": 2.3, "worldcover_code": 50},
    ]
```

- [ ] **Step 2: Add validation failure tests.**

Cover conflicting labels for one polygon, invalid coordinates and a code that is not a WorldCover code. Each case must raise `CoverageMapError`. The message must identify the invariant.

```python
def test_conflicting_labels_are_rejected(tmp_path: Path) -> None:
    build = _write_build(
        tmp_path / "build",
        {
            "train": [
                {
                    "polygon_id": "p",
                    "lat": 1,
                    "lon": 2,
                    "worldcover_code": 10,
                    "worldcover_label": "Tree cover",
                },
                {
                    "polygon_id": "p",
                    "lat": 1,
                    "lon": 2,
                    "worldcover_code": 50,
                    "worldcover_label": "Built-up",
                },
            ]
        },
    )

    with pytest.raises(CoverageMapError, match="conflicting"):
        centroids_from_build(build)


@pytest.mark.parametrize("lat, lon", [(91, 0), (0, 181)])
def test_out_of_range_coordinates_are_rejected(tmp_path: Path, lat: float, lon: float) -> None:
    build = _write_build(
        tmp_path / "build",
        {
            "train": [
                {
                    "polygon_id": "p",
                    "lat": lat,
                    "lon": lon,
                    "worldcover_code": 10,
                    "worldcover_label": "Tree cover",
                },
            ]
        },
    )

    with pytest.raises(CoverageMapError, match="coordinates"):
        centroids_from_build(build)
```

- [ ] **Step 3: Add a renderer test with a local boundary fixture.**

Build a one-polygon GeoDataFrame in EPSG:4326. Call `write_coverage_map()` with it. Assert that the output is a non-empty PNG. Assert that the returned count is the number of unique polygons. This prevents a test from needing the Natural Earth URL.

```python
def test_write_coverage_map_creates_png(tmp_path: Path) -> None:
    build = _write_build(
        tmp_path / "build",
        {
            "train": [
                {
                    "polygon_id": "p",
                    "lat": 1,
                    "lon": 2,
                    "worldcover_code": 10,
                    "worldcover_label": "Tree cover",
                },
            ]
        },
    )
    land = gpd.GeoDataFrame(
        {"geometry": [shapely.geometry.box(-180, -90, 180, 90)]},
        crs="EPSG:4326",
    )

    count = write_coverage_map(build, build / MAP_FILENAME, land=land)

    assert count == 1
    assert (build / MAP_FILENAME).read_bytes().startswith(b"\x89PNG")
```

- [ ] **Step 4: Add card assertions.**

Extend the existing card tests. They must require `worldcover_centroids.png`, the phrase `polygon centroid` and the polygon total of the manifest. This is the red test for the new card section.

- [ ] **Step 5: Run only the new tests and confirm that they fail.**

Run:

```bash
TMPDIR="$PWD/data/scratch/map-tests" UV_CACHE_DIR="$PWD/data/cache/uv" UV_LINK_MODE=copy uv run --no-sync pytest tests/unit/test_coverage_map.py tests/unit/test_card.py -q
```

Expected: collection failures or assertion failures. The adapter functions and the card section do not exist yet.

## Task 2: Implement centroid aggregation and PNG rendering

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `src/osm_worldcover/adapters/coverage_map.py`

- [ ] **Step 1: Add Matplotlib to the runtime dependencies.**

Add `"matplotlib>=3.9"` next to the existing geospatial dependencies. Then run `uv lock` with the project cache on Seagate. Do not add a second plotting stack.

- [ ] **Step 2: Implement the map adapter around the exact release files.**

The public interface is:

```python
MAP_FILENAME = "worldcover_centroids.png"


class CoverageMapError(ValueError):
    """The release cannot be represented by a valid centroid map."""


def centroids_from_build(build_dir: Path) -> pd.DataFrame:
    """Return one validated ESA-labelled centroid row per polygon."""


def write_coverage_map(
    build_dir: Path,
    output_path: Path,
    *,
    land: gpd.GeoDataFrame | None = None,
) -> int:
    """Render the release centroid map and return its unique-polygon count."""
```

Read only the three Parquets through the DuckDB function `read_parquet`. Select `polygon_id`, `lat`, `lon`, `worldcover_code` and `worldcover_label`. Filter to the ESA-labelled rows. Reject null polygon IDs and invalid labelled coordinates. Group by the string polygon ID. Reject `count(DISTINCT worldcover_code) != 1` and `count(DISTINCT worldcover_label) != 1`. Validate each code/label pair with `domain.nomenclature.CLASS_LABELS`. Order the returned frame by `polygon_id` for deterministic rendering.

Use the fixed ESA palette. The keys are the existing 11 codes:

```python
CLASS_COLORS = {
    10: "#006400",
    20: "#ffbb22",
    30: "#ffff4c",
    40: "#f096ff",
    50: "#fa0000",
    60: "#b4b4b4",
    70: "#f0f0f0",
    80: "#0064c8",
    90: "#0096a0",
    95: "#00cf75",
    100: "#fae6a0",
}
```

For production rendering, load the 1:110m Natural Earth land outline from `https://naturalearth.s3.amazonaws.com/110m_physical/ne_110m_land.zip` with GeoPandas. Use the Matplotlib `Agg` backend and an 18x10 inch figure. Set the longitude limits to `[-180, 180]` and the latitude limits to `[-90, 90]`. Use a light land fill, tiny rasterized semi-transparent points and graticules. Add a legend with two or three columns. The legend contains each class and its point count. Save a PNG with `bbox_inches="tight"` and close the figure. The function must create the parent directory of the output. It must not change the Parquets.

- [ ] **Step 3: Run the map tests and confirm that they pass.**

Run the command from Task 1. Expected: all aggregation, validation and local PNG tests pass. They do not download Natural Earth.

- [ ] **Step 4: Run static checks for the new adapter.**

Run:

```bash
TMPDIR="$PWD/data/scratch/map-checks" UV_CACHE_DIR="$PWD/data/cache/uv" UV_LINK_MODE=copy uv run --no-sync ruff check src/osm_worldcover/adapters/coverage_map.py tests/unit/test_coverage_map.py
TMPDIR="$PWD/data/scratch/map-checks" UV_CACHE_DIR="$PWD/data/cache/uv" UV_LINK_MODE=copy uv run --no-sync ty check src/osm_worldcover/adapters/coverage_map.py
```

Expected: both commands exit with success.

- [ ] **Step 5: Make future release Parquets scannable by the Dataset Viewer.**

Keep `write_batches()` streaming. Pass a fixed `PARQUET_ROW_GROUP_SIZE = 25_000` to `ParquetWriter.write_batch()`. Enable `write_page_index=True`. Add a writer regression test. The test feeds one batch with `25_001` rows. It asserts that there are two row groups and that each row group is not larger than the constant. This prevents a single large Arrow batch from producing a Parquet file that is more than the scan-size limit of the Dataset Viewer.

## Task 3: Add the card section and publisher integration

**Files:**
- Modify: `src/osm_worldcover/domain/card.py`
- Modify: `src/osm_worldcover/adapters/publish.py`
- Modify: `tests/unit/test_card.py`
- Modify: `tests/unit/test_publish.py`

- [ ] **Step 1: Make the publisher test observe map generation.**

Patch `osm_worldcover.adapters.publish.write_coverage_map` with a local fake. The fake writes a PNG signature to the requested path and returns `1`. Assert that `publish_dataset()` calls it with the build directory and `build_dir / MAP_FILENAME`. Then assert that the generated README still contains the class table and the map filename. Keep the existing contract of `files_to_publish()` with four core files. The build generates the map before the folder upload. The map is not a new Dataset Viewer split.

```python
def fake_map(build_dir: Path, output_path: Path, **_: object) -> int:
    output_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    return 1


monkeypatch.setattr("osm_worldcover.adapters.publish.write_coverage_map", fake_map)
```

- [ ] **Step 2: Run the publisher test and confirm that the new integration assertion fails.**

Run:

```bash
TMPDIR="$PWD/data/scratch/publish-tests" UV_CACHE_DIR="$PWD/data/cache/uv" UV_LINK_MODE=copy uv run --no-sync pytest tests/unit/test_publish.py -q
```

Expected: the new map-generation assertion fails. Publication does not call the adapter yet.

- [ ] **Step 3: Add the card section.**

Insert `_coverage_map(counts)` immediately after `_intro()` in `render()`:

```python
def _coverage_map(counts: Mapping[str, Any]) -> str:
    polygons = counts["polygons"]["total"]
    return f"""
## Geographic coverage map

![World map of ESA-labelled polygon centroids](worldcover_centroids.png)

The map shows the centroids of all **{polygons:,} distinct polygons** represented
in this release and carrying an ESA WorldCover label. Each point is colored by
its WorldCover class. Points are centroids, not polygon boundaries; use
`lat`, `lon`, and `centroid_wkt` for the tabular locations.
"""
```

Keep the renderer driven by the manifest. It references the stable release asset name. It takes the count from the same manifest as the rest of the card.

- [ ] **Step 4: Connect publication to the generation of the asset before the README.**

Import `MAP_FILENAME` and `write_coverage_map`. Then add:

```python
map_path = build_dir / MAP_FILENAME
write_coverage_map(build_dir, map_path)
(build_dir / "README.md").write_text(render(manifest))
```

The build must generate the map before `upload_folder`. An invalid release then fails before a remote change.

- [ ] **Step 5: Run the card and publisher tests.**

Run:

```bash
TMPDIR="$PWD/data/scratch/card-publish-tests" UV_CACHE_DIR="$PWD/data/cache/uv" UV_LINK_MODE=copy uv run --no-sync pytest tests/unit/test_card.py tests/unit/test_publish.py -q
```

Expected: all tests pass, with the map link and the publisher call.

- [ ] **Step 6: Commit the implementation and tests.**

Run:

```bash
git add pyproject.toml uv.lock src/osm_worldcover/adapters/coverage_map.py src/osm_worldcover/adapters/publish.py src/osm_worldcover/domain/card.py tests/unit/test_coverage_map.py tests/unit/test_publish.py tests/unit/test_card.py docs/superpowers/plans/2026-09-14-worldcover-centroid-map.md
git commit -m "feat: add worldcover centroid coverage map"
```

## Task 4: Regenerate, inspect, and publish the release card asset

**Files:**
- Modify: `data/out/v1.0.0/README.md` (generated release artifact)
- Create: `data/out/v1.0.0/worldcover_centroids.png` (generated release artifact)

- [ ] **Step 1: Run the full local release-card generation through the publisher.**

Use the project-scoped cache and the token that the existing public release already uses:

```bash
mkdir -p data/scratch/worldcover-map-publish
HF_TOKEN="$(< /Users/noeflandre/.cache/huggingface/token)" \
TMPDIR="$PWD/data/scratch/worldcover-map-publish" \
HF_HOME="$PWD/data/cache/hf" \
UV_CACHE_DIR="$PWD/data/cache/uv" \
UV_LINK_MODE=copy \
HF_HUB_DISABLE_PROGRESS_BARS=1 \
uv run owc publish data/out/v1.0.0 NoeFlandre/osm-wikidata-worldcover
```

Expected: the command writes a non-empty PNG, rewrites the card with the image section and uploads the folder. No Parquet byte changes.

- [ ] **Step 2: Inspect the generated map.**

Open `data/out/v1.0.0/worldcover_centroids.png` in the image viewer. Check that these items are legible: the world outline, the global point coverage, the class colors, the legend and the centroid caption. Check that the generated README contains exactly one `worldcover_centroids.png` image link.

- [ ] **Step 3: Confirm that the public Hub tree contains the map asset.**

Run:

```bash
HF_TOKEN="$(< /Users/noeflandre/.cache/huggingface/token)" HF_HOME="$PWD/data/cache/hf" hf datasets info NoeFlandre/osm-wikidata-worldcover --expand siblings
```

Expected: `README.md`, `worldcover_centroids.png`, `manifest.json` and the three Parquets are present. The build added no split file and no schema file.

- [ ] **Step 4: Rewrite any release file that reaches the Viewer scan limit.**

If Viewer processing reports `TooBigContentError`, stream each affected release Parquet through `ParquetFile.iter_batches(batch_size=25_000)` into a temporary file. Use the same compression, page-index and row-group settings as `write_batches()`. Verify the row counts and the row-group cap. Then atomically replace the exact release file. Run `/first-rows` and `/rows` again for each split. Only then is the publication complete.

## Task 5: Verify the public Dataset Viewer and final quality gates

**Files:**
- No source changes. Inspect the public dataset and the repository state.

- [ ] **Step 1: Check Dataset Viewer validity and split metadata.**

Query these public endpoints. Save the JSON under `data/scratch/worldcover-map-publish`:

```bash
curl -fsS "https://datasets-server.huggingface.co/is-valid?dataset=NoeFlandre%2Fosm-wikidata-worldcover"
curl -fsS "https://datasets-server.huggingface.co/splits?dataset=NoeFlandre%2Fosm-wikidata-worldcover"
```

Expected: the validity is successful. The default config shows exactly `train`, `validation` and `test`. None of them is pending or failed.

- [ ] **Step 2: Check rows and schema for every split.**

For each split, query:

```bash
curl -fsS "https://datasets-server.huggingface.co/first-rows?dataset=NoeFlandre%2Fosm-wikidata-worldcover&config=default&split=train"
curl -fsS "https://datasets-server.huggingface.co/rows?dataset=NoeFlandre%2Fosm-wikidata-worldcover&config=default&split=train&offset=0&length=100"
curl -fsS "https://datasets-server.huggingface.co/statistics?dataset=NoeFlandre%2Fosm-wikidata-worldcover&config=default&split=train"
```

Do the same with `validation` and `test`. Assert that each response has HTTP 200, has non-empty rows, has the expected feature names and scalar values, and reports no processing error. Also check that the Parquet endpoint returns one URL for each split.

- [ ] **Step 3: Run the complete local quality suite.**

Run:

```bash
TMPDIR="$PWD/data/scratch/worldcover-map-qa" UV_CACHE_DIR="$PWD/data/cache/uv" UV_LINK_MODE=copy uv run --no-sync pytest -q
TMPDIR="$PWD/data/scratch/worldcover-map-qa" UV_CACHE_DIR="$PWD/data/cache/uv" UV_LINK_MODE=copy uv run --no-sync ruff check .
TMPDIR="$PWD/data/scratch/worldcover-map-qa" UV_CACHE_DIR="$PWD/data/cache/uv" UV_LINK_MODE=copy uv run --no-sync ty check src tests
git diff --check
```

Expected: all tests and static checks pass. `git diff --check` shows no error.

- [ ] **Step 4: Verify release invariants and repository state.**

Compare the Parquet SHA-256 hashes before and after publication. Confirm that the map count is equal to the distinct polygon total of the manifest. Remove only the `data/scratch/worldcover-map-publish` directory of this task and the related map-test directories. Run `git status --short --branch`. The final report must include the public dataset URL, the PNG path, the Viewer results for all three splits, the test commands, and any quality gate that the engineer did not run.
