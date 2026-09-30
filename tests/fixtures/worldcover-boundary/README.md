# WorldCover raster-boundary regression fixtures

These tiny GeoTIFF windows and geometries reproduce exactextract 0.3.0 cell
misclassification on the original ESA WorldCover 2021 v200 tile grids. Tests
expand each window into a sparse, full-grid GeoTIFF so the affine origin and
cell indexing stay identical to production.

- `peru-*`: OSM way 47373550, tile S12W078; both false exterior cells and
  omitted interior cells were observed.
- `netherlands-*`: OSM way 15304016, tile N51E003; one false full-cell was
  observed at a polygon boundary.

The raster bytes are windows from ESA WorldCover 2021 v200. The source pins,
original affine transforms, offsets, and official tile URLs are recorded in the
metadata JSON files. Geometry provenance is the pinned public source snapshot
`NoeFlandre/osm-polygon-wikidata-and-wikipedia` at
`f48c5aaaec6aecd63ecf5c195565cc2787597f1e`. These fixtures are for local
regression tests only; they are not a complete dataset or a basis for claiming
that historical releases have been revalidated.

Attribution: © ESA WorldCover project 2021 / Contains modified Copernicus
Sentinel data (2021) processed by ESA WorldCover consortium. The source data is
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Recommended citation: Zanaga et al. (2022), *ESA WorldCover 10 m 2021 v200*,
[doi:10.5281/zenodo.7254221](https://doi.org/10.5281/zenodo.7254221).
