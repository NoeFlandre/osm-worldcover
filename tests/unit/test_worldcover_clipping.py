import pytest
from shapely.geometry import MultiPolygon, Polygon

from osm_worldcover.adapters import worldcover


def test_clipping_keeps_only_areal_pieces_for_exactextract():
    # The second square only touches the clip window's edge, so clipping leaves a
    # zero-area line beside the real polygon, as a GeometryCollection.
    touching = MultiPolygon(
        [
            Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
            Polygon([(4, 4), (8, 4), (8, 8), (4, 8)]),
        ]
    )

    chunks = list(worldcover._bounded_pixel_chunks(touching, 8, 4))

    assert chunks
    assert {chunk.geom_type for chunk in chunks} <= {"Polygon", "MultiPolygon"}
    assert sum(chunk.area for chunk in chunks) == pytest.approx(16.0)
