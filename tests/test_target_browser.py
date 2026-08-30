from __future__ import annotations

from fractal_flight_studio.deep_zoom_targets import load_deep_zoom_targets
from fractal_flight_studio.target_browser import (
    ALL_TARGETS,
    FAVORITES,
    filter_deep_zoom_targets,
    target_categories,
)


def test_target_categories_include_tags_and_standard_filters():
    categories = target_categories(load_deep_zoom_targets())

    assert categories[:2] == (ALL_TARGETS, FAVORITES)
    assert "Spiralen" in categories
    assert "Filamente" in categories
    assert len(categories) == len(set(categories))


def test_target_filter_searches_names_descriptions_and_tags():
    targets = load_deep_zoom_targets()

    spiral = filter_deep_zoom_targets(targets, query="spiral")
    filaments = filter_deep_zoom_targets(targets, category="Filamente")
    satellite = filter_deep_zoom_targets(targets, query="satellit")

    assert spiral
    assert all("spiral" in " ".join((target.name, target.description, *target.tags)).casefold() for target in spiral)
    assert filaments
    assert all("Filamente" in target.tags for target in filaments)
    assert {target.id for target in satellite} >= {
        "seahorse-satellite",
        "mini-mandelbrot-island",
    }


def test_target_filter_combines_query_and_category_without_reordering():
    targets = load_deep_zoom_targets()
    expected = tuple(
        target for target in targets if "Spiralen" in target.tags and "seahorse" in target.name.casefold()
    )

    assert filter_deep_zoom_targets(
        targets,
        query="seahorse",
        category="Spiralen",
    ) == expected



def test_packaged_target_previews_are_valid_and_distinct():
    from io import BytesIO

    from PIL import Image

    from fractal_flight_studio.target_thumbnail_data import (
        THUMBNAIL_HEIGHT,
        THUMBNAIL_WIDTH,
        thumbnail_bytes,
        thumbnail_ids,
    )

    targets = load_deep_zoom_targets()
    assert thumbnail_ids() == tuple(sorted(target.id for target in targets))
    for target in targets:
        with BytesIO(thumbnail_bytes(target.id)) as stream:
            preview = Image.open(stream).convert("RGB")
            preview.load()
        assert preview.size == (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT)
        colors = preview.getcolors(maxcolors=THUMBNAIL_WIDTH * THUMBNAIL_HEIGHT)
        assert colors is not None
        assert len(colors) >= 16, target.id
