from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from infoflow_pipeline import update_background_report


def make_item(
    *,
    canonical_name: str = "",
    entity_name_raw: str = "",
    file_id: str = "",
) -> dict:
    item = {
        "entity_name_raw": entity_name_raw,
        "source_narrative": {"file_id": file_id},
    }
    if canonical_name:
        item["canonical_name"] = canonical_name
    return item


def block_signature(blocks: list[list[dict]]) -> list[list[str]]:
    return [
        [update_background_report.canonical_name_for_item(item) for item in block]
        for block in blocks
    ]


def test_group_findings_groups_by_canonical_name_and_preserves_first_seen_order():
    items = [
        make_item(canonical_name="PharmEvo", entity_name_raw="PharmEvo", file_id="1"),
        make_item(canonical_name="Julphar", entity_name_raw="Julphar", file_id="2"),
        make_item(canonical_name="PharmEvo", entity_name_raw="PharmEvo Pvt Ltd", file_id="3"),
        make_item(canonical_name="DoH Abu Dhabi", entity_name_raw="Department of Health - Abu Dhabi", file_id="4"),
        make_item(canonical_name="Julphar", entity_name_raw="Gulf Pharmaceutical Industries", file_id="5"),
    ]

    groups = update_background_report.group_findings(items)

    assert [canonical_name for canonical_name, _ in groups] == [
        "PharmEvo",
        "Julphar",
        "DoH Abu Dhabi",
    ]
    assert [len(group_items) for _, group_items in groups] == [2, 2, 1]


def test_build_blocks_rolls_whole_canonical_groups_to_next_block():
    items = [
        make_item(canonical_name="PharmEvo", entity_name_raw="PharmEvo", file_id="1"),
        make_item(canonical_name="PharmEvo", entity_name_raw="PharmEvo Pvt Ltd", file_id="2"),
        make_item(canonical_name="Julphar", entity_name_raw="Julphar", file_id="3"),
        make_item(canonical_name="Julphar", entity_name_raw="Gulf Pharmaceutical Industries", file_id="4"),
        make_item(canonical_name="DoH Abu Dhabi", entity_name_raw="Department of Health - Abu Dhabi", file_id="5"),
    ]

    blocks = update_background_report.build_blocks(items, block_size=3)

    assert block_signature(blocks) == [
        ["PharmEvo", "PharmEvo"],
        ["Julphar", "Julphar", "DoH Abu Dhabi"],
    ]


def test_build_blocks_allows_single_oversized_group_to_exceed_block_size_intact():
    items = [
        make_item(canonical_name="PharmEvo", entity_name_raw="PharmEvo", file_id="1"),
        make_item(canonical_name="PharmEvo", entity_name_raw="PharmEvo Pvt Ltd", file_id="2"),
        make_item(canonical_name="PharmEvo", entity_name_raw="PHARMEVO PRIVATE LIMITED", file_id="3"),
        make_item(canonical_name="PharmEvo", entity_name_raw="PharmEvo LLC", file_id="4"),
        make_item(canonical_name="Julphar", entity_name_raw="Julphar", file_id="5"),
    ]

    blocks = update_background_report.build_blocks(items, block_size=3)

    assert block_signature(blocks) == [
        ["PharmEvo", "PharmEvo", "PharmEvo", "PharmEvo"],
        ["Julphar"],
    ]


def test_build_block_manifest_reports_unique_canonical_names_and_fallbacks():
    blocks = [
        [
            make_item(canonical_name="PharmEvo", entity_name_raw="PharmEvo", file_id="1"),
            make_item(canonical_name="PharmEvo", entity_name_raw="PharmEvo Pvt Ltd", file_id="2"),
            make_item(entity_name_raw="Unnamed Local Distributor", file_id="3"),
        ],
        [
            make_item(canonical_name="Julphar", entity_name_raw="Julphar", file_id="4"),
            make_item(canonical_name="DoH Abu Dhabi", entity_name_raw="Department of Health - Abu Dhabi", file_id="5"),
        ],
    ]

    manifest = update_background_report.build_block_manifest(blocks)

    assert manifest == [
        {
            "block_no": 1,
            "total_blocks": 2,
            "item_count": 3,
            "canonical_names": ["PharmEvo", "Unnamed Local Distributor"],
        },
        {
            "block_no": 2,
            "total_blocks": 2,
            "item_count": 2,
            "canonical_names": ["Julphar", "DoH Abu Dhabi"],
        },
    ]
