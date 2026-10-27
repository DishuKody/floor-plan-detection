| metric | v1 | v2 |
|---|---|---|
| rooms | 106 | 107 |
| by type | bathroom:39, bedroom:38, corridor_or_hallway:7, kitchen:6, lift:2, other:2, shaft_or_riser:1, stair:1, storage:1, unknown:9 | bathroom:39, bedroom:38, communal_or_shared_space:1, corridor_or_hallway:11, kitchen:6, lift:2, other:2, shaft_or_riser:2, stair:1, storage:1, unknown:4 |
| unknown rooms | 9 | 4 |
| flagged review_required | 52 | 48 |
| largest room | room-0050 shaft_or_riser 72.1 m2 | room-0048 communal_or_shared_space 37.6 m2 |
| bedroom units isolated+correct (OCR unit-ID proxy) | 37/39 | 37/39 |
| kitchen units isolated+correct | 6/7 | 6/7 |
| named spaces OK (stair, riser, lifts, AOV, store, lobby, tutor flat) | 3/8 | 7/8 |
| named spaces failing | riser:merged_with:aov+core03_lobby,implausible_area:72.1m2; core03_lobby:wrong_type:shaft_or_riser,merged_with:aov+riser,implausible_area:72.1m2; tutor_flat:implausible_area:3.4m2; stair:implausible_area:6.0m2; aov:merged_with:core03_lobby+riser,implausible_area:72.1m2 | tutor_flat:implausible_area:13.9m2 |
| B503 bedroom area CV / outliers | 0.073 / 2 | 0.077 / 2 |
| B703 bedroom area CV / outliers | 0.007 / 0 | 0.007 / 0 |
| unknown pockets touching a bedroom | 3 | 0 |
| invalid polygons / overlapping pairs | 0 / 0 | 0 / 0 |
| boundary >3pt from a wall (mean fraction) | 0.123 | 0.123 |
| max wall-ink fraction inside a polygon | 0.0058 | 0.0973 |
| coordinate check: boundary-to-ink px / max |px - pt*s| | 0.83 / 0.055 | 0.8 / 0.055 |
| elements door/window/column/wall | 92/14/36/3 | 105/14/36/54 |
| runtime s / peak RAM MB | 39.4 / 997 | 24.4 / 998 |
