| metric | v2 | v3 |
|---|---|---|
| rooms | 107 | 109 |
| by type | bathroom:39, bedroom:38, communal_or_shared_space:1, corridor_or_hallway:11, kitchen:6, lift:2, other:2, shaft_or_riser:2, stair:1, storage:1, unknown:4 | bathroom:41, bedroom:38, communal_or_shared_space:1, corridor_or_hallway:13, kitchen:6, lift:2, other:2, shaft_or_riser:2, stair:1, storage:1, unknown:2 |
| unknown rooms | 4 | 2 |
| flagged review_required | 48 | 50 |
| largest room | room-0048 communal_or_shared_space 37.6 m2 | room-0050 communal_or_shared_space 37.6 m2 |
| bedroom units isolated+correct (OCR unit-ID proxy) | 37/39 | 37/39 |
| kitchen units isolated+correct | 6/7 | 6/7 |
| named spaces OK (stair, riser, lifts, AOV, store, lobby, tutor flat) | 7/8 | 7/8 |
| tutor flat as a group (all 'other' rooms, m2) | 25.9 | 25.9 |
| named spaces failing | tutor_flat:implausible_area:13.9m2 | tutor_flat:implausible_area:13.9m2 |
| B503 bedroom area CV / outliers | 0.077 / 2 | 0.077 / 2 |
| B703 bedroom area CV / outliers | 0.007 / 0 | 0.007 / 0 |
| unknown pockets touching a bedroom | 0 | 0 |
| invalid polygons / overlapping pairs | 0 / 0 | 0 / 0 |
| boundary >3pt from a wall (mean fraction) | 0.123 | 0.123 |
| max wall-ink fraction inside a polygon | 0.0973 | 0.1134 |
| coordinate check: boundary-to-ink px / max |px - pt*s| | 0.8 / 0.055 | 0.81 / 0.055 |
| elements door/window/column/wall | 105/14/36/54 | 105/62/36/54 |
| standalone annex pods (no host room) | 0 | 2 |
| runtime s / peak RAM MB | 24.4 / 998 | 32.6 / 2081 |
