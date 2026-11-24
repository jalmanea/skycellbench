# Scene `cincinnati-downtown`: data sources and licences

`scene.json` is built by `build.py` in this folder from the sources below. The same list, with
the exact request of each download, is inside `scene.json` under `sources`.

Area: 2 km by 2 km of downtown Cincinnati and Over-the-Rhine, Ohio, USA. The south-west corner
is 39.0961 N, 84.5225 W; the north-east corner is about 39.1141 N, 84.4994 W. It contains the
820 m by 660 m area of DroneRTS (which is centred near Fountain Square, 39.1015 N, 84.512 W).
The south-east corner of the area is the Ohio River.

| Layer | Source | Licence | Retrieved |
|---|---|---|---|
| Building footprints, heights and levels | OpenStreetMap, Overpass API (`overpass-api.de`) | ODbL 1.0 | 2026-11-24, OSM data of 2026-11-23 22:24 UTC |
| Water, roads, parks | OpenStreetMap, Overpass API (`overpass-api.de`) | ODbL 1.0 | 2026-11-24, OSM data of 2026-11-23 22:25 UTC |
| Ground height | U.S. Geological Survey, 3D Elevation Program (3DEP), bare-earth elevation image service | Public domain (U.S. Government work) | 2026-11-24 |
| Heights of three towers | DroneRTS, `city-research.json` and `CINCINNATI-SOURCES.md`, commit `0dcf03f` | MIT; the values are published architectural heights (Skyscraper Center) | 2026-11-24 |

Required credits:

- **Map data © OpenStreetMap contributors**, licensed under the
  [Open Database License 1.0](https://opendatacommons.org/licenses/odbl/1-0/); see
  <https://www.openstreetmap.org/copyright>. `scene.json` is a database derived from
  OpenStreetMap, so it is under the ODbL too, whatever licence the rest of this project takes.
  Anything that shows the scene must show this credit; the viewer does.
- Elevation: **U.S. Geological Survey, 3D Elevation Program**. No licence restriction; the USGS
  asks to be credited.
- DroneRTS, © 2026 Dan McInerney, MIT licence: <https://github.com/DanMcInerney/DroneRTS>.

## What was taken from DroneRTS, and what was not

DroneRTS draws its city from an OpenStreetMap extract of 266 buildings (16 landmarks and the
250 largest other footprints), fitted with rectangles, on flat ground. Its building heights are
estimates, except for ten towers whose heights it documents from published sources. Its 3-D model
(`cincinnati.glb`) is artwork made for that project.

This scene does not use DroneRTS geometry. It takes the footprints of all mapped buildings from
OpenStreetMap again, as polygons, and the ground from the USGS. From DroneRTS it takes only the
documented height of three towers that OpenStreetMap gives no height for: PNC Tower (150.9 m),
Center at 600 Vine (127.4 m) and Hilton Cincinnati Netherland Plaza (113.4 m).

## How the layers were made

- **Projection.** Local equirectangular about the south-west corner: `x = (lon - lon0) *
  86498.061`, `y = (lat - lat0) * 111017.46` metres, x east and y north. Over 2 km this is
  within a few centimetres of a true metric projection.
- **Buildings.** Every OpenStreetMap way or multipolygon relation tagged `building`, except
  those on a negative `layer` (underground). Footprints are clipped to the area; pieces under
  4 m² are dropped. 2,332 buildings remain. Courtyards (inner rings) are kept as holes.
- **Building height**, first rule that applies, with the number of buildings it decided:

  | Rule | Buildings |
  |---|---|
  | OSM `height` tag | 131 |
  | documented tower height from DroneRTS | 3 |
  | tallest OSM `building:part` inside the outline | 110 |
  | OSM `building:levels` × 4.0 m | 1,561 |
  | 3 m for garages, sheds and roofs with no information | 15 |
  | 8 m (two levels) for every other building with no information | 512 |

  4.0 m per level is the median of height divided by levels over the 145 buildings of the wider
  extract that carry both tags. Each building in `scene.json` names the rule used
  (`height_source`).
- **Terrain.** The 3DEP service was asked for a 2 m raster of the area in geographic
  coordinates; 5 × 5 blocks were averaged to a 10 m grid of 201 × 201 nodes. Heights are metres
  above NAVD88 and run from 138.9 m (the river) to 256.6 m (the hill in the north-east).
- **Water** is the OpenStreetMap `natural=water` areas (the Ohio River and a few fountains).
  Users are not placed on it. Roads and parks are for display only.

## Known limits

- The terrain is bare earth. Bridges, viaducts and elevated roads are not in it and block
  nothing; trees are not in the scene either.
- 527 of the 2,332 buildings (23%) have no height information and get a default; 1,561 more
  have a height estimated from the number of levels.
- OSM `min_height` is ignored, so skywalks and overhangs are solid down to the ground. A
  stadium outline is one solid block of its tagged height.
- The scene is a snapshot of OpenStreetMap at the time above. Running `build.py --download`
  again fetches newer data and can change the scene; the raw downloads of this build are in
  `raw/`, with their SHA-256 in `scene.json`.
