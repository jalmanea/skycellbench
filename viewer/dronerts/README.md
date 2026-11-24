# Code reused from DroneRTS

Source: <https://github.com/DanMcInerney/DroneRTS>, commit
`0dcf03f4e1e61ac2fdfe89d3e4cbc6a89380499e`, MIT licence, © 2026 Dan McInerney. The licence text
is in `LICENSE` in this folder and applies to the files here.

DroneRTS is a Three.js game: LLM-piloted drones fight over downtown Cincinnati. The viewer reuses
its visual world only. Each file names the DroneRTS file it comes from and what was changed.

| File here | From DroneRTS | What it is |
|---|---|---|
| `daylight.js` | `client/daylight.ts` | sky, fog, sun, sky reflections and static shadows |
| `explorer.js` | `client/explorer.ts` | the free camera (W A S D, Q/E, drag to look) |
| `drone-model.js` | `client/drone-model.ts` | the quadcopter mesh, without game equipment |
| `navigation-lights.js` | `client/navigation-lights.ts` | the flashing lights on each drone |
| `city-scene.js` | `client/city-scene.ts` | wall palette, roof colour, material, `disposeGroup` |

All were converted from TypeScript to plain JavaScript modules so the viewer needs no build step.

Not reused: the game and its rules, the LLM pilots, the server, MAVLink and the radio network;
the Blender models (`cincinnati.glb`, `drone-kit.glb`) and textures; and the city data
(`city-research.json`, `shared/city-data.json`), which is 266 fitted boxes on flat ground. The
viewer draws skycellbench scenes instead: every OpenStreetMap footprint as a polygon, on real
terrain.
