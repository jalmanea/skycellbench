// Adapted from DroneRTS, client/city-scene.ts, commit 0dcf03f (MIT, © 2026 Dan McInerney; see
// LICENSE in this folder). Taken: the building wall palette, the roof colour, the wall material
// and disposeGroup. Not taken: createCity itself, which draws DroneRTS's fitted boxes and
// baked Blender city; skycellbench draws scene footprints instead (../main.js).
import * as THREE from 'three';

export const WALL_PALETTE = ['#bcb7a7', '#b6c5c6', '#c5b18f', '#a3b3bb', '#b9a7a0', '#d0c9b5'];
export const ROOF_COLOR = '#626b6e';
export const material = color => new THREE.MeshStandardMaterial({ color, roughness: 0.88 });

export function disposeGroup(group) {
  const geometries = new Set();
  const materials = new Set();
  const textures = new Set();
  group.traverse(object => {
    if (object instanceof THREE.Mesh || object instanceof THREE.Line) {
      if (object instanceof THREE.InstancedMesh) object.dispose();
      geometries.add(object.geometry);
      (Array.isArray(object.material) ? object.material : [object.material]).forEach(item => materials.add(item));
    }
  });
  geometries.forEach(item => item.dispose());
  materials.forEach(item => {
    for (const texture of [item.map, item.roughnessMap, item.metalnessMap]) if (texture) textures.add(texture);
    item.dispose();
  });
  textures.forEach(texture => texture.dispose());
  group.clear();
}
