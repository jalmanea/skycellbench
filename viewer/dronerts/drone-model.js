// Adapted from DroneRTS, client/drone-model.ts (makeLegacyDrone and addNavigationLights), commit
// 0dcf03f (MIT, © 2026 Dan McInerney; see LICENSE in this folder). Changes: TypeScript types
// removed; the colour is passed in instead of looked up from the game's fleet; everything that
// belongs to the game (gun, armour, miner, optics, battery, jammer, cargo) is left out; the
// Blender airframe is not used. The model keeps DroneRTS proportions: about 0.5 units across.
import * as THREE from 'three';
import { bindNavigationMaterial, navigationBeacon, navigationPhase } from './navigation-lights.js';

export function makeDrone(id, color) {
  const group = new THREE.Group(); group.name = id;
  const material = new THREE.MeshLambertMaterial({ color });
  const dark = new THREE.MeshLambertMaterial({ color: '#283640' });
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.29, 0.14, 0.34), material); body.name = 'team-body-panel'; group.add(body);
  for (const x of [-0.19, 0.19]) for (const z of [-0.17, 0.17]) {
    const rotor = new THREE.Mesh(new THREE.CylinderGeometry(0.075, 0.075, 0.025, 12), dark);
    rotor.position.set(x, 0.08, z); group.add(rotor);
    const arm = new THREE.Mesh(new THREE.BoxGeometry(0.31, 0.055, 0.09), material); arm.name = 'team-arm-panel';
    arm.position.set(x / 2, 0, z / 2); arm.rotation.y = x * z > 0 ? -Math.PI / 4 : Math.PI / 4; group.add(arm);
  }
  const camera = new THREE.Mesh(new THREE.BoxGeometry(0.085, 0.07, 0.035), dark); camera.position.set(0, 0.045, -0.175); group.add(camera);
  addNavigationLights(group, id, color);
  return group;
}

function addNavigationLights(group, id, color) {
  const phase = navigationPhase(id);
  group.traverse(object => {
    if (!(object instanceof THREE.Mesh)) return;
    for (const material of [object.material].flat()) if (material.name === 'Team light') bindNavigationMaterial(material, 'drone', phase);
  });
  const lights = new THREE.Group(); lights.name = 'drone-beacons';
  for (const y of [-0.075, 0.075]) {
    const beacon = navigationBeacon(color, 0.024, 'drone', phase);
    beacon.position.y = y; if (y < 0) beacon.rotation.x = Math.PI;
    lights.add(beacon);
  }
  group.add(lights);
}
