// Adapted from DroneRTS, client/navigation-lights.ts, commit 0dcf03f (MIT, © 2026 Dan McInerney;
// see LICENSE in this folder). Changes: TypeScript types removed; the apron beacon bank, which
// belongs to the game's cargo pads, is left out.
import * as THREE from 'three';

/** Deterministic optical animation driven by the time passed in. */
export function navigationIntensity(pattern, time, phase = 0) {
  const cycle = ((Math.max(0, Number.isFinite(time) ? time : 0) + phase) % 2.4) / 2.4;
  const pulse = (center, width) => Math.max(0, 1 - Math.abs(cycle - center) / width);
  // Slow roof pulses and two separated airframe flashes; never a full-screen strobe.
  return pattern === 'drone' ? Math.max(pulse(0.09, 0.045), pulse(0.29, 0.045))
    : pattern === 'cargo' ? pulse(0.22, 0.12) : pulse(0.25, 0.2);
}

export function navigationPhase(id) {
  let hash = 0;
  for (const character of id) hash = (hash * 31 + character.charCodeAt(0)) >>> 0;
  return ((Math.imul(hash, 0x9e3779b1) >>> 0) % 240) / 100;
}

export function bindNavigationMaterial(material, pattern, phase = 0, halo = false) {
  material.userData.navigationEmitter = { pattern, phase, halo };
}

/** Small physical lens, housing and local halo. Ordinary depth/fog apply to all. */
export function navigationBeacon(color, radius, pattern, phase = 0) {
  const group = new THREE.Group(); group.name = 'navigation-beacon';
  const housing = new THREE.Mesh(new THREE.CylinderGeometry(radius * 1.18, radius * 1.32, radius * 0.3, 12),
    new THREE.MeshStandardMaterial({ color: '#343c40', roughness: 0.62, metalness: 0.5 }));
  housing.position.y = radius * 0.15; group.add(housing);
  const material = new THREE.MeshStandardMaterial({ color: new THREE.Color(color).multiplyScalar(0.3),
    emissive: color, emissiveIntensity: 0.65, roughness: 0.25, metalness: 0.1, toneMapped: false });
  bindNavigationMaterial(material, pattern, phase);
  const lens = new THREE.Mesh(new THREE.SphereGeometry(radius, 12, 8), material);
  lens.name = 'navigation-lens'; lens.scale.y = 0.5; lens.position.y = radius * 0.6; group.add(lens);
  const glowMaterial = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.025,
    depthTest: true, depthWrite: false, toneMapped: false, fog: true });
  bindNavigationMaterial(glowMaterial, pattern, phase, true);
  const halo = new THREE.Mesh(new THREE.SphereGeometry(radius * 1.8, 10, 6), glowMaterial);
  halo.name = 'navigation-halo'; halo.position.copy(lens.position); halo.scale.y = 0.7; group.add(halo);
  return group;
}

export function updateNavigationLights(root, time, animated = true) {
  root.traverse(object => {
    if (!(object instanceof THREE.Mesh)) return;
    for (const material of [object.material].flat()) {
      const emitter = material.userData.navigationEmitter;
      if (!emitter) continue;
      const intensity = animated ? navigationIntensity(emitter.pattern, time, emitter.phase) : 0.5;
      if (emitter.halo) material.opacity = 0.018 + intensity * 0.15;
      else if (material instanceof THREE.MeshStandardMaterial || material instanceof THREE.MeshLambertMaterial) material.emissiveIntensity = 0.6 + intensity * 3.2;
    }
  });
}
