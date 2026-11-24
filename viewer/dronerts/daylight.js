// Adapted from DroneRTS, client/daylight.ts, commit 0dcf03f (MIT, © 2026 Dan McInerney; see
// LICENSE in this folder). Changes: TypeScript types removed; DroneRTS works in units of 10 m
// about its own origin, so every length is multiplied by `scale` and the sun aims at `target`;
// the sun keeps its direction but stands `reach` times further away and the shadow map is
// larger, because the scene is larger than the DroneRTS arena.
import * as THREE from 'three';

/** Shared daylight; static shadows render once. `scale` is scene metres per DroneRTS unit. */
export function daylight(scene, renderer, { scale = 10, target = new THREE.Vector3(), extent = 66, reach = 1 } = {}) {
  scene.background = new THREE.Color('#c4d4dc');
  scene.fog = new THREE.Fog('#c4d4dc', 95 * scale * 2, 300 * scale * 2);
  const sky = new THREE.Mesh(new THREE.SphereGeometry(500 * scale, 32, 16), new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false,
    uniforms: { top: { value: new THREE.Color('#658ca9') }, horizon: { value: new THREE.Color('#d8dfe0') } },
    vertexShader: `varying vec3 direction;
      void main(){ direction=position; vec4 clip=projectionMatrix*mat4(mat3(viewMatrix))*vec4(position,1.0);
      gl_Position=clip.xyww; }`,
    fragmentShader: `uniform vec3 top; uniform vec3 horizon; varying vec3 direction;
      void main(){ float h=max(normalize(direction).y,0.0); gl_FragColor=vec4(mix(horizon,top,pow(h,0.6)),1.0);
      #include <tonemapping_fragment>
      #include <colorspace_fragment>
      }`,
  }));
  sky.name = 'daylight-sky'; scene.add(sky);
  // A filtered sky supplies real PBR reflections without external HDR downloads.
  const environmentScene = new THREE.Scene(); environmentScene.add(sky.clone());
  const pmrem = new THREE.PMREMGenerator(renderer);
  const environment = pmrem.fromScene(environmentScene, .06); pmrem.dispose();
  scene.environment = environment.texture; scene.environmentIntensity = .55;
  scene.add(new THREE.HemisphereLight(0xd5e3ef, 0x777362, 1.65));
  const sun = new THREE.DirectionalLight(0xffefd8, 3.1);
  sun.position.set(target.x - 45 * scale * reach, target.y + 85 * scale * reach, target.z + 52 * scale * reach); sun.target.position.copy(target);
  sun.castShadow = true; sun.shadow.mapSize.set(4096, 4096);
  Object.assign(sun.shadow.camera, { left: -extent * scale, right: extent * scale, top: extent * scale, bottom: -extent * scale, near: 1 * scale, far: 220 * scale * (reach + 1) });
  sun.shadow.bias = -.0008; sun.shadow.normalBias = .06 * scale;
  scene.add(sun, sun.target);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.shadowMap.autoUpdate = false;
  return () => { environment.dispose(); sky.geometry.dispose(); sky.material.dispose(); sun.shadow.dispose(); };
}
