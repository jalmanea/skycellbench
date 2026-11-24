// Adapted from DroneRTS, client/explorer.ts, commit 0dcf03f (MIT, © 2026 Dan McInerney; see
// LICENSE in this folder). It is the free "god view" camera: W A S D to move, Q and E for
// down and up, Shift for speed, arrow keys or a mouse drag to look. Changes: TypeScript types
// removed; in DroneRTS the explorer is a modal overlay entered from the map, here it is the
// only camera and always active, so the overlay, pointer lock and focus handling are gone;
// bounds are passed in instead of read from the game's city; speeds are in metres (DroneRTS
// units are 10 m); the wheel moves along the view direction.
import * as THREE from 'three';

/** An invisible observer camera. It never changes what it looks at. */
export class Explorer {
  constructor(view, bounds, onMove = () => {}) {
    this.camera = new THREE.PerspectiveCamera(60, 1, 2, 30000);
    this.camera.rotation.order = 'YXZ';
    this.view = view; this.bounds = bounds; this.onMove = onMove;
    this.keys = new Set(); this.dragging = false; this.moved = 0;
    this.abort = new AbortController();
    const options = { signal: this.abort.signal };
    view.addEventListener('pointerdown', event => { if (event.button === 0) { this.dragging = true; this.moved = 0; view.focus(); } }, options);
    window.addEventListener('pointerup', () => { this.dragging = false; }, options);
    window.addEventListener('pointermove', event => {
      if (!this.dragging) return;
      this.moved += Math.abs(event.movementX) + Math.abs(event.movementY);
      this.camera.rotation.y -= event.movementX * 0.0022 * 1.6;
      this.camera.rotation.x = THREE.MathUtils.clamp(this.camera.rotation.x - event.movementY * 0.0022 * 1.6, -1.5, 1.5);
      this.onMove();
    }, options);
    view.addEventListener('wheel', event => {
      event.preventDefault();
      const forward = new THREE.Vector3(); this.camera.getWorldDirection(forward);
      this.camera.position.addScaledVector(forward, -event.deltaY * Math.max(0.15, this.camera.position.y * 0.0012));
      this.clamp(); this.onMove();
    }, { ...options, passive: false });
    view.addEventListener('keydown', event => {
      if (/^(Key[WASDQE]|Arrow(Up|Down|Left|Right)|Shift(Left|Right)|Space)$/.test(event.code)) {
        event.preventDefault(); this.keys.add(event.code);
      }
    }, options);
    window.addEventListener('keyup', event => { this.keys.delete(event.code); }, options);
    window.addEventListener('blur', () => { this.keys.clear(); this.dragging = false; }, options);
    document.addEventListener('visibilitychange', () => { this.keys.clear(); this.dragging = false; }, options);
  }

  /** True when the last press on the view was a click, not a drag. */
  get clicked() { return this.moved < 4; }

  place(x, y, z, yaw, pitch) {
    this.camera.position.set(x, y, z); this.camera.rotation.set(pitch, yaw, 0); this.onMove();
  }

  clamp() {
    const p = this.camera.position, b = this.bounds;
    p.x = THREE.MathUtils.clamp(p.x, b.x[0], b.x[1]);
    p.z = THREE.MathUtils.clamp(p.z, b.z[0], b.z[1]);
    p.y = THREE.MathUtils.clamp(p.y, b.y[0], b.y[1]);
  }

  tick(dt) {
    const down = (...keys) => Number(keys.some(key => this.keys.has(key)));
    const turn = down('ArrowLeft') - down('ArrowRight'), tilt = down('ArrowUp') - down('ArrowDown');
    this.camera.rotation.y += turn * dt * 1.6;
    this.camera.rotation.x = THREE.MathUtils.clamp(this.camera.rotation.x + tilt * dt * 1.3, -1.5, 1.5);
    const movement = new THREE.Vector3(down('KeyD') - down('KeyA'), down('KeyE', 'Space') - down('KeyQ'), down('KeyS') - down('KeyW'));
    if (movement.lengthSq()) {
      movement.normalize().applyAxisAngle(new THREE.Vector3(0, 1, 0), this.camera.rotation.y);
      // Fine control at street height, faster traversal at city-view altitude.
      const travelSpeed = Math.max(25, Math.min(400, this.camera.position.y * 0.8));
      this.camera.position.addScaledVector(movement, dt * travelSpeed * (down('ShiftLeft', 'ShiftRight') ? 4 : 1));
      this.clamp();
    }
    if (movement.lengthSq() || turn || tilt) this.onMove();
  }

  dispose() { this.abort.abort(); }
}
