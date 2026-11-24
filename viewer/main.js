// skycellbench viewer: a scene, an instance on it and an answer, in 3-D.
//
// It draws what the scorers used and computes nothing of its own: user throughput, the serving
// UAV and line of sight are read from the answer file (`python -m skycellbench view`). The sky,
// lights, camera, drone model and materials are DroneRTS code (./dronerts); the terrain, the
// building footprints and everything about users and links are drawn here.
//
// Scene coordinates are x east, y north, z up. Three.js has y up, so a scene point (x, y, z) is
// drawn at (x - cx, z - z0, cy - y): north is -Z, as in DroneRTS.
import * as THREE from 'three';
import { daylight } from './dronerts/daylight.js';
import { Explorer } from './dronerts/explorer.js';
import { makeDrone } from './dronerts/drone-model.js';
import { updateNavigationLights } from './dronerts/navigation-lights.js';
import { WALL_PALETTE, ROOF_COLOR, material, disposeGroup } from './dronerts/city-scene.js';

const UE_HEIGHT = 1.5, GATEWAY_HEIGHT = 10, DRONE_SCALE = 44;
const UAV_COLORS = ['#e6550d', '#1f78b4', '#2ca25f', '#7b4ea3', '#d6336c', '#b8860b', '#0e9aa7', '#8c564b',
  '#525252', '#a6761d'];
const SHARE = ['#d73027', '#fc8d59', '#fee090', '#e0f3f8', '#91bfdb', '#4575b4'].map(c => new THREE.Color(c));
const ROADS = { motorway: 15, trunk: 13, primary: 12, secondary: 10, tertiary: 9, residential: 7,
  unclassified: 7, living_street: 6, pedestrian: 5 };
const $ = id => document.getElementById(id);
const fmt = (v, n = 1) => Number(v).toFixed(n);

function shareColor(s) {
  const t = Math.min(1, Math.max(0, s)) * (SHARE.length - 1), i = Math.min(SHARE.length - 2, Math.floor(t));
  return SHARE[i].clone().lerp(SHARE[i + 1], t - i);
}

// ---- the world ---------------------------------------------------------------------------------

class World {
  constructor(view) {
    this.view = view;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    view.append(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.disposeDaylight = daylight(this.scene, this.renderer, { scale: 10, extent: 160, reach: 3 });
    this.city = new THREE.Group(); this.answer = new THREE.Group(); this.rules = new THREE.Group();
    this.scene.add(this.city, this.answer, this.rules);
    this.explorer = new Explorer(view, { x: [-4000, 4000], y: [3, 7000], z: [-4000, 4000] });
    this.resetView();
    new ResizeObserver(() => this.resize()).observe(view);
    this.resize();
    this.clock = performance.now();
    const frame = time => {
      requestAnimationFrame(frame);
      const dt = Math.min(0.05, (time - this.clock) / 1000); this.clock = time;
      this.explorer.tick(dt);
      updateNavigationLights(this.answer, time / 1000);
      // Shadows are static, as in DroneRTS: redrawn for a few frames after the city changes.
      if (this.shadowFrames > 0) { this.shadowFrames--; this.renderer.shadowMap.needsUpdate = true; }
      this.renderer.render(this.scene, this.explorer.camera);
    };
    requestAnimationFrame(frame);
  }

  resize() {
    const w = this.view.clientWidth, h = this.view.clientHeight;
    this.renderer.setSize(w, h, false);
    this.explorer.camera.aspect = w / Math.max(h, 1); this.explorer.camera.updateProjectionMatrix();
  }

  resetView() { this.explorer.place(-260, 1250, 1900, 0, -0.62); }      // left of centre: the panel covers that side
  topView() { this.explorer.place(0, 2700, 1, 0, -1.5); }

  // -- scene geometry --

  setScene(s) {
    this.s = s;
    const t = s.terrain;
    this.cx = s.size_m[0] / 2; this.cy = s.size_m[1] / 2;
    this.z0 = Math.min(...t.z_m.map(row => Math.min(...row)));
    disposeGroup(this.city);
    this.city.add(this.makeTerrain(), this.makeBorder());
    this.buildings = this.makeBuildings(); this.city.add(this.buildings);
    this.shadowFrames = 4;
  }

  ground(x, y) {
    const t = this.s.terrain, nx = t.nx, ny = t.ny;
    const fx = Math.min(nx - 1, Math.max(0, x / t.cell_m)), fy = Math.min(ny - 1, Math.max(0, y / t.cell_m));
    const ix = Math.min(nx - 2, Math.floor(fx)), iy = Math.min(ny - 2, Math.floor(fy)), ax = fx - ix, ay = fy - iy;
    const z = t.z_m;
    return (1 - ay) * ((1 - ax) * z[iy][ix] + ax * z[iy][ix + 1]) + ay * ((1 - ax) * z[iy + 1][ix] + ax * z[iy + 1][ix + 1]);
  }

  /** Scene point (x east, y north, z above the datum) to a Three.js position. */
  at(x, y, z) { return new THREE.Vector3(x - this.cx, z - this.z0, this.cy - y); }
  onGround(x, y, above = 0) { return this.at(x, y, this.ground(x, y) + above); }

  makeTerrain() {
    const s = this.s, t = s.terrain, [w, h] = s.size_m;
    const pos = [], uv = [], index = [];
    for (let j = 0; j < t.ny; j++) for (let i = 0; i < t.nx; i++) {
      const x = Math.min(w, i * t.cell_m), y = Math.min(h, j * t.cell_m), p = this.at(x, y, t.z_m[j][i]);
      pos.push(p.x, p.y, p.z); uv.push(x / w, y / h);
    }
    for (let j = 0; j < t.ny - 1; j++) for (let i = 0; i < t.nx - 1; i++) {
      const a = j * t.nx + i, b = a + 1, c = a + t.nx, d = c + 1;
      index.push(a, b, d, a, d, c);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
    g.setIndex(index); g.computeVertexNormals();
    const surface = material('#ffffff'); surface.map = this.groundTexture(); surface.roughness = 0.96;
    const mesh = new THREE.Mesh(g, surface); mesh.name = 'terrain'; mesh.receiveShadow = true;
    return mesh;
  }

  /** Streets, parks, water and footprints painted on the ground, as DroneRTS paints its ground. */
  groundTexture() {
    const s = this.s, [w, h] = s.size_m, N = 4096, k = N / w;
    const canvas = document.createElement('canvas'); canvas.width = N; canvas.height = Math.round(N * h / w);
    const ctx = canvas.getContext('2d');
    const P = ([x, y]) => [x * k, (h - y) * k];
    const fill = (items, color) => {
      ctx.fillStyle = color;
      for (const item of items) {
        ctx.beginPath();
        for (const ring of item.rings) { ring.forEach((p, i) => { const q = P(p); i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]); }); ctx.closePath(); }
        ctx.fill('evenodd');
      }
    };
    ctx.fillStyle = '#b3b0a3'; ctx.fillRect(0, 0, canvas.width, canvas.height);
    const display = s.display || {};
    fill(display.parks || [], '#93ad7c');
    fill(s.water || [], '#7fa9c0');
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    const roads = display.roads || [];
    for (const [extra, color] of [[5, '#cbc7ba'], [0, '#5d6368']]) {       // pavement, then asphalt
      ctx.strokeStyle = color;
      for (const road of roads) {
        const width = ROADS[road.kind] ?? (road.kind.endsWith('_link') ? 6 : 6);
        ctx.lineWidth = (width + extra) * k; ctx.beginPath();
        road.points.forEach((p, i) => { const q = P(p); i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]); });
        ctx.stroke();
      }
    }
    fill(s.buildings, '#8b887e');
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace; texture.anisotropy = 8;
    return texture;
  }

  makeBorder() {
    const [w, h] = this.s.size_m, pts = [];
    for (const [x0, y0, x1, y1] of [[0, 0, w, 0], [w, 0, w, h], [w, h, 0, h], [0, h, 0, 0]])
      for (let i = 0; i <= 100; i++) pts.push(this.onGround(x0 + (x1 - x0) * i / 100, y0 + (y1 - y0) * i / 100, 1.5));
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: '#2b3a44' }));
    line.name = 'area-border';
    return line;
  }

  /** Every footprint extruded from the ground to its height: walls per edge, roof by ear clipping. */
  makeBuildings() {
    const pos = [], col = [];
    const roof = new THREE.Color(ROOF_COLOR), wall = new THREE.Color();
    const push = (p, c) => { pos.push(p.x, p.y, p.z); col.push(c.r, c.g, c.b); };
    this.s.buildings.forEach((b, index) => {
      const rings = b.rings.map(r => r.filter((p, i) => {          // no repeated points, no closing point
        const q = r[i === 0 ? r.length - 1 : i - 1];
        return p[0] !== q[0] || p[1] !== q[1];
      })).filter(r => r.length >= 3);
      if (!rings.length) return;
      const grounds = rings[0].map(p => this.ground(p[0], p[1]));
      const base = Math.min(...grounds) - 1, top = grounds.reduce((a, v) => a + v, 0) / grounds.length + b.height_m;
      wall.set(WALL_PALETTE[index % WALL_PALETTE.length]);
      if (b.height_m > 60) wall.lerp(new THREE.Color('#8fb0c2'), 0.45);   // towers read as glass
      for (const ring of rings) for (let i = 0; i < ring.length; i++) {
        const a = ring[i], c = ring[(i + 1) % ring.length];
        const a0 = this.at(a[0], a[1], base), c0 = this.at(c[0], c[1], base), a1 = this.at(a[0], a[1], top), c1 = this.at(c[0], c[1], top);
        for (const p of [a0, c0, c1, a0, c1, a1]) push(p, wall);
      }
      const v2 = r => r.map(p => new THREE.Vector2(p[0], p[1]));
      const all = rings.flat();
      for (const tri of THREE.ShapeUtils.triangulateShape(v2(rings[0]), rings.slice(1).map(v2)))
        for (const i of tri) push(this.at(all[i][0], all[i][1], top), roof);
    });
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    g.computeVertexNormals();
    const surface = material('#ffffff'); surface.vertexColors = true; surface.side = THREE.DoubleSide;
    const mesh = new THREE.Mesh(g, surface); mesh.name = 'buildings'; mesh.castShadow = true; mesh.receiveShadow = true;
    return mesh;
  }

  setGlass(on) {
    const m = this.buildings.material;
    m.transparent = on; m.opacity = on ? 0.35 : 1; m.depthWrite = !on; m.needsUpdate = true;
    this.buildings.castShadow = !on && this.buildings.visible; this.shadowFrames = 4;
  }

  // -- instance and answer --

  /** Draw one epoch. `e` holds users [{x, y, demand, thr, serving, los}] and uavs [[x, y, alt]]. */
  setEpoch(inst, e) {
    disposeGroup(this.answer); disposeGroup(this.rules);
    const scored = e.users.some(u => u.thr !== undefined);
    // users
    const ball = new THREE.SphereGeometry(1, 14, 10);
    const users = new THREE.InstancedMesh(ball, new THREE.MeshBasicMaterial({ color: '#ffffff', toneMapped: false }), e.users.length);
    users.name = 'users';
    const dummy = new THREE.Object3D();
    e.users.forEach((u, i) => {
      const r = 5 + 3.4 * Math.sqrt(u.demand);
      dummy.position.copy(this.onGround(u.x, u.y, r + 0.5)); dummy.scale.setScalar(r); dummy.updateMatrix();
      users.setMatrixAt(i, dummy.matrix);
      users.setColorAt(i, scored ? shareColor(Math.min(1, u.thr / u.demand)) : new THREE.Color('#9aa3a8'));
    });
    users.instanceMatrix.needsUpdate = true; if (users.instanceColor) users.instanceColor.needsUpdate = true;
    users.computeBoundingSphere();
    this.users = users; this.answer.add(users);
    // UAVs: the DroneRTS model, a stem to the ground and a number
    const uavAt = e.uavs.map(([x, y, alt]) => this.onGround(x, y, alt));
    e.uavs.forEach(([x, y, alt], k) => {
      const color = UAV_COLORS[k % UAV_COLORS.length];
      const drone = makeDrone(`uav-${k + 1}`, color);
      drone.scale.setScalar(DRONE_SCALE); drone.position.copy(uavAt[k]);
      const foot = this.onGround(x, y, 0.6);
      const stem = new THREE.Line(new THREE.BufferGeometry().setFromPoints([foot, uavAt[k]]),
        new THREE.LineDashedMaterial({ color, dashSize: 8, gapSize: 6 }));
      stem.computeLineDistances();
      const ring = new THREE.Mesh(new THREE.RingGeometry(9, 13, 32), new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide }));
      ring.rotation.x = -Math.PI / 2; ring.position.copy(foot);
      const label = this.label(String(k + 1), color); label.position.copy(uavAt[k]).add(new THREE.Vector3(0, 26, 0));
      this.answer.add(drone, stem, ring, label);
    });
    // which UAV serves which user
    this.links = {};
    for (const kind of ['los', 'nlos']) {
      const pos = [], col = [];
      e.users.forEach(u => {
        if (u.serving === undefined || (u.los === false) !== (kind === 'nlos')) return;
        const a = this.onGround(u.x, u.y, UE_HEIGHT), b = uavAt[u.serving], c = new THREE.Color(UAV_COLORS[u.serving % UAV_COLORS.length]);
        pos.push(a.x, a.y, a.z, b.x, b.y, b.z); col.push(c.r, c.g, c.b, c.r, c.g, c.b);
      });
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
      g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
      const lines = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true,
        opacity: kind === 'los' ? 0.9 : 0.22, depthWrite: false }));
      lines.name = `links-${kind}`; this.links[kind] = lines; this.answer.add(lines);
    }
    // constrained task: no-fly zones, the gateway and the backhaul hops that are within range
    for (const [x, y, r] of inst.no_fly || []) {
      const lo = this.ground(x, y) - 30, hi = this.ground(x, y) + inst.alt_max_m;
      const zone = new THREE.Mesh(new THREE.CylinderGeometry(r, r, hi - lo, 64, 1, true),
        new THREE.MeshBasicMaterial({ color: '#c0392b', transparent: true, opacity: 0.13, side: THREE.DoubleSide, depthWrite: false }));
      zone.position.copy(this.at(x, y, (lo + hi) / 2));
      const rim = [];
      for (let i = 0; i <= 96; i++) rim.push(this.onGround(x + r * Math.cos(i / 96 * 2 * Math.PI), y + r * Math.sin(i / 96 * 2 * Math.PI), 2));
      this.rules.add(zone, new THREE.Line(new THREE.BufferGeometry().setFromPoints(rim), new THREE.LineBasicMaterial({ color: '#c0392b' })));
    }
    if (inst.gateway) {
      const [gx, gy] = inst.gateway, g = this.onGround(gx, gy, GATEWAY_HEIGHT);
      const mast = new THREE.Mesh(new THREE.CylinderGeometry(2.5, 4, GATEWAY_HEIGHT + 30, 10), new THREE.MeshBasicMaterial({ color: '#111111' }));
      mast.position.copy(this.onGround(gx, gy, (GATEWAY_HEIGHT + 30) / 2));
      const tag = this.label('GW', '#111111'); tag.position.copy(g).add(new THREE.Vector3(0, 60, 0));
      const nodes = [g, ...uavAt], hops = [];
      for (let a = 0; a < nodes.length; a++) for (let b = a + 1; b < nodes.length; b++)
        if (nodes[a].distanceTo(nodes[b]) <= inst.backhaul_range_m + 0.05) hops.push(nodes[a], nodes[b]);
      const net = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(hops),
        new THREE.LineDashedMaterial({ color: '#111111', dashSize: 14, gapSize: 9 }));
      net.computeLineDistances();
      this.rules.add(mast, tag, net);
    }
  }

  label(text, color) {
    const canvas = document.createElement('canvas'); canvas.width = canvas.height = 128;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = color; ctx.beginPath(); ctx.arc(64, 64, 58, 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = '#ffffff'; ctx.font = `700 ${text.length > 1 ? 54 : 76}px system-ui, sans-serif`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(text, 64, 69);
    const texture = new THREE.CanvasTexture(canvas); texture.colorSpace = THREE.SRGBColorSpace;
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, depthTest: false, toneMapped: false }));
    sprite.scale.setScalar(34); sprite.renderOrder = 5;
    return sprite;
  }

  /** Index of the user under a pointer position, or -1. */
  pick(clientX, clientY) {
    if (!this.users) return -1;
    const box = this.view.getBoundingClientRect();
    const ray = new THREE.Raycaster();
    ray.setFromCamera(new THREE.Vector2((clientX - box.left) / box.width * 2 - 1, -(clientY - box.top) / box.height * 2 + 1), this.explorer.camera);
    const hit = ray.intersectObject(this.users)[0];
    return hit ? hit.instanceId : -1;
  }
}

// ---- data --------------------------------------------------------------------------------------

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status} ${response.statusText}`);
  return response.json();
}

function readFile(input) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => { try { resolve(JSON.parse(reader.result)); } catch (error) { reject(error); } };
    reader.onerror = () => reject(reader.error);
    reader.readAsText(input.files[0]);
  });
}

/** An instance without a scene stands on flat, empty ground. */
function flatScene(side) {
  return { format: 'skycellbench-scene/1', id: 'flat', size_m: [side, side], channel_cell_m: 2,
    terrain: { cell_m: side, nx: 2, ny: 2, z_m: [[0, 0], [0, 0]] }, buildings: [], water: [], display: {},
    attribution: '' };
}

/** Per epoch: the active users with what they received, and the UAV positions. */
function epochsOf(inst, answer) {
  const single = typeof answer.uavs[0][0] === 'number';
  const plans = single ? [answer.uavs] : answer.uavs;
  return plans.map((uavs, e) => {
    const r = answer.result?.epochs?.[e];
    const users = [];
    inst.users.forEach(([x, y, demand], i) => {
      if (inst.activity && !inst.activity[e][i]) return;
      const [ux, uy] = inst.tracks ? inst.tracks[e][i] : [x, y], n = users.length;
      users.push({ id: i, x: ux, y: uy, demand, thr: r?.thr_mbps[n], serving: r?.serving[n], los: r?.los?.[n] });
    });
    return { uavs, users, score: r?.score };
  });
}

// ---- page --------------------------------------------------------------------------------------

const world = new World($('view'));
if (new URLSearchParams(location.search).get('view') === 'top') world.topView();
const state = { inst: null, answer: null, epochs: [], e: 0, timer: 0 };

function showEpoch() {
  const { inst, epochs } = state, e = epochs[state.e];
  world.setEpoch(inst, e);
  applyToggles();
  const scored = e.score !== undefined, n = e.users.length;
  $('epochText').textContent = epochs.length > 1 ? `epoch ${state.e + 1} of ${epochs.length}` : '';
  $('usersLine').textContent = scored
    ? `${e.users.filter(u => u.thr >= u.demand * 0.999).length} of ${n} users get all they ask for; ${e.users.filter(u => !(u.thr > 0)).length} get nothing.`
    : `${n} users.`;
  const body = $('uavs').tBodies[0]; body.replaceChildren();
  e.uavs.forEach(([, , alt], k) => {
    const mine = e.users.filter(u => u.serving === k), sight = mine.filter(u => u.los !== false).length;
    const asked = mine.reduce((a, u) => a + u.demand, 0), got = mine.reduce((a, u) => a + (u.thr || 0), 0);
    const row = body.insertRow();
    row.innerHTML = `<td><span class="chip" style="background:${UAV_COLORS[k % UAV_COLORS.length]}"></span>${k + 1}</td>
      <td>${fmt(alt, 0)}</td><td>${scored ? mine.length : '—'}</td>
      <td>${scored && mine.length && mine[0].los !== undefined ? sight : '—'}</td>
      <td>${scored ? `${fmt(got)} / ${fmt(asked)}` : '—'}</td>`;
  });
}

function show(sceneData, inst, answer) {
  const note = $('note'); note.style.display = 'none';
  const warn = text => { note.textContent = text; note.style.display = 'block'; };
  world.setScene(sceneData);
  state.inst = inst; state.answer = answer; state.epochs = epochsOf(inst, answer); state.e = 0;
  if (answer.instance !== inst.id) warn(`This answer is for ${answer.instance}, not for this instance.`);
  else if (!answer.result) warn('This answer has not been scored, so users are grey and no links are drawn. Run: python -m skycellbench view instance.json answer.json -o scored.json');
  const where = inst.scene ? `on ${inst.scene}` : 'on flat ground, statistical channel';
  $('instanceLine').textContent = `${inst.id} · task ${inst.task} · ${inst.n_uav} UAVs, ${inst.users.length} users · ${where}`;
  $('score').textContent = answer.result ? fmt(answer.result.score, 3) : '—';
  $('scoreNote').textContent = answer.result ? `score, ${answer.result.tier === 'ns3' ? 'ns-3' : 'closed-form'} scorer` : 'not scored';
  $('epochRow').style.display = state.epochs.length > 1 ? 'flex' : 'none';
  $('epoch').max = state.epochs.length - 1; $('epoch').value = 0;
  $('credit').innerHTML = `${sceneData.attribution ? sceneData.attribution.replace('OpenStreetMap contributors',
    '<a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap contributors</a>') + ' · ' : ''}
    Rendering adapted from <a href="https://github.com/DanMcInerney/DroneRTS" target="_blank" rel="noopener">DroneRTS</a> (MIT)`;
  showEpoch();
  $('status').style.display = 'none';
  window.viewerReady = true;                        // for automated checks of the page
}

async function loadUrls(instanceUrl, answerUrl, sceneUrl) {
  $('status').style.display = 'flex'; $('status').textContent = 'Loading…';
  try {
    const [inst, answer] = await Promise.all([getJson(instanceUrl), getJson(answerUrl)]);
    const sceneData = sceneUrl ? await getJson(sceneUrl)
      : inst.scene ? await getJson(`../scenes/${inst.scene}/scene.json`) : flatScene(inst.side_m);
    show(sceneData, inst, answer);
  } catch (error) {
    $('status').textContent = `Could not load: ${error.message}`;
  }
}

function applyToggles() {
  if (world.links) { world.links.los.visible = $('showLos').checked; world.links.nlos.visible = $('showNlos').checked; }
  if (world.buildings) { world.buildings.visible = $('showBuildings').checked; world.setGlass($('glass').checked); }
  world.rules.visible = $('showRules').checked;
}
for (const id of ['showLos', 'showNlos', 'showBuildings', 'glass', 'showRules']) $(id).addEventListener('change', applyToggles);
$('reset').addEventListener('click', () => world.resetView());
$('top').addEventListener('click', () => world.topView());
$('epoch').addEventListener('input', event => { state.e = Number(event.target.value); showEpoch(); });
$('play').addEventListener('click', () => {
  if (state.timer) { clearInterval(state.timer); state.timer = 0; $('play').textContent = 'Play'; return; }
  $('play').textContent = 'Pause';
  state.timer = setInterval(() => { state.e = (state.e + 1) % state.epochs.length; $('epoch').value = state.e; showEpoch(); }, 1100);
});

// hover a user for its numbers
const tip = $('tip');
$('view').addEventListener('pointermove', event => {
  if (event.buttons || !state.epochs.length) { tip.style.display = 'none'; return; }
  const i = world.pick(event.clientX, event.clientY);
  if (i < 0) { tip.style.display = 'none'; return; }
  const u = state.epochs[state.e].users[i];
  const got = u.thr === undefined ? 'not scored' : `gets ${fmt(u.thr, 2)} of ${fmt(u.demand, 1)} Mbit/s (${fmt(100 * Math.min(1, u.thr / u.demand), 0)}%)`;
  const from = u.serving === undefined ? '' : ` · UAV ${u.serving + 1}${u.los === undefined ? '' : u.los ? ', in sight' : ', not in sight'}`;
  tip.textContent = `user ${u.id + 1} · ${got}${from}`;
  tip.style.display = 'block'; tip.style.left = `${event.clientX + 14}px`; tip.style.top = `${event.clientY + 14}px`;
});
$('view').addEventListener('pointerleave', () => { tip.style.display = 'none'; });

// local files
const files = { inst: null, answer: null, scene: null };
async function filesChanged() {
  try {
    if ($('fInstance').files[0]) files.inst = await readFile($('fInstance'));
    if ($('fAnswer').files[0]) files.answer = await readFile($('fAnswer'));
    if ($('fScene').files[0]) files.scene = await readFile($('fScene'));
    if (!files.inst || !files.answer) return;
    const sceneData = files.scene ?? (files.inst.scene ? await getJson(`../scenes/${files.inst.scene}/scene.json`) : flatScene(files.inst.side_m));
    show(sceneData, files.inst, files.answer);
  } catch (error) {
    $('status').style.display = 'flex'; $('status').textContent = `Could not read the file: ${error.message}`;
  }
}
for (const id of ['fInstance', 'fAnswer', 'fScene']) $(id).addEventListener('change', filesChanged);

// start: files named in the address, or the list of examples
const query = new URLSearchParams(location.search);
if (query.get('instance') && query.get('answer')) {
  $('exampleBox').style.display = 'none';
  loadUrls(query.get('instance'), query.get('answer'), query.get('scene'));
} else {
  getJson('./examples/index.json').then(list => {
    const select = $('example');
    list.forEach((item, i) => select.add(new Option(item.label, String(i))));
    const open = () => { const item = list[Number(select.value)]; loadUrls(`./examples/${item.instance}`, `./examples/${item.answer}`); };
    select.addEventListener('change', open);
    const wanted = Number(query.get('example') ?? 0);                 // ?example=2 opens the third one
    select.value = String(Number.isInteger(wanted) && wanted >= 0 && wanted < list.length ? wanted : 0);
    open();
  }).catch(error => { $('status').textContent = `No examples found (${error.message}). Open files from the panel.`; });
}
