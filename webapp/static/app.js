"use strict";
const SVGNS = "http://www.w3.org/2000/svg";
const SIZE = 320;                       // board internal resolution
const A_RGB = [34, 211, 238], B_RGB = [251, 146, 60];

// ----------------------------------------------------------------- state
let points = [];                        // {x, y, c} in model coords [-1, 1]
let selClass = 0, netSize = "medium";
let running = false, started = false, busy = false;
let stepsPerTick = 30, gridRes = 40;
let lastBoundary = null;

// ----------------------------------------------------------------- board
const board = document.getElementById("board");
const bctx = board.getContext("2d");
const off = document.createElement("canvas");        // offscreen for the heatmap

const toCanvas = (x, y) => [(x + 1) / 2 * SIZE, (1 - y) / 2 * SIZE];
const toModel = (px, py) => [px / SIZE * 2 - 1, 1 - py / SIZE * 2];

function boardXY(e) {
  const r = board.getBoundingClientRect();
  const p = (e.touches && e.touches[0]) || e;
  return [(p.clientX - r.left) * SIZE / r.width, (p.clientY - r.top) * SIZE / r.height];
}

function drawBoard() {
  bctx.fillStyle = "#0a0d12";
  bctx.fillRect(0, 0, SIZE, SIZE);
  if (lastBoundary) {
    const res = lastBoundary.length;
    off.width = off.height = res;
    const img = off.getContext("2d").createImageData(res, res);
    for (let r = 0; r < res; r++) for (let c = 0; c < res; c++) {
      const p = lastBoundary[r][c], i = (r * res + c) * 4;
      img.data[i]     = A_RGB[0] + (B_RGB[0] - A_RGB[0]) * p;
      img.data[i + 1] = A_RGB[1] + (B_RGB[1] - A_RGB[1]) * p;
      img.data[i + 2] = A_RGB[2] + (B_RGB[2] - A_RGB[2]) * p;
      img.data[i + 3] = 120;
    }
    off.getContext("2d").putImageData(img, 0, 0);
    bctx.imageSmoothingEnabled = true;
    bctx.drawImage(off, 0, 0, res, res, 0, 0, SIZE, SIZE);
  }
  for (const pt of points) {
    const [px, py] = toCanvas(pt.x, pt.y);
    bctx.beginPath();
    bctx.arc(px, py, 6, 0, 2 * Math.PI);
    bctx.fillStyle = pt.c === 0 ? "#22d3ee" : "#fb923c";
    bctx.fill();
    bctx.lineWidth = 1.5; bctx.strokeStyle = "rgba(255,255,255,.85)"; bctx.stroke();
  }
}

let painting = false, lastPaint = null;
function addDot(px, py) {
  if (lastPaint && Math.hypot(px - lastPaint[0], py - lastPaint[1]) < 16) return;
  const [x, y] = toModel(px, py);
  points.push({ x, y, c: selClass });
  lastPaint = [px, py];
  markDirty();
  drawBoard();
}
board.addEventListener("pointerdown", e => { e.preventDefault(); painting = true; lastPaint = null; addDot(...boardXY(e)); });
board.addEventListener("pointermove", e => { if (painting) { e.preventDefault(); addDot(...boardXY(e)); } });
window.addEventListener("pointerup", () => { painting = false; });

// ----------------------------------------------------------------- API
async function postJSON(url, body) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || ("HTTP " + r.status));
  return r.json();
}

function payload(extra = {}) {
  return { points: points.map(p => [p.x, p.y]), labels: points.map(p => p.c),
           size: netSize, grid_res: gridRes, ...extra };
}

// ----------------------------------------------------------------- training loop
function setRunUI() { document.getElementById("run").textContent = running ? "⏸ Pause" : "▶ Run"; }

async function ensureStarted() {
  if (started) return true;
  if (points.length < 2 || !points.some(p => p.c === 0) || !points.some(p => p.c === 1)) {
    log("⚠ place dots of BOTH classes first"); return false;
  }
  applySnapshot(await postJSON("/train/start", payload({ seed: 0 })));
  started = true;
  log(`▶ started — ${netSize} network on ${points.length} dots`);
  return true;
}

async function loop() {
  if (!running) return;
  if (busy) { requestAnimationFrame(loop); return; }
  busy = true;
  try { applySnapshot(await postJSON("/train/step", { n: stepsPerTick, grid_res: gridRes })); }
  catch (e) { log("✕ " + e.message); running = false; setRunUI(); }
  busy = false;
  if (running) requestAnimationFrame(loop);
}

async function onRun() {
  if (running) { running = false; setRunUI(); return; }
  if (!await ensureStarted()) return;
  running = true; setRunUI(); loop();
}
async function onStep() {
  if (running || !await ensureStarted()) return;
  applySnapshot(await postJSON("/train/step", { n: stepsPerTick, grid_res: gridRes }));
}
async function onRestart() {
  if (!started) return;
  applySnapshot(await postJSON("/train/restart", { grid_res: gridRes }));
  log("⟲ restarted with a fresh random wiring");
}

function markDirty() {            // editing the data invalidates the running session
  started = false;
  if (running) { running = false; setRunUI(); }
}

// ----------------------------------------------------------------- snapshot -> UI
function applySnapshot(s) {
  lastBoundary = s.boundary;
  drawBoard();
  renderGraph(s.graph);
  document.getElementById("mStep").textContent = s.step;
  document.getElementById("mAcc").textContent = (s.accuracy * 100).toFixed(0) + "%";
  document.getElementById("mLoss").textContent = s.loss == null ? "–" : s.loss.toFixed(3);
  document.getElementById("mSyn").textContent = s.synapses;
  const ph = document.getElementById("mPhase");
  ph.textContent = s.phase;
  ph.className = "mv phase-" + s.phase;
  logEvents(s);
}

// --- network graph (rebuilt each frame; topology changes as it prunes/grows) ---
const net = document.getElementById("net");
const clamp01 = t => Math.max(0, Math.min(1, t));
function hex(h){return [parseInt(h.slice(1,3),16),parseInt(h.slice(3,5),16),parseInt(h.slice(5,7),16)];}
function lerp(a, b, t){const A=hex(a),B=hex(b);const c=A.map((v,i)=>Math.round(v+(B[i]-v)*t));return `rgb(${c[0]},${c[1]},${c[2]})`;}
const confColor = c => lerp("#3b82f6", "#ef4444", clamp01(c / 3));   // plastic -> frozen

function renderGraph(g) {
  net.innerHTML = "";
  const nL = g.n_layers, byL = {};
  g.neurons.forEach(n => (byL[n.layer] = byL[n.layer] || []).push(n));
  Object.values(byL).forEach(a => a.sort((p, q) => p.id - q.id));
  const pos = new Map(), radius = L => (L === 0 || L === nL - 1 ? 2.4 : 1.7);
  Object.entries(byL).forEach(([L, arr]) => {
    const x = nL > 1 ? 8 + 84 * (+L / (nL - 1)) : 50;
    arr.forEach((n, i) => pos.set(n.id, { x, y: 6 + 88 * ((i + 0.5) / arr.length), r: radius(+L) }));
  });
  let maxRate = 1e-6;
  g.neurons.forEach(n => { if (n.rate > maxRate) maxRate = n.rate; });

  const ef = document.createDocumentFragment();
  g.synapses.forEach(s => {
    const a = pos.get(s.pre), b = pos.get(s.post);
    const ln = document.createElementNS(SVGNS, "line");
    ln.setAttribute("x1", a.x); ln.setAttribute("y1", a.y);
    ln.setAttribute("x2", b.x); ln.setAttribute("y2", b.y);
    ln.setAttribute("stroke", confColor(s.conf));
    ln.setAttribute("stroke-opacity", 0.85);
    ln.setAttribute("stroke-width", 0.15 + Math.min(1.3, Math.abs(s.w) * 0.8));
    ef.append(ln);
  });
  net.append(ef);

  const last = nL - 1;
  g.neurons.forEach(n => {
    const p = pos.get(n.id), t = clamp01(n.rate / maxRate);
    const cir = document.createElementNS(SVGNS, "circle");
    cir.setAttribute("cx", p.x); cir.setAttribute("cy", p.y); cir.setAttribute("r", p.r);
    let fill = lerp("#2b313b", "#e6edf3", t);
    if (n.layer === last) fill = lerp("#1c2a2e", n.id % 2 === 0 ? "#22d3ee" : "#fb923c", 0.35 + 0.65 * t);
    cir.setAttribute("fill", fill);
    cir.setAttribute("stroke", "#0a0d12"); cir.setAttribute("stroke-width", 0.2);
    net.append(cir);
  });

  // labels: inputs x / y, outputs A / B
  const lab = (txt, p, dx) => {
    const t = document.createElementNS(SVGNS, "text");
    t.setAttribute("x", p.x + dx); t.setAttribute("y", p.y + 1.1);
    t.setAttribute("font-size", 3.2); t.setAttribute("fill", "#8b949e");
    t.setAttribute("text-anchor", "middle"); t.textContent = txt; net.append(t);
  };
  byL[0].forEach((n, i) => lab(i === 0 ? "x" : "y", pos.get(n.id), -4.5));
  byL[last].forEach((n, i) => lab(i === 0 ? "A" : "B", pos.get(n.id), 4.5));
}

// ----------------------------------------------------------------- log
function log(msg) {
  const el = document.getElementById("log");
  const line = document.createElement("div");
  line.className = "ev"; line.textContent = msg;
  el.prepend(line);
  while (el.childNodes.length > 40) el.lastChild.remove();
}
function logEvents(s) {
  const e = s.events || {};
  if (e.startle) log(`⚡ step ${s.step} — STARTLE: grew ${e.grow || 0} wires on a loss spike`);
  else if (e.sleep) log(`💤 step ${s.step} — sleep: pruned ${e.prune || 0}, grew ${e.grow || 0}`);
}

// ----------------------------------------------------------------- presets
function setPreset(kind) {
  const out = [], rng = mulberry(42);
  const push = (x, y, c) => out.push({ x: Math.max(-1, Math.min(1, x)), y: Math.max(-1, Math.min(1, y)), c });
  if (kind === "blobs") {
    for (let i = 0; i < 30; i++) push(-0.5 + rng() * .3 - .15, -0.3 + rng() * .3 - .15, 0);
    for (let i = 0; i < 30; i++) push(0.5 + rng() * .3 - .15, 0.3 + rng() * .3 - .15, 1);
  } else if (kind === "xor") {
    const q = [[-.5, -.5, 0], [.5, .5, 0], [-.5, .5, 1], [.5, -.5, 1]];
    q.forEach(([cx, cy, c]) => { for (let i = 0; i < 16; i++) push(cx + rng() * .3 - .15, cy + rng() * .3 - .15, c); });
  } else if (kind === "circles") {
    for (let i = 0; i < 36; i++) { const a = rng() * 6.28; push(0.28 * Math.cos(a) + (rng() - .5) * .1, 0.28 * Math.sin(a) + (rng() - .5) * .1, 0); }
    for (let i = 0; i < 44; i++) { const a = rng() * 6.28; push(0.78 * Math.cos(a) + (rng() - .5) * .1, 0.78 * Math.sin(a) + (rng() - .5) * .1, 1); }
  } else if (kind === "spiral") {
    for (let c = 0; c < 2; c++) for (let i = 0; i < 45; i++) {
      const t = i / 45, r = 0.12 + 0.86 * t, th = t * 6.28 * 1.1 + c * Math.PI + (rng() - .5) * .25;
      push(r * Math.cos(th), r * Math.sin(th), c);
    }
  }
  points = out;
  markDirty(); lastBoundary = null; drawBoard();
  log(`loaded "${kind}" — ${out.length} dots`);
}
function mulberry(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

// ----------------------------------------------------------------- wiring
document.getElementById("classA").onclick = () => setClass(0);
document.getElementById("classB").onclick = () => setClass(1);
function setClass(c) {
  selClass = c;
  document.getElementById("classA").classList.toggle("on", c === 0);
  document.getElementById("classB").classList.toggle("on", c === 1);
}
document.getElementById("clear").onclick = () => { points = []; lastBoundary = null; markDirty(); drawBoard(); log("cleared dots"); };
document.querySelectorAll(".preset").forEach(b => b.onclick = () => setPreset(b.dataset.preset));
document.querySelectorAll(".size").forEach(b => b.onclick = () => {
  netSize = b.dataset.size; markDirty();
  document.querySelectorAll(".size").forEach(x => x.classList.toggle("on", x === b));
});
document.getElementById("run").onclick = onRun;
document.getElementById("step").onclick = onStep;
document.getElementById("restart").onclick = onRestart;
const speed = document.getElementById("speed");
speed.oninput = () => { stepsPerTick = +speed.value; document.getElementById("speedVal").textContent = `${stepsPerTick} steps/frame`; };

// start with a demo dataset so the board isn't empty
setPreset("spiral");
drawBoard();
