"use strict";
const SVGNS = "http://www.w3.org/2000/svg";
const SEND = 140;            // downsample the drawing to SEND x SEND before posting

// ---------------------------------------------------------------- canvas
const pad = document.getElementById("pad");
const ctx = pad.getContext("2d", { willReadFrequently: true });
let drawing = false, last = null, hasInk = false;

function clearPad() {
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, pad.width, pad.height);
  hasInk = false;
}
clearPad();

function posFromEvent(e) {
  const r = pad.getBoundingClientRect();
  const p = (e.touches && e.touches[0]) || e;
  return { x: (p.clientX - r.left) * pad.width / r.width,
           y: (p.clientY - r.top) * pad.height / r.height };
}
function strokeTo(p) {
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = pad.width * 0.085;     // ~24px on a 288 canvas
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(last.x, last.y);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
  last = p;
  hasInk = true;
}
function start(e) { e.preventDefault(); drawing = true; last = posFromEvent(e); strokeTo(last); }
function move(e)  { if (!drawing) return; e.preventDefault(); strokeTo(posFromEvent(e)); }
function end()    { if (!drawing) return; drawing = false; scheduleInfer(); }

pad.addEventListener("mousedown", start);
pad.addEventListener("mousemove", move);
window.addEventListener("mouseup", end);
pad.addEventListener("touchstart", start, { passive: false });
pad.addEventListener("touchmove", move, { passive: false });
pad.addEventListener("touchend", end);

document.getElementById("clear").onclick = () => { clearPad(); resetAll(); };
document.getElementById("predict").onclick = () => infer();

let timer = null;
function scheduleInfer() { clearTimeout(timer); timer = setTimeout(infer, 150); }

// ---------------------------------------------------------------- inference
function readGray(n) {
  const off = document.createElement("canvas");
  off.width = off.height = n;
  const o = off.getContext("2d");
  o.drawImage(pad, 0, 0, n, n);
  const d = o.getImageData(0, 0, n, n).data;
  const rows = [];
  for (let r = 0; r < n; r++) {
    const row = [];
    for (let c = 0; c < n; c++) row.push(d[(r * n + c) * 4] / 255);
    rows.push(row);
  }
  return rows;
}
function setStatus(t) { document.getElementById("status").textContent = t; }

async function infer() {
  if (!hasInk) { setStatus("draw a single digit (0–9)"); return; }
  setStatus("thinking…");
  try {
    const res = await fetch("/infer", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pixels: readGray(SEND), size: SEND }),
    });
    if (!res.ok) { setStatus("error " + res.status); return; }
    render(await res.json());
  } catch (err) { setStatus("offline? " + err.message); }
}

// ---------------------------------------------------------------- colour utils
function hex(h) { return [parseInt(h.slice(1,3),16), parseInt(h.slice(3,5),16), parseInt(h.slice(5,7),16)]; }
function lerp(a, b, t) {
  const A = hex(a), B = hex(b);
  const c = A.map((v, i) => Math.round(v + (B[i]-v)*t));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
const clamp01 = t => Math.max(0, Math.min(1, t));
const green = t => lerp("#0e2a17", "#56f06f", clamp01(t));   // one fire-green ramp, used everywhere

// ---------------------------------------------------------------- render
function render(d) {
  renderPrediction(d);
  renderInput(d.input_14x14);
  const strengths = filterStrengths(d.feature_maps);   // how strongly each filter fired
  renderFilters(d.filters, strengths);
  renderFeatureMaps(d.feature_maps, strengths);
  renderNetwork(d.graph, d.prediction);
  setModelInfo(d.model_meta);
}

// per-filter response on this drawing = total pooled activation, normalised to [0,1]
function filterStrengths(maps) {
  const s = new Map();
  let mx = 1e-9;
  for (const m of maps) {
    let t = 0;
    for (const row of m.map) for (const v of row) t += v;
    s.set(m.slot, t);
    mx = Math.max(mx, t);
  }
  for (const [k, v] of s) s.set(k, v / mx);
  return s;
}

function clearInputGrid() {
  const cv = document.getElementById("inputGrid");
  const g = cv.getContext("2d");
  g.fillStyle = "#000";
  g.fillRect(0, 0, cv.width, cv.height);
}

function resetAll() {
  document.getElementById("bigDigit").textContent = "–";
  document.getElementById("probs").innerHTML = "";
  document.getElementById("featureMaps").innerHTML = "";
  clearInputGrid();
  restNetwork();                 // back to the at-rest view: all wires, nothing fired
  setStatus("draw a single digit (0–9)");
}

function renderPrediction(d) {
  document.getElementById("bigDigit").textContent = d.prediction;
  const box = document.getElementById("probs");
  box.innerHTML = "";
  const max = Math.max(...d.probs);
  d.probs.forEach((p, i) => {
    const col = document.createElement("div");
    col.className = "prob" + (i === d.prediction ? " win" : "");
    col.title = `${i}: ${(p*100).toFixed(1)}%`;
    const b = document.createElement("div");
    b.className = "b";
    b.style.height = `${Math.max(2, (p / (max || 1)) * 100)}%`;
    const l = document.createElement("div");
    l.className = "l";
    l.textContent = i;
    col.append(b, l);
    box.append(col);
  });
  setStatus(`predicted ${d.prediction} · ${(d.probs[d.prediction]*100).toFixed(0)}% confident`);
}

function renderInput(img) {
  const cv = document.getElementById("inputGrid");
  const g = cv.getContext("2d");
  const n = img.length, cell = cv.width / n;
  let lo = Infinity, hi = -Infinity;
  for (const row of img) for (const v of row) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  const span = hi - lo || 1;
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
    const t = clamp01((img[r][c] - lo) / span);
    const s = Math.round(t * 255);
    g.fillStyle = `rgb(${s},${s},${s})`;
    g.fillRect(c * cell, r * cell, cell, cell);
  }
}

function renderFilters(filters, strengths) {
  const bank = document.getElementById("filters");
  bank.innerHTML = "";
  filters.forEach(f => {
    const el = document.createElement("div");
    el.className = "filter" + (f.active ? "" : " dim");
    if (f.active) {
      const flat = f.kernel.flat();
      const maxabs = Math.max(1e-9, ...flat.map(Math.abs));
      flat.forEach(v => {
        const c = document.createElement("div");
        c.className = "c";
        c.style.background = green(Math.abs(v) / maxabs);     // tap strength, one green
        el.append(c);
      });
      if (strengths) {                                        // emphasise filters that fired
        const t = strengths.get(f.slot) || 0;
        el.style.opacity = (0.22 + 0.78 * t).toFixed(3);
        if (t > 0.55) el.classList.add("fired");
      }
    } else {
      for (let i = 0; i < 9; i++) { const c = document.createElement("div"); c.className = "c"; el.append(c); }
    }
    bank.append(el);
  });
  const n = filters.filter(f => f.active).length;
  document.getElementById("filterCap").innerHTML = strengths
    ? `<strong>${n} of ${filters.length}</strong> slots active. Brightly-lit filters fired strongly on this drawing; dim ones barely responded.`
    : `<strong>${n} of ${filters.length}</strong> slots active — the economy pruned the rest.`;
}

function renderFeatureMaps(maps, strengths) {
  const bank = document.getElementById("featureMaps");
  bank.innerHTML = "";
  let max = 1e-9;
  for (const m of maps) for (const row of m.map) for (const v of row) max = Math.max(max, v);
  maps.forEach(m => {
    const el = document.createElement("div");
    el.className = "fmap";
    for (const row of m.map) for (const v of row) {
      const c = document.createElement("div");
      c.className = "c";
      c.style.background = green(v / max);
      el.append(c);
    }
    if (strengths) {                                          // dim maps that barely fired
      const t = strengths.get(m.slot) || 0;
      el.style.opacity = (0.3 + 0.7 * t).toFixed(3);
      if (t > 0.55) el.classList.add("fired");
    }
    bank.append(el);
  });
}

// ---- network: build geometry + edges once, restyle by activation each inference
const net = document.getElementById("net");
let layout = null, edgeEls = null, nodeEls = null;

function buildNetwork(graph) {
  net.innerHTML = "";
  const nLayers = graph.n_layers;
  const byLayer = {};
  graph.neurons.forEach(n => (byLayer[n.layer] = byLayer[n.layer] || []).push(n));
  Object.values(byLayer).forEach(arr => arr.sort((a, b) => a.id - b.id));

  layout = new Map();
  const radius = L => (L === 0 ? 0.55 : L === nLayers - 1 ? 2.6 : 1.4);
  Object.entries(byLayer).forEach(([L, arr]) => {
    const x = nLayers > 1 ? 8 + 84 * (L / (nLayers - 1)) : 50;
    arr.forEach((n, i) => layout.set(n.id, {
      x, y: 5 + 90 * ((i + 0.5) / arr.length), r: radius(+L), layer: +L,
    }));
  });

  // edges first (drawn under nodes); colour by confidence is fixed (model is fixed)
  const efrag = document.createDocumentFragment();
  edgeEls = graph.synapses.map(s => {
    const a = layout.get(s.pre), b = layout.get(s.post);
    const ln = document.createElementNS(SVGNS, "line");
    ln.setAttribute("x1", a.x); ln.setAttribute("y1", a.y);
    ln.setAttribute("x2", b.x); ln.setAttribute("y2", b.y);
    ln.dataset.pre = s.pre; ln.dataset.post = s.post;
    ln._w = Math.abs(s.w); ln._sw = s.w;     // |weight| for rest, signed for flow
    efrag.append(ln);
    return ln;
  });
  net.append(efrag);

  // nodes
  const nfrag = document.createDocumentFragment();
  nodeEls = new Map();
  graph.neurons.forEach(n => {
    const p = layout.get(n.id);
    const cir = document.createElementNS(SVGNS, "circle");
    cir.setAttribute("cx", p.x); cir.setAttribute("cy", p.y); cir.setAttribute("r", p.r);
    cir.setAttribute("stroke", "#0a0d12"); cir.setAttribute("stroke-width", 0.15);
    nfrag.append(cir);
    nodeEls.set(n.id, cir);
  });
  net.append(nfrag);

  // digit labels on the output layer
  const out = byLayer[nLayers - 1];
  out.forEach((n, i) => {
    const p = layout.get(n.id);
    const t = document.createElementNS(SVGNS, "text");
    t.setAttribute("x", p.x + 4); t.setAttribute("y", p.y + 1.3);
    t.setAttribute("font-size", 3); t.setAttribute("fill", "#8b949e");
    t.textContent = i;
    net.append(t);
  });
}

// at-rest view: every neuron silent, every wire shown uniformly (thickness =
// weight), no activation highlighting — a clean look at the full topology.
function restNetwork() {
  if (!nodeEls) return;
  nodeEls.forEach(cir => {
    cir.setAttribute("fill", "#262c36");
    cir.setAttribute("stroke", "#0a0d12");
    cir.setAttribute("stroke-width", 0.15);
  });
  edgeEls.forEach(ln => {
    ln.setAttribute("stroke", "#3a4250");
    ln.setAttribute("stroke-opacity", 0.32);
    ln.setAttribute("stroke-width", 0.1 + Math.min(0.3, ln._w * 0.3));
  });
}

function renderNetwork(graph, prediction) {
  if (!nodeEls || nodeEls.size !== graph.neurons.length) buildNetwork(graph);

  // per-layer max activation, for node brightness normalisation
  const act = new Map(), maxByLayer = {};
  graph.neurons.forEach(n => {
    act.set(n.id, n.act);
    if (n.act > (maxByLayer[n.layer] || 0)) maxByLayer[n.layer] = n.act;
  });

  graph.neurons.forEach(n => {
    const cir = nodeEls.get(n.id);
    const m = maxByLayer[n.layer] || 1;
    cir.setAttribute("fill", n.act > 1e-9 ? green(n.act / m) : "#262c36");
  });

  // wire encoding = SIGNAL STRENGTH this drawing: |flow| = |pre-activation x weight|.
  // Strong wires are thick + bright green; weak ones fade; inactive ones recede.
  let maxFlow = 1e-9;
  const flow = edgeEls.map(ln => {
    const a = act.get(+ln.dataset.pre), b = act.get(+ln.dataset.post);
    const f = (a > 1e-9 && b > 1e-9) ? Math.abs(a * ln._sw) : 0;
    if (f > maxFlow) maxFlow = f;
    return f;
  });
  edgeEls.forEach((ln, i) => {
    const t = flow[i] / maxFlow;                  // 0..1 relative strength
    if (flow[i] > 0) {
      ln.setAttribute("stroke", green(t));
      ln.setAttribute("stroke-opacity", 0.10 + 0.88 * t);
      ln.setAttribute("stroke-width", 0.09 + 0.95 * t);
    } else {
      ln.setAttribute("stroke", "#262c36");
      ln.setAttribute("stroke-opacity", 0.03);    // inactive wires recede
      ln.setAttribute("stroke-width", 0.08);
    }
  });

  // ring the predicted digit (white = the answer, stands out from the green)
  const outLayer = graph.n_layers - 1;
  graph.neurons.filter(n => n.layer === outLayer).forEach((n, i) => {
    const cir = nodeEls.get(n.id);
    if (i === prediction) {
      cir.setAttribute("stroke", "#ffffff"); cir.setAttribute("stroke-width", 0.7);
    } else {
      cir.setAttribute("stroke", "#0a0d12"); cir.setAttribute("stroke-width", 0.15);
    }
  });
}

function setModelInfo(m) {
  if (!m) return;
  document.getElementById("modelInfo").textContent =
    `model: ${m.n_active_filters} active filters · test acc ${(m.test_acc*100).toFixed(1)}%`;
}

// on load: render the learned filters + the full network at rest, so the page
// shows the topology before any drawing (and Clear returns to exactly this).
async function loadResting() {
  try {
    const r = await fetch("/graph");
    if (!r.ok) return;
    const d = await r.json();
    buildNetwork(d.graph);
    restNetwork();
    renderFilters(d.filters);
    setModelInfo(d.model_meta);
  } catch (err) { /* server not up yet */ }
}
loadResting();
