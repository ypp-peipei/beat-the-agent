// Beat the Agent — game core: feeder model, one-day environment, tabular Q-learning.
// Pure functions (no DOM) so the same file runs in node for testing.
// Feeder: single-phase equivalent radial feeder, per unit on 1 MVA, backward-forward sweep (same model as the talk's demo).
(function (root) {
  const FEEDER = {
    nBus: 6,
    r: [0, 0.008, 0.009, 0.010, 0.026, 0.034],
    x: [0, 0.006, 0.00675, 0.0075, 0.0195, 0.0255],
    loadShare: [0, 0.30, 0.25, 0.25, 0.12, 0.08],
    pvShare: [0, 0, 0.20, 0, 0.30, 0.50],
    bessBus: [3, 5],
  };
  const CFG = {
    steps: 96, dtH: 0.25,
    peakLoad: 0.80, loadPF: 0.95, pvPeak: 1.20,      // pu
    evLoad: 0.30,                                     // pu, evening EV charging at the end of the line
    bessP: 0.15, bessE: 0.60, soc0: 0.25,             // pu per battery, pu·h per battery
    vMin: 0.95, vMax: 1.05,
    penStep: 15, penMag: 1500,                          // $ per violating step, $ per pu of excess per step
  };
  function gauss(t, mu, sig) { return Math.exp(-0.5 * ((t - mu) / sig) ** 2); }
  function loadShape(t) { return Math.min(1, 0.28 + 0.20 * gauss(t, 7.8, 1.2) + 0.72 * gauss(t, 19.0, 2.0) + 0.04 * gauss(t, 13, 3)); }
  function pvShape(t) { const h = (t - 12.5) / 6.5; return Math.abs(h) >= 1 ? 0 : Math.pow(Math.cos(Math.PI * h / 2), 1.3); }
  function price(t) { return (t >= 16 && t < 21) ? 0.35 : (t >= 7 && t < 16) ? 0.18 : 0.10; }

  // small seeded RNG so a "day" (cloud pattern) can be replayed exactly
  function rng(seed) { let s = seed >>> 0; return function () { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  // a day = PV multiplier per step (passing clouds) and a load multiplier
  function makeDay(seed) {
    const R = rng(seed), cloud = new Float64Array(CFG.steps).fill(1);
    const n = 1 + Math.floor(R() * 3);
    for (let c = 0; c < n; c++) {
      const start = 36 + Math.floor(R() * 24), len = 2 + Math.floor(R() * 4), depth = 0.25 + 0.35 * R();
      for (let k = start; k < Math.min(CFG.steps, start + len); k++) cloud[k] = Math.min(cloud[k], 1 - depth);
    }
    return { seed, cloud, loadMul: 0.94 + 0.12 * R(), pvMul: 0.93 + 0.14 * R() };
  }

  // backward-forward sweep on real/imag pairs; returns bus voltage magnitudes and substation active power
  function solve(P, Q) {
    const n = FEEDER.nBus, vr = new Float64Array(n).fill(1), vi = new Float64Array(n), jr = new Float64Array(n), ji = new Float64Array(n);
    for (let it = 0; it < 8; it++) {
      for (let i = n - 1; i >= 1; i--) {
        const d = vr[i] * vr[i] + vi[i] * vi[i];
        // I = conj(S / V)
        const ir = (P[i] * vr[i] + Q[i] * vi[i]) / d, ii = (P[i] * vi[i] - Q[i] * vr[i]) / d;
        jr[i] = ir + (i + 1 < n ? jr[i + 1] : 0); ji[i] = ii + (i + 1 < n ? ji[i + 1] : 0);
      }
      for (let i = 1; i < n; i++) {
        vr[i] = vr[i - 1] - (FEEDER.r[i] * jr[i] - FEEDER.x[i] * ji[i]);
        vi[i] = vi[i - 1] - (FEEDER.r[i] * ji[i] + FEEDER.x[i] * jr[i]);
      }
    }
    const V = []; for (let i = 0; i < n; i++) V.push(Math.hypot(vr[i], vi[i]));
    return { V, gridP: jr[1] };       // V0 = 1∠0, so P into the feeder = Re(J1)
  }

  // ---- environment: one day, 96 quarter-hour steps. action: 0 = charge, 1 = idle, 2 = discharge (both batteries together)
  function reset(day) { return { k: 0, soc: CFG.soc0, day, bill: 0, pen: 0, violSteps: 0, V: null, done: false }; }
  function flows(s, a) {
    const t = s.k * CFG.dtH, d = s.day;
    const L = loadShape(t) * CFG.peakLoad * d.loadMul, pv = pvShape(t) * CFG.pvPeak * d.pvMul * d.cloud[s.k];
    let pb = a === 0 ? CFG.bessP : a === 2 ? -CFG.bessP : 0;                      // + charge
    if (pb > 0) pb = Math.min(pb, (1 - s.soc) * CFG.bessE / CFG.dtH);
    if (pb < 0) pb = -Math.min(-pb, s.soc * CFG.bessE / CFG.dtH);
    const qf = Math.tan(Math.acos(CFG.loadPF)), P = [], Q = [];
    for (let i = 0; i < FEEDER.nBus; i++) {
      let p = L * FEEDER.loadShare[i], q = p * qf;
      p -= pv * FEEDER.pvShare[i];
      if ((i === 4 || i === 5) && t >= 18 && t < 22.5) { p += CFG.evLoad / 2; q += CFG.evLoad / 2 * 0.1; }
      if (FEEDER.bessBus.indexOf(i) >= 0) p += pb;
      P.push(p); Q.push(q);
    }
    return { P, Q, pb, t, L, pv };
  }
  // w = reward weights {cost, volt}; the SCORE always uses both, the agent's reward may not
  function step(s, a, w) {
    const f = flows(s, a), r = solve(f.P, f.Q);
    const cost = r.gridP * 1000 * CFG.dtH * price(f.t);                           // $ this step (export is credited)
    // what the same step would have cost with the battery idle: the reward is the money the battery saved,
    // which removes the day-to-day weather noise from the learning signal
    const costIdle = f.pb === 0 ? cost : (() => { const g = flows(s, 1); return solve(g.P, g.Q).gridP * 1000 * CFG.dtH * price(f.t); })();
    let exc = 0;
    for (let i = 1; i < FEEDER.nBus; i++) exc = Math.max(exc, r.V[i] - CFG.vMax, CFG.vMin - r.V[i]);
    const pen = exc > 0 ? CFG.penStep + CFG.penMag * exc : 0;
    s.soc = Math.max(0, Math.min(1, s.soc + f.pb * CFG.dtH / CFG.bessE));
    s.bill += cost; s.pen += pen; if (exc > 0) s.violSteps++;
    s.V = r.V; s.vHi = Math.max.apply(null, r.V.slice(1)); s.vLo = Math.min.apply(null, r.V.slice(1));
    s.pb = f.pb; s.pv = f.pv; s.L = f.L; s.gridP = r.gridP; s.t = f.t;
    s.k++; s.done = s.k >= CFG.steps;
    const ww = w || { cost: 1, volt: 1 };
    return -(ww.cost * (cost - costIdle) + ww.volt * pen);
  }
  // run a whole day with a policy(s) -> action; returns the trace
  function runDay(day, policy, w) {
    const s = reset(day), tr = { vHi: [], vLo: [], soc: [], act: [], V: [] };
    let ret = 0;
    while (!s.done) { const a = policy(s); ret += step(s, a, w); tr.vHi.push(s.vHi); tr.vLo.push(s.vLo); tr.soc.push(s.soc); tr.act.push(a); tr.V.push(s.V); }
    tr.bill = s.bill; tr.pen = s.pen; tr.violSteps = s.violSteps; tr.total = s.bill + s.pen; tr.ret = ret;
    return tr;
  }
  const idle = () => 1;
  // score = dollars saved against doing nothing on the same day (bill + violation penalties)
  function score(day, tr) { return runDay(day, idle).total - tr.total; }

  // ---- tabular Q-learning. state = (step, state of charge bin, voltage zone seen after the previous step)
  const NSOC = 17, NV = 4, NA = 3;
  function vZone(s) {
    if (s.V === null) return 1;
    if (s.vHi > CFG.vMax) return 3; if (s.vHi > 1.035) return 2; if (s.vLo < CFG.vMin + 0.01) return 0; return 1;
  }
  function sIdx(s) { return (s.k * NSOC + Math.min(NSOC - 1, Math.round(s.soc * (NSOC - 1)))) * NV + vZone(s); }
  function makeAgent(seed, w) {
    return { Q: new Float64Array(CFG.steps * NSOC * NV * NA), R: rng(seed || 1), ep: 0, w: w || { cost: 1, volt: 1 }, alpha: 0.3, alphaMin: 0.01, alphaDecay: 10000, gamma: 1.0, eps0: 1.0, eps1: 0.02, epsDecay: 10000 };
  }
  function greedy(ag, s) { const b = sIdx(s) * NA, Q = ag.Q; let a = 1; if (Q[b] > Q[b + a]) a = 0; if (Q[b + 2] > Q[b + a]) a = 2; return a; }
  function epsilon(ag) { return ag.eps1 + (ag.eps0 - ag.eps1) * Math.exp(-ag.ep / ag.epsDecay); }
  // one training episode on a random day: act epsilon-greedily, then replay the day's transitions backwards
  // (reverse experience replay), so a late reward reaches the early decisions in the same episode
  function trainEpisode(ag, day) {
    const eps = epsilon(ag), Q = ag.Q, s = reset(day || makeDay(Math.floor(ag.R() * 1e9)));
    const tr = { vHi: [], vLo: [], soc: [], act: [] }, n = CFG.steps;
    const si = ag._si || (ag._si = new Int32Array(n + 1)), ac = ag._ac || (ag._ac = new Int8Array(n)), rw = ag._rw || (ag._rw = new Float64Array(n));
    for (let k = 0; k < n; k++) {
      const i = sIdx(s), a = ag.R() < eps ? Math.floor(ag.R() * NA) : greedy(ag, s);
      si[k] = i; ac[k] = a; rw[k] = step(s, a, ag.w) / 10;                    // rewards scaled to O(1)
      tr.vHi.push(s.vHi); tr.vLo.push(s.vLo); tr.soc.push(s.soc); tr.act.push(a);
    }
    const alpha = Math.max(ag.alphaMin, ag.alpha / (1 + ag.ep / ag.alphaDecay));
    for (let k = n - 1; k >= 0; k--) {
      let target = rw[k];
      if (k < n - 1) { const j = si[k + 1] * NA; target += ag.gamma * Math.max(Q[j], Q[j + 1], Q[j + 2]); }
      const q = si[k] * NA + ac[k]; Q[q] += alpha * (target - Q[q]);
    }
    tr.bill = s.bill; tr.pen = s.pen; tr.violSteps = s.violSteps; tr.total = s.bill + s.pen;
    ag.ep++; return tr;
  }
  const TRAIN_EPISODES = 40000;
  const api = { TRAIN_EPISODES, FEEDER, CFG, loadShape, pvShape, price, makeDay, reset, step, runDay, idle, score, makeAgent, greedy, epsilon, trainEpisode, rng };
  if (typeof module !== 'undefined') module.exports = api; else root.Core = api;
})(typeof window !== 'undefined' ? window : globalThis);
