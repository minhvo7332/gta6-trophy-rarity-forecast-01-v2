// gta6-trophy-rarity-forecast-01 — v2
// "Logit-space CQR quantile fan v2: dual-clock walk-forward blend + velocity signal"
//
// Forecast, per trophy, the Exophase earn rate at the case target, as five
// logit-space quantiles. Deterministic, offline, Node stdlib only.
//
// How it works, in one paragraph. Everything is done in logit space, where
// a rate moves additively and rare trophies count as much as common ones.
// Three predictors look forward from a trophy's history: a damped trend of
// its logit rate; a split of that rate into earners over players tracked,
// with each side given its own damped line and recombined; and a velocity
// predictor, new in v2, that reads the earn rate off the *players* clock —
// logit rate regressed on log players tracked, evaluated at the players
// count forecast for the target — which captures dilution as the tracked
// population explodes. Each predictor is fitted on two clocks, the
// saturating ln(1+h) and plain hours, and a walk-forward backtest on the
// case's own history (every prefix predicts 24h ahead, because a 1-step
// backtest just rewards persistence) scores every predictor on both clocks.
// Inverse-error weights blend clocks, then predictors, then the blend with
// persistence, and the same backtest supplies a CQR-style spread and a
// clamped bias correction — both calibrated on the current regime only:
// backtest rows straddling the first-refresh regime break are left out of
// the bias and spread (the break is a one-off; the random-walk floor
// prices the remaining jump risk). All fits run through prefix-sum tables, so a fit
// on any prefix is O(1) and the backtest is linear in the number of reads.
// When the history is flat — launch night, before Exophase's first refresh —
// no trend exists to fit, so the forecast falls back to the refresh-shock
// prior from the rules: players tracked 638 -> 3,868, median earners growth
// 1.5x, with the spread carrying the earners-growth uncertainty. Nothing is
// ever fitted to the practice cases or to any other case: every number above
// comes from the history inside the case file being forecast, plus the two
// facts the rules state about the first refresh.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const IN_DIR = process.env.MAKEFUN_INPUT_DIR ?? "/input";
const OUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? "/output";

const TAUS = [0.05, 0.25, 0.5, 0.75, 0.95];
const ZS = [-1.645, -0.674, 0, 0.674, 1.645];
const EPOCH = Date.parse("2026-11-19T00:00:00Z"); // trend clocks run from here
const PLAYERS_BEFORE = 638; // tracked players on launch night (rules)
const PLAYERS_AFTER = 3868; // tracked players after the 04:30 refresh (rules)
const G_MEDIAN = 1.5; // median earners growth across the first refresh
const G_LOG_SD = 0.5; // log-sd of that growth (platinum ~1x, chapter-1 ~2.4x)
const REFRESH_SD = 0.25; // extra logit sd for refresh timing / rounding

// Tunables (chosen on the 24h walk-forward backtest + synthetic final-like
// cases, never on the practice answers).
const DAMP_TREND = 0.75;
const DAMP_EARNERS = 0.75;
const DAMP_PLAYERS = 0.5;
const FLOOR_MULT = 0.2; // random-walk spread floor, x sqrt(days)
const CORE_LO = 0.2; // clamp on the core-vs-persistence weight
const CORE_HI = 0.8;
const BIAS_MAX = 0.5; // clamp on the backtest bias correction
const MIN_SPREAD = 0.15;
const USE_VELOCITY = true; // third signal; false = trend + decomposition only
// Spread calibration uses the 90th percentile of the backtest errors, but
// only of the calm rows (see the jump split below): the one-off first
// refresh would otherwise dominate the 90th percentile. The random-walk
// floor prices the remaining jump risk.
const SPREAD_Q = 0.9;
const SPREAD_Z = 1.645; // normal quantile at SPREAD_Q, for |error|
// A single-step logit move this big marks a regime break (the first
// Exophase refresh). Backtest rows straddling it calibrate the *old*
// regime, so they are left out of the bias and spread (not the weights:
// which predictor handles a jump best is still informative).
const JUMP_MIN = 0.5;
const USE_JUMP_SPLIT = true;

const pctClamp = (x: number) => Math.min(99.99, Math.max(0.01, x));
const toLogit = (pct: number) => {
  const p = pctClamp(pct) / 100;
  return Math.log(p / (1 - p));
};
const fromLogit = (l: number) => 100 / (1 + Math.exp(-l));
const r4 = (x: number) => Math.round(x * 10000) / 10000;
const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));
const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / Math.max(1, v.length);

// ---- Prefix-sum least squares -------------------------------------------
// PrefixFit holds the running sums for one (x, y) pair, so the OLS line on
// the first k points — and a damped prediction at any x — costs O(1).
class PrefixFit {
  private sx: number[] = [0];
  private sy: number[] = [0];
  private sxx: number[] = [0];
  private sxy: number[] = [0];
  constructor(xs: number[], ys: number[]) {
    for (let i = 0; i < xs.length; i++) {
      this.sx.push(this.sx[i] + xs[i]);
      this.sy.push(this.sy[i] + ys[i]);
      this.sxx.push(this.sxx[i] + xs[i] * xs[i]);
      this.sxy.push(this.sxy[i] + xs[i] * ys[i]);
    }
  }
  /** Damped prediction at xPred from the line fitted on the first k points. */
  predict(k: number, xPred: number, damp: number): number {
    const n = k;
    const mx = this.sx[n] / n;
    const my = this.sy[n] / n;
    const sxx = this.sxx[n] - n * mx * mx;
    const sxy = this.sxy[n] - n * mx * my;
    if (sxx < 1e-12) return my;
    const b = sxy / sxx;
    return my + damp * b * (xPred - mx);
  }
}

// ---- One trophy's history, as clocks and series --------------------------
interface Trophy {
  id: number;
  lastRate: number;
  flat: boolean;
  hours: number[]; // hours since EPOCH at each read
  logit: number[];
  fits: Record<string, PrefixFit>;
}

function buildTrophy(hist: any[], idx: number, id: number): Trophy {
  const hours: number[] = [];
  const logit: number[] = [];
  const logE: number[] = [];
  const logP: number[] = [];
  let lastRate = 0;
  for (const read of hist) {
    const rate = read.rates[idx];
    if (rate === null || rate === undefined) continue;
    lastRate = rate;
    const h = Math.max(0, (Date.parse(read.at) - EPOCH) / 3600000);
    hours.push(h);
    logit.push(toLogit(rate));
    const players = Math.max(1, read.players_tracked || 1);
    logP.push(Math.log(players));
    logE.push(Math.log(Math.max(0.001, (rate / 100) * players)));
  }
  const lnH = hours.map((h) => Math.log1p(h));
  const fits: Record<string, PrefixFit> = {
    trendLn: new PrefixFit(lnH, logit),
    trendH: new PrefixFit(hours, logit),
    earnLn: new PrefixFit(lnH, logE),
    earnH: new PrefixFit(hours, logE),
    playLn: new PrefixFit(lnH, logP),
    playH: new PrefixFit(hours, logP),
    velo: new PrefixFit(logP, logit), // logit rate on the players clock
  };
  let flat = true;
  for (let i = 1; i < logit.length; i++) {
    if (Math.abs(logit[i] - logit[0]) > 1e-12) {
      flat = false;
      break;
    }
  }
  return { id, lastRate, flat, hours, logit, fits };
}

// ---- Predictors ----------------------------------------------------------
// Each predictor answers, for a prefix of length k and a target point,
// "what logit rate do you expect?" The target point is described by the two
// clocks (ln-hours, hours) plus the forecast log players at the target.
interface Aim {
  xLn: number;
  xH: number;
  logP: number;
}

type Predictor = (t: Trophy, k: number, aim: Aim) => number;

const trendLn: Predictor = (t, k, aim) => t.fits.trendLn.predict(k, aim.xLn, DAMP_TREND);
const trendH: Predictor = (t, k, aim) => t.fits.trendH.predict(k, aim.xH, DAMP_TREND);

function decomp(t: Trophy, k: number, aim: Aim, lnClock: boolean): number {
  const x = lnClock ? aim.xLn : aim.xH;
  const eFit = lnClock ? t.fits.earnLn : t.fits.earnH;
  const pFit = lnClock ? t.fits.playLn : t.fits.playH;
  const earners = Math.exp(eFit.predict(k, x, DAMP_EARNERS));
  const players = Math.exp(pFit.predict(k, x, DAMP_PLAYERS));
  return toLogit(pctClamp((100 * earners) / Math.max(1, players)));
}
const decompLn: Predictor = (t, k, aim) => decomp(t, k, aim, true);
const decompH: Predictor = (t, k, aim) => decomp(t, k, aim, false);

// Velocity: the rate read off the players clock. Its two variants differ in
// where the target players count comes from (ln-clock vs hours-clock fit).
function velocity(t: Trophy, k: number, aim: Aim, lnClock: boolean): number {
  const x = lnClock ? aim.xLn : aim.xH;
  const pFit = lnClock ? t.fits.playLn : t.fits.playH;
  const logP = pFit.predict(k, x, DAMP_PLAYERS);
  return t.fits.velo.predict(k, logP, DAMP_TREND);
}
const veloLn: Predictor = (t, k, aim) => velocity(t, k, aim, true);
const veloH: Predictor = (t, k, aim) => velocity(t, k, aim, false);

// Predictors grouped by signal; each signal has one entry per clock.
const SIGNALS: Predictor[][] = [
  [trendLn, trendH],
  [decompLn, decompH],
  ...(USE_VELOCITY ? [[veloLn, veloH] as Predictor[]] : []),
];

/** Aim point for a read index j (its own clocks and log players). */
function aimAt(t: Trophy, j: number): Aim {
  const h = t.hours[j];
  return { xLn: Math.log1p(Math.max(0, h)), xH: Math.max(0, h), logP: 0 };
}

/** Quantile of a sorted array, nearest-rank. */
function quantile(sorted: number[], q: number): number {
  return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
}

// ---- Walk-forward backtest ----------------------------------------------
// For every prefix k >= 3, predict the first read at least 24h after the
// prefix's last read. Returns the blend weights the errors imply, the bias,
// and the blended predictor's absolute errors (the CQR spread input).
interface Calibration {
  clockW: number[]; // per signal: weight on the ln-clock variant
  signalW: number[]; // weight of each signal in the core
  coreW: number; // weight of the core vs persistence
  bias: number;
  spreadErrs: number[]; // blended errors, jump-straddling rows excluded
}

function calibrate(t: Trophy): Calibration | null {
  const n = t.logit.length;
  if (n < 5) return null;
  // Regime break: the index of the largest single-step logit move, if big.
  let jumpAt = -1;
  let jumpSize = 0;
  for (let i = 1; i < n; i++) {
    const d = Math.abs(t.logit[i] - t.logit[i - 1]);
    if (d > jumpSize) {
      jumpSize = d;
      jumpAt = i;
    }
  }
  if (!USE_JUMP_SPLIT || jumpSize < JUMP_MIN) jumpAt = -1;
  // Rows of one backtest step: actual logit + every predictor's answer.
  // A row straddles the regime break when its prefix ends before the jump
  // and its target sits at or after it.
  const rows: { actual: number; persist: number; preds: number[][]; straddles: boolean }[] = [];
  for (let k = 3; k < n; k++) {
    let j = -1;
    for (let m = k; m < n; m++) {
      if (t.hours[m] >= t.hours[k - 1] + 24) {
        j = m;
        break;
      }
    }
    if (j < 0) continue;
    const aim = aimAt(t, j);
    rows.push({
      actual: t.logit[j],
      persist: t.logit[k - 1],
      preds: SIGNALS.map((pair) => pair.map((fn) => fn(t, k, aim))),
      straddles: jumpAt >= 0 && k <= jumpAt && j >= jumpAt,
    });
  }
  if (rows.length === 0) return null;
  const mae = (pick: (r: (typeof rows)[number]) => number) =>
    mean(rows.map((r) => Math.abs(r.actual - pick(r))));
  const errP = mae((r) => r.persist);
  // Per-signal clock weights from inverse errors, kept away from the edges.
  const clockW = SIGNALS.map((_, s) => {
    const eLn = mae((r) => r.preds[s][0]);
    const eH = mae((r) => r.preds[s][1]);
    return clamp(eH / (eLn + eH + 1e-9), 0.25, 0.75);
  });
  const sigErr = SIGNALS.map((_, s) =>
    clockW[s] * mae((r) => r.preds[s][0]) + (1 - clockW[s]) * mae((r) => r.preds[s][1]),
  );
  // Signal weights: inverse error, each kept in [0.1, 0.6], renormalised.
  const raw = sigErr.map((e) => 1 / (e + 1e-9));
  const signalW = raw.map((w) => clamp(w / mean(raw), 0.1 / (1 / 3), 0.6 / (1 / 3)));
  const sigSum = signalW.reduce((a, b) => a + b, 0);
  for (let s = 0; s < signalW.length; s++) signalW[s] /= sigSum;
  const core = (r: (typeof rows)[number]) =>
    signalW.reduce(
      (acc, w, s) => acc + w * (clockW[s] * r.preds[s][0] + (1 - clockW[s]) * r.preds[s][1]),
      0,
    );
  const errCore = mae(core);
  const coreW = clamp(errP / (errP + errCore + 1e-9), CORE_LO, CORE_HI);
  // Bias and spread calibrate to the current regime: rows straddling the
  // regime break are left out (fall back to all rows if too few remain).
  const calm = rows.filter((r) => !r.straddles);
  const calRows = calm.length >= 3 ? calm : rows;
  const signed: number[] = [];
  const spreadErrs: number[] = [];
  for (const r of calRows) {
    const pred = coreW * core(r) + (1 - coreW) * r.persist;
    spreadErrs.push(Math.abs(r.actual - pred));
    signed.push(r.actual - pred);
  }
  return { clockW, signalW, coreW, bias: clamp(mean(signed), -BIAS_MAX, BIAS_MAX), spreadErrs };
}

// ---- One trophy's forecast ----------------------------------------------
function forecast(t: Trophy, targetMs: number, horizonDays: number): number[] {
  if (t.lastRate >= 100) return [100, 100, 100, 100, 100];
  if (t.logit.length === 0) return [0.01, 0.01, 0.01, 0.01, 0.01];
  const lastL = t.logit[t.logit.length - 1];

  // Flat history: the refresh-shock prior. The median is the exact scaling
  // the rules imply — the rate times the earners growth over the players
  // growth — not a constant logit shift, so it stays right for common
  // trophies too. The spread is the earners-growth uncertainty plus
  // refresh-timing noise, growing slowly with the horizon.
  if (t.flat || t.logit.length < 3) {
    const scaled = pctClamp(t.lastRate * (G_MEDIAN * PLAYERS_BEFORE / PLAYERS_AFTER));
    const med = toLogit(scaled);
    const sig = Math.sqrt(G_LOG_SD * G_LOG_SD + REFRESH_SD * REFRESH_SD + 0.25 * horizonDays);
    return ZS.map((z) => r4(pctClamp(fromLogit(med + z * sig))));
  }

  const hTarget = Math.max(0, (targetMs - EPOCH) / 3600000);
  const aim: Aim = { xLn: Math.log1p(hTarget), xH: hTarget, logP: 0 };
  const n = t.logit.length;
  const cal = calibrate(t);
  let med: number;
  let spread: number;
  if (cal === null) {
    // Too little history for a backtest: an equal blend of the signals
    // against persistence, with the random-walk spread.
    const core = mean(SIGNALS.flatMap((pair) => pair.map((fn) => fn(t, n, aim))));
    med = 0.5 * core + 0.5 * lastL;
    spread = FLOOR_MULT * Math.sqrt(horizonDays) + 0.3;
  } else {
    const core = cal.signalW.reduce(
      (acc, w, s) =>
        acc +
        w * (cal.clockW[s] * SIGNALS[s][0](t, n, aim) + (1 - cal.clockW[s]) * SIGNALS[s][1](t, n, aim)),
      0,
    );
    med = cal.coreW * core + (1 - cal.coreW) * lastL + cal.bias;
    const sorted = [...cal.spreadErrs].sort((a, b) => a - b);
    const sig = quantile(sorted, SPREAD_Q) / SPREAD_Z;
    spread = Math.max(FLOOR_MULT * Math.sqrt(horizonDays), sig, MIN_SPREAD);
  }
  med = clamp(med, toLogit(0.01), toLogit(99.99));
  const qs = ZS.map((z) => r4(pctClamp(fromLogit(med + z * spread))));
  for (let i = 1; i < qs.length; i++) if (qs[i] < qs[i - 1]) qs[i] = qs[i - 1];
  return qs;
}

// ---- Driver --------------------------------------------------------------
mkdirSync(OUT_DIR, { recursive: true });
const inputs = readdirSync(IN_DIR)
  .filter((f) => f.endsWith(".in"))
  .sort();
for (const file of inputs) {
  const id = file.slice(0, -".in".length);
  const c = JSON.parse(readFileSync(join(IN_DIR, file), "utf8"));
  if (c.levels.length !== TAUS.length || c.levels.some((v: number, j: number) => v !== TAUS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}`);
  }
  if (c.history.length === 0) throw new Error(`${file} has no history`);
  const days = c.horizon_minutes / 1440;
  const trophies = c.trophies.map((tr: any, i: number) => ({
    id: tr.id,
    q: forecast(buildTrophy(c.history, i, tr.id), Date.parse(c.target), days),
  }));
  writeFileSync(join(OUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
