const API = "";

type ModelInfo = {
  trained: boolean;
  path: string;
  size_kb?: number;
  modified_utc?: string;
  message?: string;
};

type MetricRow = {
  strategy: string;
  cumulative_return: number;
  annualized_return: number;
  sharpe: number;
  max_drawdown: number;
  sortino?: number;
  calmar?: number;
};

type EquityCurve = {
  strategy: string;
  dates: string[];
  equity: number[];
};

type EvalResult = {
  tickers: string[];
  test_start: string;
  test_end: string;
  test_days: number;
  model: ModelInfo;
  metrics: MetricRow[];
  equity_curves: EquityCurve[];
  latest_ppo_weights: Record<string, number> | null;
};

const runBtn = document.getElementById("run-btn") as HTMLButtonElement;
const statusEl = document.getElementById("status") as HTMLSpanElement;
const metaEl = document.getElementById("meta") as HTMLParagraphElement;
const modelEl = document.getElementById("model-info") as HTMLParagraphElement;
const tableWrap = document.getElementById("table-wrap") as HTMLDivElement;
const canvas = document.getElementById("chart") as HTMLCanvasElement;
const weightsPanel = document.getElementById("weights-panel") as HTMLElement;
const weightsList = document.getElementById("weights-list") as HTMLUListElement;

function fmt(n: number, digits = 4): string {
  return Number.isFinite(n) ? n.toFixed(digits) : "—";
}

function renderTable(rows: MetricRow[]): void {
  const cols: { key: keyof MetricRow; label: string }[] = [
    { key: "strategy", label: "Strategy" },
    { key: "sharpe", label: "Sharpe" },
    { key: "cumulative_return", label: "Cum. return" },
    { key: "annualized_return", label: "Ann. return" },
    { key: "max_drawdown", label: "Max DD" },
  ];
  let html = "<table><thead><tr>";
  for (const c of cols) html += `<th>${c.label}</th>`;
  html += "</tr></thead><tbody>";
  for (const row of rows) {
    html += "<tr>";
    for (const c of cols) {
      const v = row[c.key];
      html += `<td>${typeof v === "number" ? fmt(v) : String(v)}</td>`;
    }
    html += "</tr>";
  }
  html += "</tbody></table>";
  tableWrap.innerHTML = html;
}

const COLORS = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444", "#a855f7"];

function drawChart(curves: EquityCurve[]): void {
  const ctx = canvas.getContext("2d");
  if (!ctx || curves.length === 0) return;

  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 900;
  const h = canvas.clientHeight || 360;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  ctx.scale(dpr, dpr);

  const pad = { l: 48, r: 16, t: 16, b: 36 };
  const plotW = w - pad.l - pad.r;
  const plotH = h - pad.t - pad.b;

  let ymin = Infinity;
  let ymax = -Infinity;
  for (const c of curves) {
    for (const v of c.equity) {
      ymin = Math.min(ymin, v);
      ymax = Math.max(ymax, v);
    }
  }
  const margin = (ymax - ymin) * 0.05 || 1;
  ymin -= margin;
  ymax += margin;

  const n = curves[0].dates.length;

  ctx.fillStyle = "#0f1419";
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = "#2a3441";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + (plotH * i) / 4;
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(pad.l + plotW, y);
    ctx.stroke();
  }

  curves.forEach((curve, idx) => {
    ctx.strokeStyle = COLORS[idx % COLORS.length];
    ctx.lineWidth = 2;
    ctx.beginPath();
    curve.equity.forEach((eq, i) => {
      const x = pad.l + (i / (n - 1)) * plotW;
      const y = pad.t + plotH * (1 - (eq - ymin) / (ymax - ymin));
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });

  ctx.fillStyle = "#94a3b8";
  ctx.font = "12px system-ui, sans-serif";
  ctx.fillText(fmt(ymax, 0), 4, pad.t + 10);
  ctx.fillText(fmt(ymin, 0), 4, pad.t + plotH);

  let legendX = pad.l;
  ctx.font = "13px system-ui, sans-serif";
  curves.forEach((curve, idx) => {
    ctx.fillStyle = COLORS[idx % COLORS.length];
    ctx.fillRect(legendX, h - 22, 12, 12);
    ctx.fillStyle = "#e2e8f0";
    ctx.fillText(curve.strategy, legendX + 16, h - 12);
    legendX += ctx.measureText(curve.strategy).width + 40;
  });
}

function showWeights(weights: Record<string, number> | null): void {
  if (!weights) {
    weightsPanel.hidden = true;
    return;
  }
  weightsPanel.hidden = false;
  weightsList.innerHTML = "";
  for (const [ticker, w] of Object.entries(weights)) {
    const li = document.createElement("li");
    li.textContent = `${ticker}: ${(w * 100).toFixed(1)}%`;
    weightsList.appendChild(li);
  }
}

async function loadModel(): Promise<void> {
  const res = await fetch(`${API}/api/model`);
  const m: ModelInfo = await res.json();
  if (m.trained) {
    modelEl.textContent = `Model: ${m.path} (${m.size_kb} KB, updated ${m.modified_utc ?? "?"})`;
  } else {
    modelEl.textContent = m.message ?? "No model on disk.";
  }
}

async function runEval(): Promise<void> {
  runBtn.disabled = true;
  statusEl.textContent = "Running backtest…";
  try {
    const res = await fetch(`${API}/api/evaluate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail ?? res.statusText);
    }
    const data: EvalResult = await res.json();
    statusEl.textContent = "Done";
    metaEl.textContent = `Test window: ${data.test_start} → ${data.test_end} (${data.test_days} days) · ${data.tickers.join(", ")}`;
    renderTable(data.metrics);
    drawChart(data.equity_curves);
    showWeights(data.latest_ppo_weights);
  } catch (e) {
    statusEl.textContent = "Error";
    metaEl.textContent = e instanceof Error ? e.message : String(e);
  } finally {
    runBtn.disabled = false;
  }
}

runBtn.addEventListener("click", () => void runEval());
void loadModel();
