"""Build a self-contained index.html: a CLICK-THROUGH GUIDE.

Each step shows the screen you are on with a marker on WHERE TO CLICK, and a small
static inset of the resulting screen. Prev/Next walk through the whole sequence —
"click here -> next screen -> click here -> ..." — with no transition animation.

Markers that come from a confident, freshly-tracked cursor are solid red; markers
inferred from a stale/uncertain cursor are dashed amber and labelled approximate,
so you know which to trust. The stage uses a fixed aspect ratio from the video
dimensions so it is always sized, even before images load (matters over file://).
"""

from __future__ import annotations

import json
import pathlib

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>clickshot — click guide</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.5 system-ui,Segoe UI,Roboto,sans-serif;
         background:#0e1116; color:#e6edf3; display:flex; height:100vh; overflow:hidden; }
  #side { width:280px; flex:none; border-right:1px solid #222a35; overflow-y:auto; background:#0b0e13; }
  #side h1 { font-size:15px; padding:14px 16px; margin:0; border-bottom:1px solid #222a35; }
  #side .meta { padding:8px 16px; color:#8b98a9; font-size:12px; border-bottom:1px solid #222a35; }
  .item { padding:10px 16px; border-bottom:1px solid #161b22; cursor:pointer; display:flex; gap:10px; align-items:center; }
  .item:hover { background:#11161d; }
  .item.active { background:#16202c; border-left:3px solid #2f81f7; padding-left:13px; }
  .item img { width:64px; height:36px; object-fit:cover; border-radius:3px; background:#000; flex:none; }
  .item .lbl { font-size:12px; }
  .item .conf { font-size:11px; color:#8b98a9; }
  .item .dot { width:7px; height:7px; border-radius:50%; flex:none; }
  #main { flex:1; display:flex; flex-direction:column; min-width:0; }
  #stage { flex:1; display:flex; align-items:center; justify-content:center; padding:20px; min-height:0; }
  #viewport { position:relative; background:#000; border-radius:6px; overflow:hidden;
              aspect-ratio: __VW__ / __VH__; width: min(100%, calc(80vh * __VW__ / __VH__)); }
  #screen { position:absolute; inset:0; width:100%; height:100%; object-fit:contain; display:block; }
  /* persistent hotspot: where to click */
  #hotspot { position:absolute; width:48px; height:48px; margin:-24px 0 0 -24px; border-radius:50%;
             border:3px solid #ff4d4d; opacity:0; pointer-events:none; z-index:6;
             box-shadow:0 0 16px rgba(255,77,77,.8); }
  #hotspot::after { content:''; position:absolute; inset:36%; border-radius:50%;
                    background:#ff4d4d; box-shadow:0 0 8px rgba(255,77,77,.95); }
  #hotspot.show { opacity:1; animation:pulse 1.4s ease-out infinite; }
  #hotspot.approx { border-style:dashed; border-color:#ffce4d; box-shadow:0 0 16px rgba(255,206,77,.6); }
  #hotspot.approx::after { background:#ffce4d; box-shadow:0 0 8px rgba(255,206,77,.9); }
  @keyframes pulse { 0%{ box-shadow:0 0 0 0 rgba(255,77,77,.5);} 70%{ box-shadow:0 0 0 20px rgba(255,77,77,0);} 100%{ box-shadow:0 0 0 0 rgba(255,77,77,0);} }
  /* static inset of the resulting screen */
  #result { position:absolute; right:10px; bottom:10px; width:28%; z-index:5;
            border:1px solid #2b3440; border-radius:5px; overflow:hidden; background:#000;
            box-shadow:0 4px 14px rgba(0,0,0,.6); opacity:.92; }
  #result.left { left:10px; right:auto; }
  #result img { width:100%; display:block; }
  #result .cap { font-size:10px; color:#cdd6e0; background:#0b0e13cc; padding:2px 6px; }
  #bar { flex:none; border-top:1px solid #222a35; padding:12px 18px; background:#0b0e13; }
  #ctrls { display:flex; gap:10px; align-items:center; }
  button { background:#1f6feb; color:#fff; border:0; border-radius:6px; padding:8px 16px; font-size:13px; cursor:pointer; }
  button.secondary { background:#21262d; }
  button:disabled { opacity:.4; cursor:default; }
  #caption { margin-top:10px; color:#c9d4e0; }
  #caption .t { color:#8b98a9; }
  #caption .warn { color:#ffce4d; }
  .reasons { margin-top:4px; color:#8b98a9; font-size:12px; }
  .pill { display:inline-block; padding:1px 8px; border-radius:10px; background:#16202c; margin:2px 4px 0 0; font-size:11px; }
</style>
</head>
<body>
<div id="side">
  <h1>clickshot · click guide</h1>
  <div class="meta" id="sideMeta"></div>
  <div id="list"></div>
</div>
<div id="main">
  <div id="stage">
    <div id="viewport">
      <img id="screen" alt="screen"/>
      <div id="hotspot"></div>
      <div id="result"><img id="resultImg" alt="result"/><div class="cap">result of this click &rarr;</div></div>
    </div>
  </div>
  <div id="bar">
    <div id="ctrls">
      <button class="secondary" id="prev">&larr; Prev</button>
      <button id="next">Next &rarr;</button>
      <button class="secondary" id="pulse">◎ Show spot</button>
      <span id="counter" style="color:#8b98a9"></span>
    </div>
    <div id="caption"></div>
  </div>
</div>
<script>
const STEPS = __STEPS__;
const META = __META__;
let cur = 0;

const screenImg = document.getElementById('screen');
const resultImg = document.getElementById('resultImg');
const hotspot = document.getElementById('hotspot');
const list = document.getElementById('list');

document.getElementById('sideMeta').innerHTML =
  `${STEPS.length} step(s) · click where marked to advance<br>${META.source} · ${META.w}×${META.h}`;

STEPS.forEach((s, i) => {
  const d = document.createElement('div');
  d.className = 'item';
  const color = s.approx ? '#ffce4d' : '#ff4d4d';
  d.innerHTML = `<span class="dot" style="background:${color}"></span>` +
                `<img src="${s.before}"/><div><div class="lbl">Step ${i+1}` +
                ` <span class="conf">· ${(s.confidence*100).toFixed(0)}%</span></div>` +
                `<div class="conf">${s.t.toFixed(1)}s</div></div>`;
  d.onclick = () => show(i);
  list.appendChild(d);
});

function placeHotspot() {
  const s = STEPS[cur];
  hotspot.classList.toggle('approx', !!s.approx);
  if (!s.click) { hotspot.classList.remove('show'); return; }
  hotspot.style.left = (s.click.x_norm * 100) + '%';
  hotspot.style.top  = (s.click.y_norm * 100) + '%';
  hotspot.classList.add('show');
}

const resultBox = document.getElementById('result');

function render() {
  const s = STEPS[cur];
  screenImg.src = s.before;          // the screen you click ON
  resultImg.src = s.after;           // what the click produces (static inset)
  // keep the result inset out from under the marker (opposite bottom corner)
  resultBox.classList.toggle('left', !!(s.click && s.click.x_norm > 0.5));
  placeHotspot();
  document.getElementById('counter').textContent = `Step ${cur+1} / ${STEPS.length}`;
  document.getElementById('prev').disabled = cur === 0;
  document.getElementById('next').disabled = cur === STEPS.length - 1;
  const reasons = s.reasons.map(r => `<span class="pill">${r}</span>`).join('');
  const where = s.click
    ? `click the ${s.approx ? '<span class="warn">amber (approximate)</span>' : 'red'} marker`
    : '<span class="warn">click location unknown</span>';
  document.getElementById('caption').innerHTML =
    `<b>Step ${cur+1}.</b> On this screen, ${where}.` +
    ` <span class="t">· ${s.t.toFixed(1)}s · confidence ${(s.confidence*100).toFixed(0)}%</span>` +
    `<div class="reasons">${reasons}</div>`;
  [...list.children].forEach((el, i) => el.classList.toggle('active', i === cur));
}

function pulseOnce() {
  hotspot.classList.remove('show');
  void hotspot.offsetWidth;
  placeHotspot();
}

function show(i) { cur = Math.max(0, Math.min(STEPS.length - 1, i)); render(); }
document.getElementById('prev').onclick = () => show(cur - 1);
document.getElementById('next').onclick = () => show(cur + 1);
document.getElementById('pulse').onclick = pulseOnce;
window.addEventListener('resize', placeHotspot);
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowRight') show(cur + 1);
  if (e.key === 'ArrowLeft')  show(cur - 1);
});

if (STEPS.length) render();
else document.getElementById('caption').textContent = 'No click consequences detected.';
</script>
</body>
</html>
"""


def build(manifest: dict, outdir: str) -> str:
    steps = []
    for e in manifest["events"]:
        cf = e["consequence_frame"]
        reasons = e["reasons"]
        approx = (e["confidence"] < 0.6
                  or any(("stale" in r or "no cursor" in r) for r in reasons))
        steps.append({
            "before": cf["before_file"],
            "after": cf["file"],
            "t": e["transition"]["start_t_s"],
            "confidence": e["confidence"],
            "approx": approx,
            "reasons": reasons,
            "click": ({"x_norm": e["click"]["x_norm"], "y_norm": e["click"]["y_norm"]}
                      if e.get("click") else None),
        })
    vw = manifest["meta"]["video_w"] or 16
    vh = manifest["meta"]["video_h"] or 9
    meta = {"source": pathlib.Path(manifest["meta"]["source_video"]).name, "w": vw, "h": vh}
    html = (_TEMPLATE
            .replace("__STEPS__", json.dumps(steps))
            .replace("__META__", json.dumps(meta))
            .replace("__VW__", str(vw))
            .replace("__VH__", str(vh)))
    path = pathlib.Path(outdir) / "index.html"
    path.write_text(html, encoding="utf-8")
    return str(path)
