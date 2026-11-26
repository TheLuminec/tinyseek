// The detector's web page (served at /). The board sends the camera JPEG untouched
// (/frame.jpg, with that frame's detections in the X-Meta header) and the browser draws
// the boxes; nothing is re-encoded on the board.
#pragma once

namespace det {

static const char kIndexHtml[] = R"HTML(<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>tinyseek detector</title>
<style>
:root{--bg:#111;--fg:#eee;--dim:#999;--hit:#3ddc84;--low:#888;--err:#ff6b6b;--panel:#1d1d1d}
@media (prefers-color-scheme: light){:root{--bg:#f6f6f6;--fg:#111;--dim:#555;--hit:#0a8f3c;--low:#777;--err:#c00;--panel:#fff}}
body{margin:0;padding:12px 16px;background:var(--bg);color:var(--fg);font:15px/1.4 system-ui,sans-serif}
h1{font-size:18px;margin:0 0 8px}
#wrap{max-width:820px;margin:0 auto}
canvas{width:100%;height:auto;background:#000;border-radius:6px;display:block}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:10px 0}
input[type=text]{flex:1;min-width:140px;padding:8px;font-size:16px;border-radius:6px;border:1px solid var(--low);background:var(--panel);color:var(--fg)}
button{padding:8px 14px;font-size:15px;border-radius:6px;border:1px solid var(--low);background:var(--panel);color:var(--fg);cursor:pointer}
.chip{padding:4px 10px;font-size:14px}
#msg{min-height:1.4em}
.err{color:var(--err)}
.panel{background:var(--panel);border-radius:6px;padding:8px 10px;font:13px/1.5 ui-monospace,monospace;white-space:pre-wrap;word-break:break-word}
.dim{color:var(--dim)}
</style></head><body><div id="wrap">
<h1>tinyseek: <span id="prompt">?</span> <span id="state" class="dim"></span></h1>
<canvas id="cv" width="800" height="600"></canvas>
<form class="row" id="form"><input type="text" id="name" placeholder="type a prompt, e.g. cup" autocomplete="off">
<button type="submit">Detect</button>
<label class="dim"><input type="checkbox" id="all"> show low scores</label></form>
<div id="msg"></div>
<div class="row" id="avail"></div>
<div class="panel" id="stats">waiting for the first frame</div>
<div class="panel dim" id="info" style="margin-top:8px"></div>
</div><script>
const cv=document.getElementById('cv'),cx=cv.getContext('2d');
let have=-1,last=null,img=null;
function css(n){return getComputedStyle(document.documentElement).getPropertyValue(n).trim()}
function draw(){
  if(!img||!last)return;
  if(cv.width!==img.width||cv.height!==img.height){cv.width=img.width;cv.height=img.height}
  cx.drawImage(img,0,0);
  const all=document.getElementById('all').checked,lw=Math.max(2,cv.width/300);
  cx.font=(lw*7)+'px sans-serif';
  for(const d of last.dets){
    const hit=d.s>=last.thr; if(!hit&&!all)continue;
    cx.strokeStyle=hit?css('--hit'):css('--low'); cx.lineWidth=hit?lw:lw/2;
    cx.strokeRect(d.x,d.y,d.w,d.h);
    cx.fillStyle=cx.strokeStyle; cx.fillText(d.s.toFixed(2),d.x+2,Math.max(lw*7,d.y-3));
  }
}
function showStats(m){
  const t=m.t;
  document.getElementById('stats').textContent=
    `frame ${m.n}${m.file?' ('+m.file+')':''}  ${m.fps.toFixed(2)} fps  ${m.hit?'DETECTED':'not detected'}\n`+
    `capture ${t.cap} ms  decode ${t.dec}  preprocess ${t.pre}  inference ${t.inf}  boxes ${t.post}\n`+
    `latency ${t.lat} ms  period ${t.period} ms  detections ${m.ndet} (threshold ${m.thr})`+(m.ok?'':'\nerror: '+m.error);
}
async function tick(){
  try{
    const r=await fetch('/frame.jpg?have='+have,{cache:'no-store'});
    if(r.status===200){
      const m=JSON.parse(r.headers.get('X-Meta')),b=await r.blob(),u=URL.createObjectURL(b),i=new Image();
      i.onload=()=>{img=i;last=m;have=m.n;draw();showStats(m);URL.revokeObjectURL(u)};
      i.onerror=()=>URL.revokeObjectURL(u); i.src=u;
    }
  }catch(e){document.getElementById('state').textContent='(offline)'}
  setTimeout(tick,150);
}
function chips(list){
  const a=document.getElementById('avail'); a.innerHTML='';
  for(const p of list){const b=document.createElement('button');b.className='chip';b.textContent=p;b.onclick=()=>ask(p);a.appendChild(b)}
}
async function status(){
  try{
    const s=await (await fetch('/status.json',{cache:'no-store'})).json();
    document.getElementById('prompt').textContent=s.prompt||'(none)';
    document.getElementById('state').textContent='('+s.state+(s.message?': '+s.message:'')+')';
    chips(s.available);
    document.getElementById('info').textContent=JSON.stringify({settings:s.settings,model:s.model,mem:s.mem,wifi:s.wifi},null,1);
  }catch(e){}
  setTimeout(status,2000);
}
async function ask(name){
  const msg=document.getElementById('msg');
  try{
    const r=await (await fetch('/prompt?name='+encodeURIComponent(name),{cache:'no-store'})).json();
    msg.className=r.ok?'':'err';
    msg.textContent=r.message+(r.ok?'':'  Available: '+(r.available.join(', ')||'none'));
    if(r.available)chips(r.available);
  }catch(e){msg.className='err';msg.textContent='no answer from the board'}
}
document.getElementById('form').onsubmit=e=>{e.preventDefault();const v=document.getElementById('name').value;if(v.trim())ask(v)};
document.getElementById('all').onchange=draw;
tick();status();
</script></body></html>
)HTML";

}  // namespace det
