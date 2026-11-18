fetch('assets/data/land-dots.json').then(r=>r.json()).then(DOTS=>{
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const lerp=(a,b,t)=>a+(b-a)*t;
const ease=t=>t<.5?2*t*t:1-Math.pow(-2*t+2,2)/2;

/* ---------- Globe ---------- */
const cv=document.getElementById('globe'), cx=cv.getContext('2d');
let W,H,R,dpr;
function size(){dpr=Math.min(devicePixelRatio||1,2);const r=cv.getBoundingClientRect();W=r.width;H=r.height;cv.width=W*dpr;cv.height=H*dpr;R=Math.min(W,H)*.4}
size();addEventListener('resize',size);
const N=DOTS.length/2, LON=new Float32Array(N), LAT=new Float32Array(N);
for(let i=0;i<N;i++){LON[i]=DOTS[2*i]/10*Math.PI/180;LAT[i]=DOTS[2*i+1]/10*Math.PI/180}
// Illustrative lit points: a deterministic sample of land dots
const LIT=[];let seed=7;const rnd=()=>(seed=(seed*16807)%2147483647)/2147483647;
for(let i=0;i<N;i++) if(rnd()<.035) LIT.push({i,ph:rnd()*Math.PI*2,sp:.4+rnd()*.9});
// Camera keyframes per story step: [longitude centre, latitude tilt, zoom]
const KEYS=[[-40,18,1],[20,8,1.1],[15,4,1.18],[78,16,1.1],[78,16,0.9]];
let prog=0;
function cam(p){const s=p*(KEYS.length-1);const k=Math.min(Math.floor(s),KEYS.length-2);const t=ease(s-k);
  return KEYS[k].map((v,j)=>lerp(v,KEYS[k+1][j],t))}
function drawGlobe(time){
  const [lon0,lat0,z]=cam(prog);
  const drift=reduce?0:time*0.0000;
  const l0=(lon0)*Math.PI/180+drift, p0=lat0*Math.PI/180, r=R*z;
  const cp=Math.cos(p0), sp=Math.sin(p0);
  cx.setTransform(dpr,0,0,dpr,0,0);cx.clearRect(0,0,W,H);
  const ox=W/2, oy=H/2;
  // sphere
  cx.beginPath();cx.arc(ox,oy,r,0,Math.PI*2);cx.fillStyle='#17261F';cx.fill();
  cx.strokeStyle='rgba(142,155,138,.25)';cx.lineWidth=1;cx.stroke();
  // land dots
  const ds=Math.max(1,r/260);
  cx.fillStyle='#8E9B8A';
  for(let i=0;i<N;i++){
    const la=LAT[i], lo=LON[i]-l0, cl=Math.cos(la);
    const x=cl*Math.sin(lo), y=cp*Math.sin(la)-sp*cl*Math.cos(lo), zc=sp*Math.sin(la)+cp*cl*Math.cos(lo);
    if(zc<0)continue;
    cx.globalAlpha=.25+.6*zc;
    cx.fillRect(ox+x*r-ds/2, oy-y*r-ds/2, ds, ds);
  }
  // lit points
  const vis=clamp((prog-.08)*4,0,1);
  if(vis>0){for(const p of LIT){
    const la=LAT[p.i], lo=LON[p.i]-l0, cl=Math.cos(la);
    const zc=sp*Math.sin(la)+cp*cl*Math.cos(lo); if(zc<.05)continue;
    const x=cl*Math.sin(lo), y=cp*Math.sin(la)-sp*cl*Math.cos(lo);
    const f=reduce?.7:(.5+.5*Math.sin(time*.001*p.sp+p.ph));
    cx.globalAlpha=vis*zc*(.25+.75*f*f);
    cx.fillStyle='#E3A33B';
    cx.beginPath();cx.arc(ox+x*r,oy-y*r,ds*(1.1+f*1.2),0,Math.PI*2);cx.fill();
  }}
  cx.globalAlpha=1;
}

/* ---------- Scroll story ---------- */
const hero=document.getElementById('hero'), steps=[...document.querySelectorAll('.step')],
      ticks=[...document.querySelectorAll('.progress i')], cue=document.getElementById('cue'),
      nav=document.getElementById('nav');
function onScroll(){
  const r=hero.getBoundingClientRect(), span=hero.offsetHeight-innerHeight;
  prog=clamp(-r.top/span,0,1);
  const idx=Math.min(steps.length-1,Math.floor(prog*steps.length*1.0001));
  steps.forEach((s,i)=>s.classList.toggle('on',i===idx));
  ticks.forEach((t,i)=>t.classList.toggle('on',i===idx));
  cue.style.opacity=prog>.02?0:1;
  nav.classList.toggle('solid',r.bottom<innerHeight*.6);
  clearUpdate(); traceUpdate();
}

/* ---------- Blur-to-clear words ---------- */
const ct=document.getElementById('clearText');
ct.innerHTML=ct.textContent.split(' ').map(w=>`<span>${w}</span>`).join(' ');
const words=[...ct.querySelectorAll('span')];
function clearUpdate(){
  if(reduce)return;
  const r=ct.getBoundingClientRect();
  const t=clamp((innerHeight*.85-r.top)/(r.height+innerHeight*.35),0,1);
  const n=words.length;
  words.forEach((w,i)=>{const k=clamp(t*1.25*n-i,0,1);
    w.style.filter=`blur(${(1-k)*6}px)`;w.style.opacity=.18+.82*k});
}

/* ---------- Illustrative EEG trace ---------- */
function wave(x0,x1,fn){let d='';for(let x=x0;x<=x1;x+=2){d+=(x===x0?'M':'L')+x+','+(110+fn(x)).toFixed(1)}return d}
let s2=11;const r2=()=>(s2=(s2*16807)%2147483647)/2147483647-.5;
const noise=[];for(let i=0;i<1700;i++)noise.push(r2());
const sm=x=>{const i=Math.floor(x/2);return (noise[i%1700]+noise[(i+1)%1700]+noise[(i+3)%1700])};
const calm=x=>sm(x)*16+Math.sin(x*.07)*7+Math.sin(x*.19)*4;
const burst=x=>{const t=(x-780)/420;const env=Math.sin(Math.PI*clamp(t,0,1));
  const ph=(x*.055)%(Math.PI*2);const spike=Math.pow(Math.max(0,Math.cos(ph)),9)*-78+Math.sin(ph+1.2)*26;
  return env*spike+(1-env)*calm(x)+sm(x)*4};
const pc=document.getElementById('eegCalm'), pb=document.getElementById('eegBurst'), pa=document.getElementById('eegAfter');
pc.setAttribute('d',wave(0,780,calm));pb.setAttribute('d',wave(780,1200,burst));pa.setAttribute('d',wave(1200,1600,calm));
const segs=[pc,pb,pa].map(p=>{const L=p.getTotalLength();p.style.strokeDasharray=L;p.style.strokeDashoffset=reduce?0:L;return L});
const traceEl=document.getElementById('trace');
function traceUpdate(){
  if(reduce)return;
  const r=traceEl.getBoundingClientRect();
  const t=clamp((innerHeight-r.top)/(innerHeight*.9),0,1);
  const total=segs[0]+segs[1]+segs[2];let drawn=t*total;
  [pc,pb,pa].forEach((p,i)=>{const d=clamp(drawn,0,segs[i]);p.style.strokeDashoffset=segs[i]-d;drawn-=segs[i]});
}
// little wrist wave
(function(){let d='';for(let x=46;x<=154;x+=1.5){const ph=(x-46)/108;const y=100+Math.sin(ph*Math.PI*6)*Math.sin(ph*Math.PI)*18;d+=(x===46?'M':'L')+x+','+y.toFixed(1)}document.getElementById('wristWave').setAttribute('d',d)})();


/* ---------- Loop ---------- */
addEventListener('scroll',onScroll,{passive:true});onScroll();
function frame(t){drawGlobe(t);if(!reduce)requestAnimationFrame(frame)}
requestAnimationFrame(frame);
if(reduce){addEventListener('scroll',()=>drawGlobe(0),{passive:true})}

});
