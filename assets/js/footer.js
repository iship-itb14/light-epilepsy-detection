/* Footer art: brainwave ridgelines that drift as the page scrolls. Shared by every page. */
(function(){
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
const dpr = Math.min(devicePixelRatio || 1, 2);
const rc = document.getElementById('ridges'); if (!rc) return;
const rx = rc.getContext('2d'); let RW, RH;
function sizeRidges(){const r=rc.getBoundingClientRect();RW=r.width;RH=r.height;rc.width=RW*dpr;rc.height=RH*dpr}
sizeRidges(); addEventListener('resize', sizeRidges);
function drawRidges(time){
  const r=rc.getBoundingClientRect();if(r.bottom<0||r.top>innerHeight)return;
  const shift=(innerHeight-r.top)*.35;
  rx.setTransform(dpr,0,0,dpr,0,0);rx.clearRect(0,0,RW,RH);
  const rows=14;
  for(let j=0;j<rows;j++){
    const base=40+j*(RH-50)/rows, amp=6+j*1.6, off=shift*(.3+j*.06)+(reduce?0:time*.004*(j%3+1));
    const top=new Path2D();
    for(let x=0;x<=RW;x+=4){
      const u=(x+off);
      const y=base-Math.abs(Math.sin(u*.012+j))*amp-Math.sin(u*.031+j*1.7)*amp*.4-Math.pow(Math.max(0,Math.sin(u*.004+j*.6)),12)*amp*2.2;
      x?top.lineTo(x,y):top.moveTo(x,y);
    }
    const area=new Path2D(top);area.lineTo(RW,RH);area.lineTo(0,RH);area.closePath();
    rx.fillStyle='#0F1A15';rx.fill(area);
    rx.strokeStyle=j===6?'rgba(227,163,59,.85)':`rgba(142,155,138,${.18+j*.03})`;
    rx.lineWidth=1;rx.stroke(top);
  }
}


function frame(t){drawRidges(t); if(!reduce) requestAnimationFrame(frame)}
requestAnimationFrame(frame);
if (reduce) addEventListener('scroll', () => drawRidges(0), {passive:true});
})();

/* Pages without the home-page globe: give the nav a solid background once the page scrolls. */
(function(){
  if (document.getElementById('hero')) return;
  const nav = document.getElementById('nav'); if (!nav || nav.classList.contains('solid')) return;
  const on = () => nav.classList.toggle('solid', scrollY > innerHeight * 0.5);
  addEventListener('scroll', on, {passive:true}); on();
})();
