/* Research page: figures and the "Show the working" switch.
   Every number below is copied from results/chbmit/summary.csv, results/leakage/summary.csv
   and the int8 runs (RESEARCH_NOTES.md sections 13.3–13.6). */
const NS="http://www.w3.org/2000/svg";
function el(tag,attrs,parent,text){const e=document.createElementNS(NS,tag);for(const k in attrs)e.setAttribute(k,attrs[k]);if(text!=null)e.textContent=text;if(parent)parent.appendChild(e);return e}
const BONE="#E7E3D6",MOSS="#8E9B8A",LINE="rgba(142,155,138,.28)";

/* ---- show the working toggle ---- */
const lens=document.getElementById("lens");
lens.addEventListener("change",()=>document.querySelectorAll("details.work").forEach(d=>d.open=lens.checked));

/* ---- 01 seizures caught per child (18 channels, final rule) ---- */
/* [patient, seizures, EEGNet caught, CNN caught] */
const PT=[["chb01",7,7,7],["chb03",7,7,7],["chb05",5,5,5],["chb06",10,0,0],["chb08",5,5,5],["chb10",7,7,7],
["chb13",12,2,4],["chb14",8,0,0],["chb15",20,12,19],["chb16",10,0,0],["chb18",6,6,5],["chb20",8,7,3],["chb23",7,6,5],["chb24",16,16,16]];
(function(){
  const g=document.getElementById("sz");
  const add=(cls,html)=>{const d=document.createElement("div");d.className=cls;d.innerHTML=html;g.appendChild(d)};
  add("hd","Child");add("hd","EEGNet · 1,729 weights");add("hd","Baseline CNN · 328,385 weights");
  const dots=(n,c)=>{let s="";for(let i=0;i<n;i++)s+=`<i class="dot${i<c?" e":""}"></i>`;return s};
  PT.forEach(([p,n,e,c])=>{
    add("id",p);
    add("row",`<span class="n">${e}/${n}</span><span class="sd">${dots(n,e)}</span>`);
    add("row",`<span class="n">${c}/${n}</span><span class="sd">${dots(n,c)}</span>`);
  });
  const t=(k)=>PT.reduce((a,r)=>a+r[k],0);
  add("id","Total");
  add("row",`<span class="tot">${t(2)} of ${t(1)}</span>`);
  add("row",`<span class="tot">${t(3)} of ${t(1)}</span>`);
})();

/* ---- 02 leakage: shuffled vs grouped, dot plot on 0.80–1.00 ---- */
(function(){
  const s=document.getElementById("leak");
  const M=[["Accuracy",.988,.974],["Sensitivity",.958,.887],["Specificity",.995,.995],["F1",.970,.930]];
  const L=150,R=640,lo=.80,hi=1.0,x=v=>L+(v-lo)/(hi-lo)*(R-L),top=40,step=56;
  [.80,.85,.90,.95,1.0].forEach(v=>{
    el("line",{x1:x(v),x2:x(v),y1:top-16,y2:top+step*3+16,stroke:LINE},s);
    el("text",{x:x(v),y:top+step*3+40,class:"sv-c","text-anchor":"middle"},s,Math.round(v*100)+"%");
  });
  M.forEach(([n,a,b],i)=>{
    const y=top+i*step,strong=n==="Sensitivity";
    el("text",{x:0,y:y+5,class:strong?"sv-t":"sv-s"},s,n);
    el("line",{x1:x(b),x2:x(a),y1:y,y2:y,stroke:strong?BONE:MOSS},s);
    el("circle",{cx:x(b),cy:y,r:6,fill:BONE},s);
    el("circle",{cx:x(a),cy:y,r:6,fill:"#0F1A15",stroke:BONE},s);
    const gap=Math.round((a-b)*1000)/10;
    el("text",{x:R+28,y:y+5,class:strong?"sv-t":"sv-s"},s,gap>0?`+${gap.toFixed(1)} pts`:"no change");
  });
})();

/* ---- 03 channels: seizures caught at 18, 4, 2 ---- */
(function(){
  const s=document.getElementById("chan");
  const L=60,R=560,T=20,B=280,xs=[L,(L+R)/2,R],y=v=>B-v/128*(B-T);
  [0,32,64,96,128].forEach(v=>{el("line",{x1:L-12,x2:R+12,y1:y(v),y2:y(v),stroke:LINE},s);el("text",{x:L-20,y:y(v)+4,class:"sv-c","text-anchor":"end"},s,v)});
  ["18 channels","4 channels","2 channels"].forEach((t,i)=>el("text",{x:xs[i],y:B+32,class:"sv-s","text-anchor":"middle"},s,t));
  const S=[["Baseline CNN",[83,88,85],BONE],["EEGNet",[80,83,80],MOSS]];
  S.forEach(([n,v,c],k)=>{
    el("path",{d:v.map((u,i)=>(i?"L":"M")+xs[i]+" "+y(u)).join(" "),stroke:c,fill:"none","stroke-width":1.5},s);
    v.forEach((u,i)=>{el("circle",{cx:xs[i],cy:y(u),r:4,fill:c},s);
      el("text",{x:xs[i],y:y(u)+(k?22:-12),class:"sv-c","text-anchor":"middle",fill:c},s,u)});
    el("text",{x:R+24,y:y(v[2])+(k?14:-2),class:"sv-t",fill:c},s,n);
  });
})();

/* ---- 04 size squares, area to scale (2-channel models, KB) ---- */
(function(){
  const s=document.getElementById("size");
  const K=300/Math.sqrt(1279); // side in px per sqrt(KB)
  const Q=[["CNN, float",1279,false],["CNN, 8-bit",350,true],["EEGNet, float",11.0,false],["EEGNet, 8-bit",9.7,true]];
  let x=0;const base=320;
  Q.forEach(([n,kb,q],i)=>{
    const a=Math.sqrt(kb)*K;
    el("rect",{x,y:base-a,width:a,height:a,fill:q?BONE:"none",stroke:BONE,"fill-opacity":q?.14:0},s);
    el("text",{x,y:base+22,class:"sv-t","font-size":14},s,(kb>=100?Math.round(kb).toLocaleString("en"):kb.toFixed(1))+" KB");
    el("text",{x,y:base+40,class:"sv-s"},s,n);
    x+=Math.max(a,96)+(i===1?72:24);
  });
})();
