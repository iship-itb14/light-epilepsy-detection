/* Data & prep page: figures, code excerpts and the "Show the working" switch.
   Figure data comes from assets/data/data-page-figures.js (made by code/site/make_data_page_figures.py). */
const NS="http://www.w3.org/2000/svg", D=window.DATA;
function el(tag,attrs,parent,text){const e=document.createElementNS(NS,tag);for(const k in attrs)e.setAttribute(k,attrs[k]);if(text!=null)e.textContent=text;if(parent)parent.appendChild(e);return e}
const BONE="#E7E3D6",MOSS="#8E9B8A",EMBER="#E3A33B",CAN="#17261F";

/* ---- show the working toggle ---- */
const lens=document.getElementById("lens");
lens.addEventListener("change",()=>document.querySelectorAll("details.work").forEach(d=>d.open=lens.checked));

/* ---- code panels (excerpts copied from the scripts) ---- */
function code(id,start,lines,hl=[]){
  const pre=document.getElementById(id);
  pre.innerHTML=lines.map((l,i)=>{const n=start+i;const esc=l.replace(/&/g,"&amp;").replace(/</g,"&lt;");
    const withC=esc.replace(/(#.*)$/,'<span class="c">$1</span>');
    return `<span class="ln${hl.includes(n)?" hl":""}" data-n="${n}">${withC||" "}</span>`}).join("");
}
code("code-dl",18,[
'for p in $PATIENTS; do',
'  grep "^$p/" "$OUT/RECORDS" | while read -r rec; do',
'    ...                     # skip files already verified',
'    curl -fL -C - --retry 3 -o "$OUT/$rec" "$BASE/$rec"',
'  done',
'done',
'shasum -a 256 -c --quiet -  # every file checked'],[21]);
code("code-parse",47,[
'def parse_summary(path):',
'    """Return {file name: [(start_s, end_s), ...]}"""',
'    seizures, current, start = {}, None, None',
'    for line in path.read_text().splitlines():',
'        m = re.match(r"File Name:\\s*(\\S+)", line)',
'        ...',
'        m = re.match(r"Seizure\\s*\\d*\\s*Start Time:\\s*(\\d+)", line)',
'        ...',
'            seizures[current].append((start, int(m.group(1))))',
'    return seizures'],[55]);
code("code-filt",44,[
'SOS = butter(4, [0.5, 40], btype="bandpass", fs=FS_IN, output="sos")',
'...',
'raw_uv = raw.get_data(picks=names) * 1e6           # volts -> microvolts',
'x = sosfiltfilt(SOS, raw_uv, axis=1)               # zero phase',
'x = resample_poly(x, FS_OUT, FS_IN, axis=1)        # 256 -> 128 Hz'],[44,47,48]);
document.querySelectorAll("#code-filt .ln").forEach((s,i)=>{s.dataset.n=[44,"",114,115,116][i]});
code("code-flag",100,[
'flat = bool((seg_raw.std(axis=1) < FLAT_UV).any())   # before filtering',
'extreme = bool((np.abs(seg) > EXTREME_UV).any())     # after filtering'],[100,101]);
code("code-label",81,[
'def seizure_overlap(t0, t1, intervals):',
'    """Fraction of [t0, t1) that lies inside any seizure interval."""',
'    covered = sum(max(0, min(t1, b) - max(t0, a)) for a, b in intervals)',
'    return covered / (t1 - t0)',
'',
'label = int(frac >= SEIZURE_FRAC)   # SEIZURE_FRAC = 0.5'],[83,86]);
document.querySelectorAll("#code-label .ln").forEach((s,i)=>{if(i>3)s.dataset.n=i===5?140:""});
code("code-csv",1,[
'patient,file,start_s,seizure_frac,label,flat,extreme,aug',
'chb01,chb01_03.edf,2992.0,0.0,0,False,False,0',
'chb01,chb01_03.edf,2996.0,1.0,1,False,False,0',
'chb01,chb01_03.edf,3000.0,1.0,1,False,False,0'],[]);
code("code-norm",66,[
'def normalisation_stats(train_patients, stats_path=...):',
'    """Per-channel mean and std from the given (training) patients only."""',
'    st = np.load(stats_path)',
'    n  = sum(st[f"{p}_n"]     for p in train_patients)',
'    s  = sum(st[f"{p}_sum"]   for p in train_patients)',
'    ss = sum(st[f"{p}_sumsq"] for p in train_patients)',
'    mean = s / n',
'    std = np.sqrt(ss / n - mean ** 2)',
'    return mean, std'],[69,70,71]);

/* ---- game plan nodes ---- */
const steps=[
 ["Gather","children's hospital EEG","done"],["Clean","remove hum and drift","done"],
 ["Cut and label","4-second slices","done"],["Split by patient","no child in two sets","done"],
 ["Train two models","yardstick and wearable","now"],["Shrink","8-bit, then pruning","todo"],
 ["Test fairly","new children, new hospital","todo"],["Share","results and toolkit","todo"]];
const xs=[80,230,380,530,720,870,1020,1170], g=document.getElementById("nodes");
steps.forEach((s,i)=>{
  const x=xs[i];
  if(s[2]==="now"){el("circle",{cx:x,cy:150,r:22,fill:"none",stroke:EMBER,"stroke-width":1,class:"pulse"},g);}
  el("circle",{cx:x,cy:150,r:22,fill:"#0F1A15",stroke:s[2]==="now"?EMBER:(s[2]==="done"?BONE:MOSS),"stroke-width":1},g);
  if(s[2]==="done")el("circle",{cx:x,cy:150,r:5,fill:BONE},g);
  if(s[2]==="now")el("circle",{cx:x,cy:150,r:5,fill:EMBER},g);
  el("text",{x,y:206,class:"sv-k","text-anchor":"middle"},g,String(i+1).padStart(2,"0"));
  el("text",{x,y:230,class:"sv-t","text-anchor":"middle","font-size":15},g,s[0]);
  el("text",{x,y:250,class:"sv-s","text-anchor":"middle"},g,s[1]);
});
el("text",{x:720,y:108,"text-anchor":"middle",fill:EMBER,style:"font:500 11px Instrument Sans,sans-serif;letter-spacing:.22em"},g,"WE ARE HERE");
const pl=document.getElementById("plan-list");
steps.forEach((s,i)=>{const li=document.createElement("li");li.innerHTML=`<span class="d ${s[2]==="done"?"f":s[2]==="now"?"now":""}"></span><span>${String(i+1).padStart(2,"0")} · ${s[0]}<small>${s[1]}${s[2]==="now"?" · we are here":""}</small></span>`;pl.appendChild(li)});

/* ---- head montage ---- */
(function(){
  const s=document.getElementById("head"),cx=230,cy=210,R=170;
  el("circle",{cx,cy,r:R,class:"sv-l"},s);
  el("path",{d:`M${cx-18} ${cy-R+2} L${cx} ${cy-R-22} L${cx+18} ${cy-R+2}`,class:"sv-l"},s);
  el("path",{d:`M${cx-R} ${cy-26} q-20 26 0 52`,class:"sv-l"},s);
  el("path",{d:`M${cx+R} ${cy-26} q20 26 0 52`,class:"sv-l"},s);
  el("circle",{cx,cy,r:R*.5,class:"sv-m","stroke-dasharray":"2 4"},s);
  const P={FP1:[-.31,-.95],FP2:[.31,-.95],F7:[-.81,-.59],F8:[.81,-.59],T7:[-1,0],T8:[1,0],P7:[-.81,.59],P8:[.81,.59],O1:[-.31,.95],O2:[.31,.95],
    F3:[-.42,-.5],F4:[.42,-.5],C3:[-.5,0],C4:[.5,0],P3:[-.42,.5],P4:[.42,.5],FZ:[0,-.5],CZ:[0,0],PZ:[0,.5]};
  const k=R*.84,pt=n=>[cx+P[n][0]*k,cy+P[n][1]*k];
  const chains=[["FP1","F7","T7","P7","O1"],["FP1","F3","C3","P3","O1"],["FP2","F4","C4","P4","O2"],["FP2","F8","T8","P8","O2"],["FZ","CZ","PZ"]];
  chains.forEach(c=>{for(let i=0;i<c.length-1;i++){const a=pt(c[i]),b=pt(c[i+1]);el("line",{x1:a[0],y1:a[1],x2:b[0],y2:b[1],stroke:BONE,"stroke-width":1,opacity:.75},s)}});
  for(const n in P){const [x,y]=pt(n);el("circle",{cx:x,cy:y,r:7,fill:"#0F1A15",stroke:BONE,"stroke-width":1},s);el("text",{x:x+(P[n][0]>=0?10:-10),y:y-10,class:"sv-c","text-anchor":P[n][0]>=0?"start":"end"},s,n)}
  // patent-style callouts
  const call=(from,ty,label,sub)=>{const [x,y]=from;el("path",{d:`M${x} ${y} L${Math.max(x+40,cx+R+20)} ${ty} H450`,class:"sv-m"},s);el("circle",{cx:450,cy:ty,r:2.5,fill:MOSS},s);el("text",{x:462,y:ty+4,class:"sv-t","font-size":14},s,label);el("text",{x:462,y:ty+22,class:"sv-s"},s,sub)};
  const a=pt("F8"),b=pt("T8");call([(a[0]+b[0])/2,(a[1]+b[1])/2],110,"F8–T8","one channel");
  call(pt("T8"),210,"T8","one electrode");
  const c=pt("P8"),d=pt("O2");call([(c[0]+d[0])/2,(c[1]+d[1])/2],330,"18 channels","4 chains + midline");
})();

/* ---- patient tags ---- */
const pp=D.patients,ids=Object.keys(pp);
const tg=document.getElementById("ptags");ids.forEach(p=>{const t=document.createElement("span");t.className="tag";t.textContent=p;tg.appendChild(t)});

/* ---- real EEG trace ---- */
(function(){
  const s=document.getElementById("trace"),T=D.trace,chs=["FP1-F7","F8-T8","T7-P7","C3-P3","FZ-CZ","P8-O2"];
  const L=80,W=1160,top=10,row=62,t0=T.t0,dur=T[chs[0]].length/T.fs,X=t=>L+(t-t0)/dur*W;
  // seizure band
  el("rect",{x:X(2996),y:0,width:X(3036)-X(2996),height:row*chs.length+20,fill:EMBER,opacity:.07},s);
  el("path",{d:`M${X(2996)} 0 V${row*chs.length+20} M${X(3036)} 0 V${row*chs.length+20}`,stroke:EMBER,"stroke-width":1,"stroke-dasharray":"2 4",fill:"none"},s);
  chs.forEach((c,i)=>{
    const y0=top+row*i+row/2,a=T[c];let d="";
    a.forEach((v,j)=>{const vv=Math.max(-320,Math.min(320,v));d+=(j?"L":"M")+(L+j/(a.length-1)*W).toFixed(1)+" "+(y0-vv*0.095).toFixed(1)});
    el("path",{d,fill:"none",stroke:BONE,"stroke-width":.8,"stroke-linejoin":"round",opacity:.9},s);
    el("text",{x:L-14,y:y0+4,class:"sv-c","text-anchor":"end"},s,c);
  });
  // window strip
  const wy=top+row*chs.length+30;
  T.y.forEach((lab,i)=>{const x=X(t0+i*4);el("rect",{x:x+1,y:wy,width:X(t0+4)-X(t0)-2,height:14,fill:lab?EMBER:"none",stroke:lab?EMBER:MOSS,"stroke-width":1,opacity:lab?.9:.6},s)});
  el("text",{x:L-14,y:wy+11,class:"sv-c","text-anchor":"end"},s,"4 s slices");
  [2980,3000,3020,3040,3060].forEach(t=>{el("text",{x:X(t),y:wy+40,class:"sv-c","text-anchor":"middle"},s,t.toLocaleString()+" s")});
  el("text",{x:X(2996)+8,y:top+4,fill:EMBER,style:"font:500 11px Instrument Sans;letter-spacing:.2em"},s,"ONSET 2,996 s");
  el("text",{x:X(3036)+8,y:top+4,fill:EMBER,style:"font:500 11px Instrument Sans;letter-spacing:.2em"},s,"END 3,036 s");
})();

/* ---- pipeline ---- */
(function(){
  const s=document.getElementById("pipe"),b=[["Raw .edf","256 Hz · 23 ch"],["Keep 18","shared channels"],["Band-pass","0.5–40 Hz"],["Downsample","to 128 Hz"],["Flag","flat · extreme"]];
  const w=128,gap=30;
  b.forEach((x,i)=>{const X=i*(w+gap);el("rect",{x:X,y:40,width:w,height:70,rx:2,class:i===0?"sv-m":"sv-l"},s);
    el("text",{x:X+14,y:70,class:"sv-t","font-size":14},s,x[0]);el("text",{x:X+14,y:92,class:"sv-s","font-size":12},s,x[1]);
    el("text",{x:X,y:24,class:"sv-k"},s,"0"+(i+1));
    if(i<b.length-1){el("path",{d:`M${X+w} 75 H${X+w+gap-6}`,class:"sv-l"},s);el("path",{d:`M${X+w+gap-10} 71 l4 4 l-4 4`,class:"sv-l"},s)}});
})();

/* ---- spectrum ---- */
(function(){
  const s=document.getElementById("psd"),F=D.filt,L=80,W=1160,H=300,T=20,ymin=-80,ymax=35;
  const X=f=>L+f/100*W,Y=v=>T+(ymax-Math.max(ymin,Math.min(ymax,v)))/(ymax-ymin)*H;
  el("rect",{x:X(.5),y:T,width:X(40)-X(.5),height:H,fill:CAN},s);
  el("text",{x:X(2),y:T+20,class:"sv-k"},s,"KEPT · 0.5–40 HZ");
  el("text",{x:X(42),y:T+20,class:"sv-k"},s,"REMOVED");
  [0,20,40,60,80,100].forEach(f=>{el("line",{x1:X(f),x2:X(f),y1:T+H,y2:T+H+6,stroke:MOSS},s);el("text",{x:X(f),y:T+H+24,class:"sv-c","text-anchor":"middle"},s,f+" Hz")});
  [-60,-30,0,30].forEach(v=>{el("line",{x1:L,x2:L+W,y1:Y(v),y2:Y(v),stroke:MOSS,opacity:.15},s);el("text",{x:L-12,y:Y(v)+4,class:"sv-c","text-anchor":"end"},s,v+" dB")});
  const path=a=>F.psd_f.map((f,i)=>(i?"L":"M")+X(f).toFixed(1)+" "+Y(a[i]).toFixed(1)).join("");
  el("path",{d:path(F.psd_raw),fill:"none",stroke:MOSS,"stroke-width":1.2,"stroke-dasharray":"4 3"},s);
  el("path",{d:path(F.psd_clean),fill:"none",stroke:BONE,"stroke-width":1.4},s);
  el("path",{d:`M${X(60)} ${Y(1.9)-6} V${Y(1.9)-40} H${X(60)+30}`,class:"sv-m"},s);
  el("text",{x:X(60)+38,y:Y(1.9)-36,class:"sv-t","font-size":14},s,"60 Hz: power-line hum");
})();

/* ---- cut into windows (diagram) ---- */
(function(){
  const s=document.getElementById("cut"),L=20,W=720,n=10,w=W/n;let d="";
  for(let i=0;i<=360;i++){const x=L+i/360*W,t=i/360,amp=(t>.38&&t<.72)?22:7;d+=(i?"L":"M")+x.toFixed(1)+" "+(70+Math.sin(i*1.7)*amp*Math.sin(i*.31+1)).toFixed(1)}
  const sz0=L+.38*W,sz1=L+.72*W;
  el("rect",{x:sz0,y:20,width:sz1-sz0,height:8,fill:EMBER},s);
  el("text",{x:sz0,y:12,fill:EMBER,style:"font:500 11px Instrument Sans;letter-spacing:.2em"},s,"DOCTORS' MARK");
  el("path",{d,fill:"none",stroke:BONE,"stroke-width":1},s);
  for(let i=0;i<n;i++){const x0=L+i*w,x1=x0+w,ov=Math.max(0,Math.min(x1,sz1)-Math.max(x0,sz0))/w,lab=ov>=.5;
    el("rect",{x:x0+2,y:120,width:w-4,height:28,fill:lab?EMBER:"none",stroke:lab?EMBER:MOSS,opacity:lab?.9:.7},s);
    el("text",{x:x0+w/2,y:139,"text-anchor":"middle",style:`font:11px ui-monospace,monospace;fill:${lab?"#0F1A15":MOSS}`},s,(ov*100|0)+"%");
    el("line",{x1:x0,x2:x0,y1:40,y2:112,stroke:MOSS,opacity:.35,"stroke-dasharray":"2 3"},s)}
  el("text",{x:L,y:170,class:"sv-k"},s,"SLICES · % INSIDE THE MARK");
  for(let i=0;i<n-1;i++){const x0=L+i*w+w/2,ov=Math.max(0,Math.min(x0+w,sz1)-Math.max(x0,sz0))/w;if(ov>=.5)el("rect",{x:x0+2,y:186,width:w-4,height:16,fill:"none",stroke:EMBER,"stroke-dasharray":"3 3"},s)}
  el("text",{x:L,y:224,class:"sv-k"},s,"SHIFTED 2 s · TRAINING ONLY");
})();

/* ---- dots & bars ---- */
const dots=document.getElementById("dots");for(let i=0;i<236;i++){const e=document.createElement("i");if(i===171)e.className="e";dots.appendChild(e)}
const split={chb03:"exam",chb10:"exam",chb20:"exam",chb05:"tune",chb23:"tune"};
const mx=Math.max(...ids.map(p=>pp[p][1])),bars=document.getElementById("bars");
ids.forEach(p=>{const [w,z]=pp[p];const li=document.createElement("li");
  li.innerHTML=`<span class="id">${p}</span><span class="sp">${split[p]||"train"}</span><span class="bar"><i style="width:${z/mx*100}%"></i></span><span class="v">${z}</span>`;
  li.title=`${p}: ${w.toLocaleString()} slices, ${(w*4/3600).toFixed(0)} hours, ${z} seizure slices`;bars.appendChild(li)});

/* ---- split board ---- */
const fx={train:["chb01","chb06","chb08","chb13","chb14","chb15","chb16","chb18","chb24"],val:["chb05","chb23"],test:["chb03","chb10","chb20"]};
for(const k in fx){const b=document.getElementById("b-"+k);fx[k].forEach(p=>{const e=document.createElement("span");e.textContent=p;b.appendChild(e)})}

/* ---- LOPO grid ---- */
(function(){
  const s=document.getElementById("lopo"),L=110,cw=76,ch=16,g=3;
  ids.forEach((p,j)=>el("text",{x:L+j*cw+cw/2,y:12,class:"sv-c","text-anchor":"middle"},s,p));
  ids.forEach((p,i)=>{el("text",{x:L-14,y:30+i*(ch+g)+12,class:"sv-c","text-anchor":"end"},s,"round "+(i+1));
    ids.forEach((q,j)=>el("rect",{x:L+j*cw+2,y:30+i*(ch+g),width:cw-4,height:ch,fill:i===j?BONE:"none",stroke:i===j?BONE:MOSS,opacity:i===j?1:.35},s))});
})();

/* ---- question flow ---- */
(function(){
  const s=document.getElementById("flow");
  // slice
  for(let r=0;r<6;r++){let d="";for(let i=0;i<=60;i++)d+=(i?"L":"M")+(20+i*2.6).toFixed(1)+" "+(40+r*20+Math.sin(i*.9+r)*5*Math.sin(i*.2+r*2)).toFixed(1);el("path",{d,fill:"none",stroke:BONE,"stroke-width":.8},s)}
  el("rect",{x:10,y:24,width:176,height:132,fill:"none",stroke:MOSS},s);
  el("text",{x:10,y:180,class:"sv-c"},s,"18 × 512 · one 4 s slice");
  const arrow=(x1,x2,y=90)=>{el("path",{d:`M${x1} ${y} H${x2}`,class:"sv-l"},s);el("path",{d:`M${x2-5} ${y-4} l5 4 l-5 4`,class:"sv-l"},s)};
  arrow(196,286);
  el("rect",{x:290,y:40,width:200,height:100,rx:50,class:"sv-l"},s);
  el("text",{x:390,y:86,class:"sv-t","text-anchor":"middle"},s,"Model");
  el("text",{x:390,y:106,class:"sv-s","text-anchor":"middle"},s,"yardstick or wearable");
  arrow(500,590);
  // probability gauge
  el("line",{x1:600,x2:860,y1:90,y2:90,stroke:MOSS},s);
  [0,.5,1].forEach(v=>el("text",{x:600+v*260,y:116,class:"sv-c","text-anchor":"middle"},s,v.toFixed(1)));
  el("line",{x1:600+.62*260,x2:600+.62*260,y1:70,y2:110,stroke:BONE,"stroke-dasharray":"2 3"},s);
  el("text",{x:600+.62*260,y:62,class:"sv-c","text-anchor":"middle"},s,"threshold");
  el("circle",{cx:600+.81*260,cy:90,r:6,fill:EMBER},s);
  el("text",{x:600,y:150,class:"sv-s"},s,"P(seizure) = 0.81");
  arrow(880,960);
  // 2 of 3
  [0,1,2].forEach(i=>el("rect",{x:970+i*44,y:72,width:38,height:36,fill:i<2?EMBER:"none",stroke:i<2?EMBER:MOSS},s));
  el("text",{x:970,y:136,class:"sv-s"},s,"2 of 3 in a row");
  arrow(1110,1170);
  el("circle",{cx:1200,cy:90,r:24,fill:"none",stroke:EMBER},s);el("circle",{cx:1200,cy:90,r:6,fill:EMBER},s);
  el("text",{x:1200,y:136,class:"sv-s","text-anchor":"middle"},s,"alarm");
})();

/* ---- architecture sketches ---- */
(function(){
  const c=document.getElementById("arch-cnn");
  [[0,96],[90,80],[170,64],[240,48]].forEach(([x,h],i)=>{for(let k=0;k<3;k++)el("rect",{x:x+k*8,y:70-h/2-k*6,width:56,height:h,fill:"#0F1A15",stroke:k===2?BONE:MOSS},c);el("text",{x:x+8,y:142,class:"sv-c"},c,"conv "+(i+1))});
  el("path",{d:"M320 64 H360",class:"sv-l"},c);el("rect",{x:366,y:52,width:60,height:24,fill:"none",stroke:BONE},c);el("text",{x:366,y:142,class:"sv-c"},c,"pool");
  el("path",{d:"M426 64 H470",class:"sv-l"},c);el("circle",{cx:490,cy:64,r:16,fill:"none",stroke:BONE},c);el("text",{x:476,y:142,class:"sv-c"},c,"P(seizure)");
  const e=document.getElementById("arch-eeg");
  el("rect",{x:0,y:20,width:120,height:90,fill:"none",stroke:BONE},e);for(let k=0;k<4;k++){let d="";for(let i=0;i<=40;i++)d+=(i?"L":"M")+(8+i*2.6)+" "+(38+k*20+Math.sin(i*(.5+k*.35))*5);el("path",{d,fill:"none",stroke:MOSS,"stroke-width":.8},e)}
  el("text",{x:0,y:142,class:"sv-c"},e,"rhythms in time");
  el("path",{d:"M120 65 H160",class:"sv-l"},e);
  for(let r=0;r<5;r++)for(let q=0;q<5;q++)el("circle",{cx:180+q*14,cy:38+r*14,r:3,fill:(r+q)%3===0?BONE:"none",stroke:MOSS},e);
  el("text",{x:166,y:142,class:"sv-c"},e,"which electrodes");
  el("path",{d:"M250 65 H300",class:"sv-l"},e);
  el("rect",{x:306,y:45,width:70,height:40,fill:"none",stroke:BONE},e);el("rect",{x:316,y:55,width:50,height:20,fill:"none",stroke:MOSS},e);
  el("text",{x:306,y:142,class:"sv-c"},e,"combine");
  el("path",{d:"M376 65 H470",class:"sv-l"},e);el("circle",{cx:490,cy:65,r:16,fill:"none",stroke:BONE},e);el("text",{x:476,y:142,class:"sv-c"},e,"P(seizure)");
})();
