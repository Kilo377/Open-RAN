const data=JSON.parse(document.getElementById('replay-data').textContent);
const $=id=>document.getElementById(id),frames=data.frames,totalTime=data.meta.simulation_s;
const REDUCE=matchMedia('(prefers-reduced-motion:reduce)');
const color=data.manifest.cells.map((_,i)=>['#137d70','#b17217','#7862a3'][i]||`hsl(${(i*137.5)%360},52%,42%)`);
let time=0,selected=0,playing=!REDUCE.matches,speed=data.meta.playback_speed,frameIndex=-1,previousTimestamp=0,dirty=true;
let focused=false,linksVisible=true,sceneReady=false;
const fmt=(x,n=1)=>x==null?'N/A':Number(x).toFixed(n);
const pct=(bad,n)=>n?`${(100*bad/n).toFixed(2)}%`:'N/A';
const integer=n=>n.toLocaleString('en-US');
const sumCounts=rows=>rows.reduce((a,row)=>a.map((v,i)=>v+row[i]),[0,0,0,0]);
const cells=data.manifest.cells;
const legend=document.querySelector('.legend');legend.replaceChildren();cells.forEach((c,i)=>{const entry=document.createElement('span'),swatch=document.createElement('i');swatch.className='swatch';swatch.style.background=color[i];entry.append(swatch,document.createTextNode(c.cell_id.replace(/^cell_/,'Cell ')));legend.append(entry)});
const ueLabel=i=>data.users[i].id.replace(/^ue_/, 'UE ');
const windowMs=data.meta.window_ms;
const timeDivisor=totalTime<1?.001:totalTime<120?1:totalTime<7200?60:3600;
const axisTime=t=>t/timeDivisor;
const axisUnit=totalTime<1?'ms':totalTime<120?'s':totalTime<7200?'min':'h';
const formatTime=t=>totalTime<1?`${fmt(t*1000,3)} ms`:totalTime<120?`${fmt(t,3)} s`:`${Math.floor(t/3600)?Math.floor(t/3600)+':':''}${String(Math.floor(t/60)%60).padStart(2,'0')}:${(t%60).toFixed(3).padStart(6,'0')}`;
const cellLabel=i=>data.manifest.cells[i].cell_id.replace(/^cell_/, 'Cell ');
function indexAt(t){let lo=0,hi=frames.length;while(lo<hi){const mid=(lo+hi)>>1;if(frames[mid].t<=t+1e-10)lo=mid+1;else hi=mid}return Math.max(0,lo-1)}
const view=$('view'),labelLayer=$('labels');
let renderer,scene,camera,building,cameraDirty=true;
const ueObjects=[],baseObjects=[],linkObjects=[],labels=[];
const target=new THREE.Vector3(...data.view.center);
let yaw=.8,pitch=.85,distance=data.view.distance,selectedRing;
const vtemp=new THREE.Vector3();
function position(i,t){
 const tr=data.manifest.ues[i].trajectory;let j=0;
 while(j<tr.times_s.length-2&&t>tr.times_s[j+1])j++;
 const a=tr.positions_m[j],b=tr.positions_m[j+1]||a;
 const f=Math.max(0,Math.min(1,(t-tr.times_s[j])/(tr.times_s[j+1]-tr.times_s[j]||1)));
 return new THREE.Vector3(a[0]+(b[0]-a[0])*f,a[2]+(b[2]-a[2])*f,-a[1]-(b[1]-a[1])*f);
}
function world(p){return new THREE.Vector3(p[0],p[2],-p[1])}
function buffer(s){return Uint8Array.from(atob(s),c=>c.charCodeAt(0)).buffer}
function label(text,object,ue=false){const el=document.createElement('span');el.className='pin'+(ue?' ue-pin':'');el.textContent=text;labelLayer.append(el);labels.push({el,object,ue})}
function cameraUpdate(){camera.position.set(target.x+distance*Math.cos(pitch)*Math.sin(yaw),target.y+distance*Math.sin(pitch),target.z+distance*Math.cos(pitch)*Math.cos(yaw));camera.lookAt(target);camera.updateMatrixWorld();cameraDirty=true;dirty=true}
function initScene(){
 renderer=new THREE.WebGLRenderer({antialias:true,alpha:false,preserveDrawingBuffer:true});
 renderer.setPixelRatio(Math.min(devicePixelRatio||1,2));renderer.setClearColor('#dce5e1');
 renderer.domElement.setAttribute('aria-label','Interactive 3D network replay');
 view.prepend(renderer.domElement);scene=new THREE.Scene();camera=new THREE.PerspectiveCamera(42,1,.1,Math.max(2200,data.view.distance*10));
 scene.add(new THREE.HemisphereLight('#f3faf6','#87998e',2.1));
 const sunlight=new THREE.DirectionalLight('#fff9eb',2.2);sunlight.position.set(70,220,100);scene.add(sunlight);
 const raw=data.mesh.vertex_bits===32?new Int32Array(buffer(data.mesh.vertices)):new Int16Array(buffer(data.mesh.vertices)),vertices=new Float32Array(raw.length);
 for(let i=0;i<raw.length;i+=3){vertices[i]=raw[i]*data.mesh.quantization_m;vertices[i+1]=raw[i+2]*data.mesh.quantization_m;vertices[i+2]=-raw[i+1]*data.mesh.quantization_m}
 const geom=new THREE.BufferGeometry();geom.setAttribute('position',new THREE.BufferAttribute(vertices,3));
 const indices=data.mesh.index_bits===16?new Uint16Array(buffer(data.mesh.indices)):new Uint32Array(buffer(data.mesh.indices));
 geom.setIndex(new THREE.BufferAttribute(indices,1));geom.computeVertexNormals();
 building=new THREE.Mesh(geom,new THREE.MeshStandardMaterial({color:'#a9bbb1',roughness:.94,metalness:0,side:THREE.DoubleSide}));scene.add(building);
 const edges=new THREE.LineSegments(new THREE.EdgesGeometry(geom,35),new THREE.LineBasicMaterial({color:'#677e71',transparent:true,opacity:.3}));scene.add(edges);
 const ground=new THREE.Mesh(new THREE.PlaneGeometry(data.view.ground_size,data.view.ground_size),new THREE.MeshStandardMaterial({color:'#d6e0da',roughness:1}));ground.rotation.x=-Math.PI/2;ground.position.set(data.view.center[0],-.08,data.view.center[2]);scene.add(ground);
 const grid=new THREE.GridHelper(data.view.ground_size,20,'#becdc3','#c8d4cd');grid.position.set(data.view.center[0],-.04,data.view.center[2]);scene.add(grid);
 cells.forEach((c,i)=>{
  const p=world(c.position_m),group=new THREE.Group();group.position.copy(p);
  const base=new THREE.Mesh(new THREE.CylinderGeometry(1.4,1.4,5,10),new THREE.MeshBasicMaterial({color:color[i],depthTest:false}));base.renderOrder=3;group.add(base);scene.add(group);baseObjects.push(group);label(cellLabel(i),group);
  const mast=new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(p.x,0,p.z),p]),new THREE.LineBasicMaterial({color:color[i],transparent:true,opacity:.65}));scene.add(mast);
 });
 data.users.forEach((u,i)=>{
  const point=new THREE.Mesh(new THREE.SphereGeometry(.95,14,10),new THREE.MeshBasicMaterial({color:color[frames[0].users[i][0]],depthTest:false,transparent:true}));point.renderOrder=5;point.userData.ue=i;scene.add(point);ueObjects.push(point);
  label(ueLabel(i),point,true);
  const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(new Float32Array(6),3));
  const line=new THREE.Line(geometry,new THREE.LineBasicMaterial({color:color[frames[0].users[i][0]],transparent:true,opacity:.2,depthTest:false}));line.renderOrder=1;scene.add(line);linkObjects.push(line);
  const trail=new THREE.Line(new THREE.BufferGeometry().setFromPoints(data.manifest.ues[i].trajectory.positions_m.map(world)),new THREE.LineBasicMaterial({color:color[frames[0].users[i][0]],depthTest:false,transparent:true,opacity:.6}));trail.renderOrder=4;scene.add(trail);
 });
 selectedRing=new THREE.Mesh(new THREE.TorusGeometry(1.7,.16,6,32),new THREE.MeshBasicMaterial({color:'#20352e',depthTest:false}));selectedRing.rotation.x=Math.PI/2;selectedRing.renderOrder=6;scene.add(selectedRing);
 sceneReady=true;cameraUpdate();$('scene-notice').hidden=true;resizeScene();
 renderer.domElement.addEventListener('webglcontextlost',e=>{e.preventDefault();sceneReady=false;$('scene-notice').textContent='3D context was lost. Reload this file to restore the scene. Recorded metrics remain available.';$('scene-notice').hidden=false});
 setupCamera();
}
function resizeScene(){if(!sceneReady)return;const w=view.clientWidth,h=view.clientHeight;if(!w||!h)return;renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();cameraDirty=true;dirty=true}
function setupCamera(){
 let last=null,moved=0,start=null;const canvas=renderer.domElement;
 canvas.addEventListener('pointerdown',e=>{last={x:e.clientX,y:e.clientY};start={...last};moved=0;canvas.setPointerCapture(e.pointerId)});
 canvas.addEventListener('pointermove',e=>{if(!last)return;const dx=e.clientX-last.x,dy=e.clientY-last.y;moved+=Math.abs(dx)+Math.abs(dy);last={x:e.clientX,y:e.clientY};
  if(e.shiftKey||e.button===2){const scale=distance/view.clientHeight*.6;target.x-=dx*Math.cos(yaw)*scale;target.z+=dx*Math.sin(yaw)*scale;target.y+=dy*scale}
  else{yaw-=dx*.007;pitch=Math.max(.18,Math.min(1.53,pitch+dy*.006))}focused=false;cameraUpdate();
 });
 canvas.addEventListener('pointerup',e=>{if(start&&moved<6){const r=canvas.getBoundingClientRect();const pointer=new THREE.Vector2((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1);const ray=new THREE.Raycaster();ray.setFromCamera(pointer,camera);const hits=ray.intersectObjects(ueObjects);if(hits.length)selectUE(hits[0].object.userData.ue)}last=null;start=null});
 canvas.addEventListener('pointercancel',()=>{last=null;start=null});
 canvas.addEventListener('wheel',e=>{e.preventDefault();distance=Math.max(18,Math.min(data.view.distance*3,distance*Math.exp(e.deltaY*.001)));cameraUpdate()},{passive:false});
 canvas.addEventListener('contextmenu',e=>e.preventDefault());
}
function updateScene(){if(!sceneReady)return;const f=frames[Math.max(0,frameIndex)];
 ueObjects.forEach((obj,i)=>{obj.position.copy(position(i,time));const ci=f.users[i][0];obj.material.color.set(color[ci]);obj.material.opacity=f.connections[i]==='connected'?1:.35;const line=linkObjects[i],a=baseObjects[ci].position,b=obj.position,attr=line.geometry.attributes.position;attr.setXYZ(0,a.x,a.y,a.z);attr.setXYZ(1,b.x,b.y,b.z);attr.needsUpdate=true;line.geometry.computeBoundingSphere();line.material.color.set(color[ci]);line.material.opacity=i===selected?.85:.14;line.visible=linksVisible&&f.connections[i]==='connected'});
 selectedRing.position.copy(ueObjects[selected].position);
 if(focused){const p=ueObjects[selected].position;target.copy(p);target.y+=5;cameraUpdate()}
 labels.forEach(({el,object,ue},i)=>{if(ue&&object.userData.ue!==selected){el.hidden=true;return}const p=object.position.clone();p.y+=ue?3:5;p.project(camera);el.hidden=p.z>1||p.z< -1;if(!el.hidden)el.style.transform=`translate(${(p.x+1)*view.clientWidth/2}px,${(1-p.y)*view.clientHeight/2}px) translate(-50%,-100%)`});
 renderer.render(scene,camera);cameraDirty=false;
}
const chartDefs={};
const NS='http://www.w3.org/2000/svg';
function svgEl(tag,attrs,text){const n=document.createElementNS(NS,tag);Object.entries(attrs||{}).forEach(([k,v])=>n.setAttribute(k,v));if(text!=null)n.textContent=text;return n}
function plotSpec(kind){
 if(kind==='goodput')return {unit:'Mbit/s',title:'Network throughput',series:[{values:frames.map(f=>f.g),color:'var(--accent)'},{values:frames.map(f=>f.offered_mbps),color:'var(--muted)',dash:'5 4'}]};
 const metric=$('detail-metric').value;
 if(metric==='completion-delay')return {unit:'ms',title:`${ueLabel(selected)} · Packet latency`,scope:`Cumulative completed-packet mean · Budget ${data.users[selected].budget_ms} ms`,budget:data.users[selected].budget_ms,series:[{values:frames.map(f=>f.users[selected][9]??0),color:'var(--accent)'}]};
 if(metric==='energy')return {unit:'J',title:'RF transmit energy',scope:'RF output only',series:[{values:frames.map(f=>f.rf_energy_j),color:'var(--accent)'}]};
 if(metric==='power')return {unit:'W',title:'RF transmit power',scope:'RF output only',series:[{values:frames.map(f=>f.rf_power_w),color:'var(--accent)'}]};
 if(metric==='ue-throughput')return {unit:'Mbit/s',title:`${ueLabel(selected)} · Throughput`,scope:`Delivered payload · ${windowMs} ms`,series:[{values:frames.map(f=>f.users[selected][3]),color:'var(--accent)'}]};
 return {unit:'ms',title:`${ueLabel(selected)} · HOL delay`,scope:`${windowMs} ms maxima · Budget ${data.users[selected].budget_ms} ms`,budget:data.users[selected].budget_ms,series:[{values:frames.map(f=>f.hol_max_ms[selected]),color:'var(--accent)'}]};
}
function drawChart(kind){
 const host=$(kind==='goodput'?'goodput-chart':'hol-chart'),w=host.clientWidth,h=185;
 if(w<1)return;const m={l:48,r:18,t:24,b:31},pw=w-m.l-m.r,ph=h-m.t-m.b;
 const spec=plotSpec(kind),values=spec.series[0].values,budget=spec.budget;
 const observedMax=Math.max(0,...spec.series.flatMap(s=>s.values));const max=observedMax>0?observedMax:1,base=10**Math.floor(Math.log10(max)),step=base*(max/base<=2?.5:max/base<=5?1:2),upper=Math.ceil(max*1.05/step)*step;
 const x=t=>m.l+t/totalTime*pw,y=v=>m.t+ph-v/(upper||1)*ph;
 const svg=svgEl('svg',{viewBox:`0 0 ${w} ${h}`,role:'img','aria-label':`${spec.title}, ${spec.unit}, over simulation ${axisUnit}`});
 svg.append(svgEl('rect',{x:x(0),y:m.t,width:x(data.meta.warmup_s)-x(0),height:ph,fill:'var(--panel)'}));
 for(let k=0;k<=3;k++){const v=upper*k/3;svg.append(svgEl('line',{x1:m.l,x2:w-m.r,y1:y(v),y2:y(v),class:'grid'}));svg.append(svgEl('text',{x:m.l-8,y:y(v)+4,'text-anchor':'end'},fmt(v,upper<10?1:0)))}
 for(let k=0;k<=4;k++){const t=k*totalTime/4;svg.append(svgEl('text',{x:x(t),y:h-10,'text-anchor':k===4?'end':'middle'},fmt(axisTime(t),totalTime<1?1:2)+(k===4?' '+axisUnit:'' )))}
 svg.append(svgEl('text',{x:m.l,y:13},spec.unit));
 if(budget!=null&&budget<=upper){svg.append(svgEl('line',{x1:x(0),x2:x(totalTime),y1:y(budget),y2:y(budget),class:'budget'}))}
 spec.series.forEach(s=>svg.append(svgEl('path',{d:frames.map((f,i)=>`${i?'L':'M'}${x(f.t).toFixed(2)},${y(s.values[i]).toFixed(2)}`).join(' '),class:'series',style:`stroke:${s.color}`,'stroke-dasharray':s.dash||''})));
 const cursor=svgEl('line',{x1:x(time),x2:x(time),y1:m.t,y2:m.t+ph,class:'cursor'}),marker=svgEl('circle',{cx:x(time),cy:y(values[Math.max(0,frameIndex)]),r:3.2,class:'marker'});svg.append(cursor,marker);
 const hit=svgEl('rect',{x:m.l,y:m.t,width:pw,height:ph,fill:'transparent'});svg.append(hit);
 function seek(e){const r=svg.getBoundingClientRect();setTime(Math.max(0,Math.min(totalTime,((e.clientX-r.left)*w/r.width-m.l)/pw*totalTime)));playing=false;updatePlay()}
 hit.addEventListener('pointerdown',seek);host.replaceChildren(svg);chartDefs[kind]={cursor,marker,x,y,values,spec};
 if(kind!=='goodput'){$('hol-title').textContent=spec.title;$('detail-scope').textContent=spec.scope;$('detail-scope').title=spec.scope==='RF output only'?data.energy_scope:$('detail-metric').value==='completion-delay'?'Completed-packet mean; pending and dropped packets are excluded.':'HOL is the waiting time of the oldest queued packet.'}
}
function updateCharts(){for(const c of Object.values(chartDefs)){c.cursor.setAttribute('x1',c.x(time));c.cursor.setAttribute('x2',c.x(time));c.marker.setAttribute('cx',c.x(frames[Math.max(0,frameIndex)].t));c.marker.setAttribute('cy',c.y(c.values[Math.max(0,frameIndex)]))}const c=chartDefs.hol;if(c)$('hol-chart-value').textContent=`${fmt(c.values[Math.max(0,frameIndex)],2)} ${c.spec.unit}`}
function selectUE(i){selected=i;document.querySelectorAll('#ue-grid button').forEach((b,j)=>b.setAttribute('aria-pressed',j===i));$('ue-name').textContent=ueLabel(i);$('ue-slice').textContent=data.users[i].slice;drawChart('hol');frameIndex=-1;dirty=true}
data.users.forEach((u,i)=>{const b=document.createElement('button');b.type='button';b.textContent=ueLabel(i);b.setAttribute('aria-pressed',i===0);b.addEventListener('click',()=>selectUE(i));$('ue-grid').append(b)});
const ueFields=['Serving cell',`Goodput · ${windowMs} ms`,'Mean SINR / last MCS','Pending / HOL delay','Mean packet latency / budget','Deadline misses · cumulative','First-TX errors / attempts','Retx errors / attempts','Overflow · cumulative'];
const details=ueFields.map(text=>{const row=document.createElement('div');row.className='pair';const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=text;row.append(dt,dd);$('ue-stats').append(row);return dd});
const cellRows=cells.map((c,i)=>{const row=document.createElement('tr');row.innerHTML=`<td><i class="swatch" style="background:${color[i]}"></i><span></span></td><td></td><td></td><td></td><td></td>`;row.querySelector('span').textContent=cellLabel(i);$('cell-rows').append(row);return row});
function updateMetrics(){const i=indexAt(time);if(i===frameIndex)return;frameIndex=i;const f=frames[i],n=sumCounts(f.link),u=f.users[selected],c=f.link[selected];
 document.querySelectorAll('.window-basis').forEach(el=>el.textContent=`${fmt(f.window_ms||windowMs,(f.window_ms||windowMs)%1?2:0)} ms window`);
 if(data.meta.xapp_bindings_recorded){const active=[...new Set(Object.values(f.xapps||{}).flatMap(Object.values))];$('xapps-state').textContent=active.length?`${active.length} active`:'Off';$('xapps-state').title=active.join(', ')}
 $('net-goodput').textContent=fmt(f.g,2);$('net-first').textContent=n[0]?fmt(100*n[1]/n[0],2):'N/A';$('net-retx').textContent=n[2]?fmt(100*n[3]/n[2],2):'N/A';
 $('net-first-n').textContent=`${integer(n[1])} / ${integer(n[0])}`;$('net-retx-n').textContent=`${integer(n[3])} / ${integer(n[2])}`;
 $('net-overflow').textContent=fmt(f.overflow/1e6,2);$('net-queue').textContent=fmt(f.q/1e3,1);$('kpi-time').textContent=`t = ${formatTime(f.t)}`;
 $('net-rf-power').textContent=fmt(f.rf_power_w,2);$('net-rf-energy').textContent=fmt(f.rf_energy_j,2);
 $('net-completion-delay').textContent=fmt(f.cdelay,2);$('net-deadline-misses').textContent=integer(f.deadline_misses);
 details[0].textContent=cellLabel(u[0]);details[1].textContent=`${fmt(u[3],3)} Mbit/s`;
 details[2].textContent=`${u[12]?'No path':fmt(u[4],1)+' dB'} / ${u[5]??'N/A'}`;details[2].title='No path denotes a recorded serving-link outage; the MCS value is the last valid scheduled value.';details[3].textContent=`${fmt(u[1]/1e3,1)} kB / ${fmt(u[2],1)} ms`;
 details[4].textContent=`${fmt(u[9],2)} / ${data.users[selected].budget_ms} ms`;details[4].title='Cumulative mean latency of completed packets; pending packets are excluded. HOL is the waiting time of the oldest queued packet.';
 details[5].textContent=integer(u[10]);details[5].title='Completed-late packets plus currently overdue pending packets; excludes buffer overflow.';
 details[6].textContent=`${integer(c[1])} / ${integer(c[0])} (${pct(c[1],c[0])})`;details[7].textContent=`${integer(c[3])} / ${integer(c[2])} (${pct(c[3],c[2])})`;details[8].textContent=`${fmt(u[8]/1e6,3)} MB`;
 cellRows.forEach((r,j)=>{const d=r.children;d[1].textContent=f.users.filter((u,k)=>u[0]===j&&f.connections[k]==='connected').length;d[2].textContent=fmt(f.cells[j][0],2);d[3].textContent=`${fmt(f.cells[j][1]*100,1)}%`;d[4].textContent=fmt(f.cells[j][2],2)});
 $('goodput-chart-value').textContent=`${fmt(f.g,2)} Mbit/s`;updateCharts();
}
function updateClock(){$('clock').textContent=formatTime(time);$('slot').textContent=Math.min(data.meta.slots,Math.floor((time+1e-10)/(data.meta.slot_ms/1000)));$('seek').value=Math.round(time*1000/data.meta.slot_ms)*data.meta.slot_ms}
function updatePlay(){$('play').textContent=playing?'Pause':'Play';$('play').setAttribute('aria-label',playing?'Pause replay':'Play replay')}
function setTime(t){time=Math.max(0,Math.min(totalTime,t));dirty=true;previousTimestamp=0;updateMetrics();updateClock()}
function overview(){focused=false;target.set(...data.view.center);yaw=.8;pitch=.85;distance=data.view.distance;if(sceneReady)cameraUpdate()}
function focus(){focused=true;distance=65;pitch=.7;if(sceneReady){target.copy(position(selected,time));target.y+=5;cameraUpdate()}}
function links(){linksVisible=!linksVisible;$('links').setAttribute('aria-pressed',linksVisible);dirty=true}
function playToggle(){if(playing)playing=false;else{if(time>=totalTime)setTime(0);playing=true}previousTimestamp=0;updatePlay()}
function restart(){setTime(0);playing=!REDUCE.matches;updatePlay()}
$('play').addEventListener('click',playToggle);$('restart').addEventListener('click',restart);$('seek').addEventListener('input',e=>{playing=false;setTime(Number(e.target.value)/1000);updatePlay()});$('speed').addEventListener('change',e=>{speed=Number(e.target.value);previousTimestamp=0});$('reset-view').addEventListener('click',overview);$('focus-ue').addEventListener('click',focus);$('links').addEventListener('click',links);
$('detail-metric').addEventListener('change',()=>{drawChart('hol');updateCharts();dirty=true});
document.addEventListener('keydown',e=>{if(e.code==='Space'&&!['INPUT','SELECT','BUTTON','SUMMARY'].includes(document.activeElement.tagName)){e.preventDefault();playToggle()}});
REDUCE.addEventListener('change',()=>{if(REDUCE.matches){playing=false;updatePlay()}});
document.addEventListener('visibilitychange',()=>{previousTimestamp=0});
let resizeQueued=false;function queueResize(){if(resizeQueued)return;resizeQueued=true;requestAnimationFrame(()=>{resizeQueued=false;resizeScene();drawChart('goodput');drawChart('hol');dirty=true})}
// Defer size mutations outside ResizeObserver delivery and ignore unchanged widths.
const observedSizes=new WeakMap();const observer=new ResizeObserver(entries=>{let changed=false;for(const entry of entries){const key=`${Math.round(entry.contentRect.width)}:${Math.round(entry.contentRect.height)}`;if(observedSizes.get(entry.target)!==key){observedSizes.set(entry.target,key);changed=true}}if(changed)queueResize()});
observer.observe(view);observer.observe($('goodput-chart'));observer.observe($('hol-chart'));
try{initScene()}catch(error){$('scene-notice').textContent='3D rendering needs a WebGL-enabled browser. The recorded KPIs and playback controls remain available.';console.error('3D initialization:',error)}
$('avg-goodput').textContent=(data.totals.delivered_bytes*8/totalTime/1e6).toFixed(2);
selectUE(0);drawChart('goodput');updateMetrics();updateClock();updatePlay();
function tick(timestamp){
 if(playing&&!document.hidden){if(previousTimestamp)time=Math.min(totalTime,time+Math.min(.15,(timestamp-previousTimestamp)/1000)*speed);previousTimestamp=timestamp;dirty=true;if(time>=totalTime){playing=false;updatePlay()}}
 else previousTimestamp=0;
 if(dirty){updateMetrics();updateClock();updateCharts();updateScene();dirty=false}
 requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
// Read-only diagnostics for reproducibility and export validation.
window.replayDiagnostics=()=>({time,frameIndex,selected,playing,sceneReady,renderCalls:renderer?.info.render.calls,triangles:renderer?.info.render.triangles,positions:ueObjects.map(u=>u.position.toArray()),camera:camera?.position.toArray(),source:data.meta.source_sha256,ueIds:data.users.map(u=>u.id),windowMs});
