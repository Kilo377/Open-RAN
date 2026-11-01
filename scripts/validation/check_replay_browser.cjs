// Optional development check: NODE_PATH must contain Playwright.
const {chromium}=require('playwright');
const fs=require('fs'),assert=require('assert'),path=require('path'),vm=require('vm');
async function main(){
 const files=process.argv.slice(2);assert(files.length,'Pass one or more exported HTML files');
 const app=fs.readFileSync(path.join(__dirname,'../../experiments/replay_assets/replay.js'),'utf8');
 const timeCode=app.slice(app.indexOf('const timeDivisor='),app.indexOf('function indexAt'));
 for(const [totalTime,t,unit] of [[.2,.125,'ms'],[10,5,'s'],[600,123,'min'],[7200,7200,'h']]){
  const ctx={totalTime,fmt:(n,p=1)=>Number(n).toFixed(p)};vm.createContext(ctx);vm.runInContext(timeCode+';result={label:formatTime('+t+'),unit:axisUnit,x:axisTime('+totalTime+')}',ctx);assert.equal(ctx.result.unit,unit);assert(Number.isFinite(ctx.result.x));if(totalTime===7200)assert.equal(ctx.result.label,'2:00:00.000');
 }
 const browser=await chromium.launch({headless:true,executablePath:process.env.REPLAY_CHROMIUM_PATH||undefined,args:['--enable-unsafe-swiftshader']});
 const reports=[];
 try{
 for(const file of files){
  const context=await browser.newContext({offline:true,reducedMotion:'reduce',viewport:{width:1440,height:1100}});
  const page=await context.newPage(),errors=[],requests=[];
  page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(/^https?:/.test(r.url()))requests.push(r.url())});
  await page.goto('file://'+path.resolve(file));await page.waitForFunction(()=>window.replayDiagnostics?.().sceneReady,{timeout:20000});
  const data=await page.evaluate(()=>JSON.parse(document.getElementById('replay-data').textContent));
  assert.equal(await page.locator('#ue-grid button').count(),data.users.length);assert.equal(await page.locator('#cell-rows tr').count(),data.manifest.cells.length);
  assert(Math.abs(Number(await page.locator('#speed').inputValue())-data.meta.playback_speed)<1e-9);
  const seek=async t=>{await page.locator('#seek').evaluate((e,v)=>{e.value=String(v*1000);e.dispatchEvent(new Event('input',{bubbles:true}))},t);await page.waitForTimeout(50)};
  const midpoint=Math.round(data.meta.slots/2)*data.meta.slot_ms/1000;
  for(const t of [0,midpoint,data.meta.simulation_s]){
   await seek(t);const d=await page.evaluate(()=>window.replayDiagnostics());assert(Math.abs(d.time-t)<1e-8);assert(d.positions.every(p=>p.every(Number.isFinite)));
  }
  const last=data.frames.at(-1),sum=last.link.reduce((a,b)=>a.map((x,i)=>x+b[i]),[0,0,0,0]);
  assert.equal(await page.locator('#slot').innerText(),String(data.meta.slots));
  assert.equal(await page.locator('#avg-goodput').innerText(),(data.totals.delivered_bytes*8/data.meta.simulation_s/1e6).toFixed(2));
  assert.equal(await page.locator('#net-first-n').innerText(),`${sum[1].toLocaleString('en-US')} / ${sum[0].toLocaleString('en-US')}`);
  assert.equal(await page.locator('#net-retx-n').innerText(),`${sum[3].toLocaleString('en-US')} / ${sum[2].toLocaleString('en-US')}`);
  assert.equal(await page.locator('#net-rf-energy').innerText(),last.rf_energy_j.toFixed(2));
  for(let i=0;i<data.users.length;i++){
   await page.locator('#ue-grid button').nth(i).click();assert.equal((await page.evaluate(()=>window.replayDiagnostics())).selected,i);
   for(const metric of ['energy','power','ue-throughput','completion-delay','hol']){await page.locator('#detail-metric').selectOption(metric);assert.equal(await page.locator('#hol-chart svg').count(),1);}
  }
  const layouts=[];for(const width of [320,375,414,768,1440]){await page.setViewportSize({width,height:1100});await page.waitForTimeout(100);const scroll=await page.evaluate(()=>document.documentElement.scrollWidth);assert(scroll<=width,`Overflow at ${width}: ${scroll}`);layouts.push(width);}
  await page.locator('#links').click();await page.locator('#focus-ue').click();await page.locator('#reset-view').click();
  // Use accelerated replay so every duration reaches the endpoint promptly.
  await seek(data.meta.simulation_s*.9);
  const accelerated=await page.locator('#speed option').evaluateAll((options,target)=>options.find(o=>Math.abs(Number(o.value)-target)<Math.max(1e-15,target*1e-10)).value,data.meta.playback_speed*2);
  await page.locator('#speed').selectOption(accelerated);await page.locator('#play').click();
  await page.waitForFunction(()=>window.replayDiagnostics().time>=JSON.parse(document.getElementById('replay-data').textContent).meta.simulation_s&&!window.replayDiagnostics().playing,{timeout:8000});
  assert.deepEqual(errors,[]);assert.deepEqual(requests,[]);
  reports.push({file,passed:true,model:data.meta.model,duration_s:data.meta.simulation_s,frames:data.frames.length,offline:true,external_requests:requests,errors,layouts});
  await context.close();
 }
 }finally{await browser.close()}
 console.log(JSON.stringify({passed:true,time_scale_units:['ms','s','min','h'],reports},null,2));
}
main().catch(e=>{console.error(e);process.exit(1)});
