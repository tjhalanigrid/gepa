"""
Interactive bbox canvas component served via st.components.v1.html().

NOTE: st.components.v1.html() does not support return values (always returns None).
This component is display-only — it renders the image with draggable SVG overlays
for visual feedback. Actual state changes are captured via native Streamlit widgets
in the caller (selectboxes, buttons). See dashboard/app.py Step 2.

The canvas provides:
- Numbered coloured bounding boxes over the image
- Click-to-select with class/part/severity edit panel
- Drag to move, resize handle, delete button (visual feedback only)
- Draw new box mode
- Low-confidence boxes rendered with dashed stroke
"""

import json
import streamlit.components.v1 as components

DAMAGE_CLASSES = [
    "dent", "scratch", "crack", "glass_shatter", "lamp_broken", "tire_flat"
]

VALID_PARTS = [
    "front_bumper", "rear_bumper", "hood", "windshield", "rear_windshield",
    "front_left_door", "front_right_door", "rear_left_door", "rear_right_door",
    "left_fender", "right_fender", "trunk_lid", "roof_panel",
    "headlight", "taillight", "tire",
]

CLASS_COLORS = {
    "dent":          "#378ADD",
    "scratch":       "#1D9E75",
    "crack":         "#BA7517",
    "glass_shatter": "#D4537E",
    "lamp_broken":   "#D85A30",
    "tire_flat":     "#888780",
}


def render_bbox_canvas(
    image_url: str,
    detections: list,
    img_width: int,
    img_height: int,
    canvas_height: int = 560,
    key: str = "bbox_canvas",
) -> None:
    """
    Renders an interactive bbox canvas. Always returns None.
    Editing state must be captured via native Streamlit widgets in the caller.
    """
    detections_json = json.dumps(detections)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #111; font-family: -apple-system, sans-serif; overflow: hidden; }}
  #toolbar {{
    display: flex; gap: 8px; padding: 8px 10px;
    background: #1a1a1a; border-bottom: 1px solid #333;
    align-items: center; flex-wrap: wrap;
  }}
  .tbtn {{
    font-size: 12px; padding: 4px 10px; border-radius: 5px; cursor: pointer;
    border: 1px solid #444; background: #222; color: #ccc; transition: all 0.15s;
  }}
  .tbtn.active {{ background: #1a3a5c; border-color: #378ADD; color: #7FC6FF; }}
  .tbtn:hover {{ background: #2a2a2a; }}
  .tbtn.danger {{ border-color: #7a2020; color: #ff8080; }}
  .tbtn.danger:hover {{ background: #2a1010; }}
  #canvas-wrap {{
    position: relative; overflow: hidden;
    background: #000;
    height: {canvas_height - 48}px;
  }}
  #bg-img {{
    display: block;
    width: 100%; height: 100%;
    object-fit: contain;
    object-position: center center;
    user-select: none; pointer-events: none;
  }}
  #svg-layer {{
    position: absolute; top: 0; left: 0; width: 100%; height: 100%;
    pointer-events: none;
  }}
  .bbox-group {{ pointer-events: all; }}
  .bbox-rect {{ fill: none; stroke-width: 2.5; cursor: move; }}
  .bbox-rect.selected {{ stroke-width: 3; }}
  .bbox-rect.lowconf {{ stroke-dasharray: 6 3; }}
  #edit-panel {{
    position: absolute; right: 8px; top: 8px;
    background: #1c1c1e; border: 1px solid #333; border-radius: 8px;
    padding: 12px; min-width: 180px; z-index: 100;
    display: none; font-size: 12px; color: #ccc;
  }}
  #edit-panel h4 {{ font-size: 11px; color: #888; text-transform: uppercase;
    letter-spacing: 0.05em; margin-bottom: 8px; }}
  #edit-panel label {{ display: block; color: #888; margin-bottom: 2px; margin-top: 6px; }}
  #edit-panel select {{
    width: 100%; font-size: 12px; background: #2a2a2a;
    border: 1px solid #444; color: #eee; border-radius: 4px; padding: 3px;
  }}
  .conf-badge {{
    display: inline-block; font-size: 10px; padding: 2px 6px;
    border-radius: 8px; margin-top: 6px;
  }}
  #add-panel {{
    padding: 8px 10px; background: #1a1a1a; border-top: 1px solid #2a2a2a;
    display: none; gap: 6px; align-items: center; flex-wrap: wrap;
  }}
  #add-panel select {{ font-size: 12px; background: #222;
    border: 1px solid #444; color: #ccc; border-radius: 4px; padding: 3px; }}
  #status {{ font-size: 11px; color: #666; margin-left: auto; }}
</style>
</head>
<body>
<div id="toolbar">
  <button class="tbtn active" id="btn-select" onclick="setTool('select')">▣ Select</button>
  <button class="tbtn" id="btn-draw" onclick="setTool('draw')">＋ Draw box</button>
  <button class="tbtn danger" id="btn-delete" onclick="deleteSelected()" style="display:none">✕ Delete</button>
  <div id="status">Click a box to edit</div>
  <button class="tbtn" onclick="toggleDebug()" style="font-size:10px;margin-left:4px">dbg</button>
</div>
<div id="canvas-wrap">
  <img id="bg-img" src="{image_url}" onload="onImgLoad()" onerror="onImgError()">
  <svg id="svg-layer"></svg>
  <div id="edit-panel">
    <h4>Edit detection</h4>
    <label>Class</label><select id="ep-class" onchange="applyEdit()"></select>
    <label>Part</label><select id="ep-part" onchange="applyEdit()"></select>
    <label>Severity</label>
    <select id="ep-severity" onchange="applyEdit()">
      <option>minor</option><option>moderate</option><option>severe</option>
    </select>
    <div id="ep-conf" class="conf-badge"></div>
  </div>
</div>
<div id="add-panel">
  <span style="font-size:11px;color:#888;">Class:</span>
  <select id="new-class"></select>
  <span style="font-size:11px;color:#888;">part:</span>
  <select id="new-part"></select>
  <span style="font-size:11px;color:#888;">sev:</span>
  <select id="new-severity"><option>minor</option><option>moderate</option><option>severe</option></select>
  <button class="tbtn" onclick="confirmNewBox()">✓ Add</button>
  <button class="tbtn" onclick="cancelNewBox()">Cancel</button>
</div>
<script>
const COLORS = {json.dumps(CLASS_COLORS)};
const CLASSES = {json.dumps(DAMAGE_CLASSES)};
const PARTS   = {json.dumps(VALID_PARTS)};
const ORIG_W  = {img_width};
const ORIG_H  = {img_height};

let boxes = {detections_json}.map((d,i) => ({{...d, id: d.id||('b'+i), bbox:[...(d.bbox||[0,0,100,100])], _selected:false, _removed:false}}));
let tool='select', selId=null, imgEl, svgEl;
let scaleX=1, scaleY=1, offX=0, offY=0;
let drawing=false, drawStart=null, drawRect=null;
let dragging=false, dragBox=null, dragStart=null, dragOrigBbox=null;
let resizing=false, resizeBox=null, resizeOrigBbox=null;
let _debug=false;
function toggleDebug(){{_debug=!_debug;render();}}

function onImgLoad(){{
  imgEl=document.getElementById('bg-img');
  svgEl=document.getElementById('svg-layer');
  requestAnimationFrame(function(){{
    requestAnimationFrame(function(){{
      updateScale(); populateSelects(); render();
      const resizeObs=new ResizeObserver(function(){{updateScale();render();}});
      resizeObs.observe(document.getElementById('canvas-wrap'));
    }});
  }});
}}
function onImgError(){{document.getElementById('status').textContent='Image failed to load';}}

function updateScale(){{
  const wrap=document.getElementById('canvas-wrap');
  const wrapW=wrap.offsetWidth;
  const wrapH=wrap.offsetHeight;
  if(wrapW===0||wrapH===0){{setTimeout(updateScale,100);return;}}
  const imgAspect=ORIG_W/ORIG_H;
  const wrapAspect=wrapW/wrapH;
  let rendW,rendH;
  if(imgAspect>wrapAspect){{rendW=wrapW;rendH=wrapW/imgAspect;}}
  else{{rendH=wrapH;rendW=wrapH*imgAspect;}}
  offX=(wrapW-rendW)/2;
  offY=(wrapH-rendH)/2;
  scaleX=rendW/ORIG_W;
  scaleY=rendH/ORIG_H;
  svgEl.setAttribute('viewBox',`0 0 ${{wrapW}} ${{wrapH}}`);
}}

function populateSelects(){{
  ['ep-class','new-class'].forEach(id=>{{
    const s=document.getElementById(id); s.innerHTML='';
    CLASSES.forEach(c=>{{const o=document.createElement('option');o.value=c;o.textContent=c;s.appendChild(o);}});
  }});
  ['ep-part','new-part'].forEach(id=>{{
    const s=document.getElementById(id); s.innerHTML='';
    PARTS.forEach(p=>{{const o=document.createElement('option');o.value=p;o.textContent=p.replace(/_/g,' ');s.appendChild(o);}});
  }});
}}

function toSvg(bx,by){{return[bx*scaleX+offX,by*scaleY+offY];}}
function toBbox(sx,sy){{return[(sx-offX)/scaleX,(sy-offY)/scaleY];}}

function render(){{
  svgEl.innerHTML='';
  boxes.forEach(b=>{{
    if(b._removed)return;
    const[x1,y1,x2,y2]=b.bbox;
    const[sx1,sy1]=toSvg(x1,y1);
    const[sx2,sy2]=toSvg(x2,y2);
    const sw=sx2-sx1,sh=sy2-sy1;
    const color=COLORS[b.damage]||'#888';
    const isSel=b.id===selId;
    const isLow=(b.confidence||1)<0.5&&b.source!=='human';
    const g=document.createElementNS('http://www.w3.org/2000/svg','g');
    g.setAttribute('class','bbox-group'); g.dataset.id=b.id;

    const rect=document.createElementNS('http://www.w3.org/2000/svg','rect');
    rect.setAttribute('x',sx1);rect.setAttribute('y',sy1);
    rect.setAttribute('width',sw);rect.setAttribute('height',sh);
    rect.setAttribute('stroke',color);rect.setAttribute('rx',3);
    rect.setAttribute('class','bbox-rect'+(isSel?' selected':'')+(isLow?' lowconf':''));
    rect.setAttribute('fill',isSel?color+'18':'none');
    rect.setAttribute('stroke-width',isSel?3:2);
    if(isLow)rect.setAttribute('stroke-dasharray','6 3');
    rect.addEventListener('mousedown',e=>startDrag(e,b.id));
    g.appendChild(rect);

    const lblW=Math.max(80,b.damage.length*7+40),lblH=20;
    const lblY=sy1>lblH+4?sy1-lblH-2:sy1+2;
    const lbg=document.createElementNS('http://www.w3.org/2000/svg','rect');
    lbg.setAttribute('x',sx1);lbg.setAttribute('y',lblY);
    lbg.setAttribute('width',lblW);lbg.setAttribute('height',lblH);
    lbg.setAttribute('fill',color);lbg.setAttribute('rx',3);
    lbg.addEventListener('click',e=>selectBox(b.id));
    g.appendChild(lbg);

    const lbl=document.createElementNS('http://www.w3.org/2000/svg','text');
    lbl.setAttribute('x',sx1+6);lbl.setAttribute('y',lblY+14);
    lbl.setAttribute('fill','#fff');lbl.setAttribute('font-size','11');
    lbl.setAttribute('font-weight','600');lbl.setAttribute('font-family','sans-serif');
    const confStr=b.confidence!=null?` · ${{b.confidence.toFixed?b.confidence.toFixed(2):b.confidence}}`:'';
    const srcStr=b.source==='human'?' · ✎':'';
    lbl.textContent=`${{b.index||'?'}}. ${{b.damage}}${{confStr}}${{srcStr}}`;
    lbl.addEventListener('click',e=>selectBox(b.id));
    g.appendChild(lbl);

    if(isSel){{
      const hr=6,hx=sx2-hr,hy=sy2-hr;
      const handle=document.createElementNS('http://www.w3.org/2000/svg','rect');
      handle.setAttribute('x',hx-hr);handle.setAttribute('y',hy-hr);
      handle.setAttribute('width',hr*2);handle.setAttribute('height',hr*2);
      handle.setAttribute('fill',color);handle.setAttribute('rx',2);
      handle.style.cursor='se-resize';handle.style.pointerEvents='all';
      handle.addEventListener('mousedown',e=>startResize(e,b.id));
      g.appendChild(handle);

      const delBg=document.createElementNS('http://www.w3.org/2000/svg','rect');
      delBg.setAttribute('x',sx2-20);delBg.setAttribute('y',lblY);
      delBg.setAttribute('width',20);delBg.setAttribute('height',lblH);
      delBg.setAttribute('fill','#7a2020');delBg.setAttribute('rx',3);
      delBg.style.pointerEvents='all';delBg.style.cursor='pointer';
      delBg.addEventListener('click',e=>{{e.stopPropagation();removeBox(b.id);}});
      g.appendChild(delBg);

      const delTxt=document.createElementNS('http://www.w3.org/2000/svg','text');
      delTxt.setAttribute('x',sx2-10);delTxt.setAttribute('y',lblY+14);
      delTxt.setAttribute('text-anchor','middle');delTxt.setAttribute('fill','#ffaaaa');
      delTxt.setAttribute('font-size','14');delTxt.setAttribute('font-weight','700');
      delTxt.style.pointerEvents='all';delTxt.style.cursor='pointer';
      delTxt.textContent='×';
      delTxt.addEventListener('click',e=>{{e.stopPropagation();removeBox(b.id);}});
      g.appendChild(delTxt);
    }}
    svgEl.appendChild(g);
  }});

  if(drawing&&drawRect){{
    const dr=document.createElementNS('http://www.w3.org/2000/svg','rect');
    dr.setAttribute('x',drawRect.x);dr.setAttribute('y',drawRect.y);
    dr.setAttribute('width',drawRect.w);dr.setAttribute('height',drawRect.h);
    dr.setAttribute('fill','none');dr.setAttribute('stroke','#fff');
    dr.setAttribute('stroke-dasharray','5 3');dr.setAttribute('stroke-width',1.5);
    svgEl.appendChild(dr);
  }}
  if(_debug){{
    const txt=svgEl.ownerDocument.createElementNS('http://www.w3.org/2000/svg','text');
    txt.setAttribute('x',8);txt.setAttribute('y',20);
    txt.setAttribute('fill','#0f0');txt.setAttribute('font-size','11');
    txt.setAttribute('font-family','monospace');
    txt.textContent=`scale=${{scaleX.toFixed(3)}},${{scaleY.toFixed(3)}} off=${{Math.round(offX)}},${{Math.round(offY)}} img=${{ORIG_W}}x${{ORIG_H}}`;
    svgEl.appendChild(txt);
  }}
}}

function selectBox(id){{
  selId=id;
  const b=boxes.find(x=>x.id===id);if(!b)return;
  document.getElementById('ep-class').value=b.damage||'dent';
  document.getElementById('ep-part').value=b.part||'front_bumper';
  document.getElementById('ep-severity').value=b.severity||'minor';
  const cb=document.getElementById('ep-conf');
  const conf=b.confidence;
  if(conf!=null&&b.source!=='human'){{
    cb.textContent=`Confidence: ${{conf.toFixed?conf.toFixed(2):conf}}${{conf<0.5?' ⚠ low':''}}`;
    cb.style.background=conf<0.5?'#3a2000':'#1a3a1a';
    cb.style.color=conf<0.5?'#ffaa44':'#88cc88';
  }}else if(b.source==='human'){{cb.textContent='Human annotated';cb.style.background='#1a1a3a';cb.style.color='#aaaaff';}}
  else{{cb.textContent='';}}
  document.getElementById('edit-panel').style.display='block';
  document.getElementById('btn-delete').style.display='inline-block';
  document.getElementById('status').textContent=`Editing: ${{b.damage.replace('_',' ')}} on ${{(b.part||'').replace(/_/g,' ')}}`;
  render();
}}

function applyEdit(){{
  const b=boxes.find(x=>x.id===selId);if(!b)return;
  b.damage=document.getElementById('ep-class').value;
  b.part=document.getElementById('ep-part').value;
  b.severity=document.getElementById('ep-severity').value;
  render();
}}

function removeBox(id){{
  const b=boxes.find(x=>x.id===id);if(!b)return;
  b._removed=true; selId=null;
  document.getElementById('edit-panel').style.display='none';
  document.getElementById('btn-delete').style.display='none';
  document.getElementById('status').textContent='Box removed (visual only — use Remove button in right panel)';
  render();
}}

function deleteSelected(){{if(selId)removeBox(selId);}}

const canvasWrap=document.getElementById('canvas-wrap');

canvasWrap.addEventListener('mousedown',e=>{{
  if(tool!=='draw')return;
  const r=e.currentTarget.getBoundingClientRect();
  drawStart={{x:e.clientX-r.left,y:e.clientY-r.top}};
  drawing=true; drawRect={{x:drawStart.x,y:drawStart.y,w:0,h:0}};
}});

canvasWrap.addEventListener('mousemove',e=>{{
  if(dragging&&dragBox){{
    const r=e.currentTarget.getBoundingClientRect();
    const dx=(e.clientX-r.left-dragStart.x)/scaleX;
    const dy=(e.clientY-r.top-dragStart.y)/scaleY;
    dragBox.bbox=[dragOrigBbox[0]+dx,dragOrigBbox[1]+dy,dragOrigBbox[2]+dx,dragOrigBbox[3]+dy];
    render();return;
  }}
  if(resizing&&resizeBox){{
    const r=e.currentTarget.getBoundingClientRect();
    const nx=(e.clientX-r.left-offX)/scaleX,ny=(e.clientY-r.top-offY)/scaleY;
    resizeBox.bbox=[resizeOrigBbox[0],resizeOrigBbox[1],Math.max(resizeOrigBbox[0]+20,nx),Math.max(resizeOrigBbox[1]+20,ny)];
    render();return;
  }}
  if(drawing&&drawStart){{
    const r=e.currentTarget.getBoundingClientRect();
    const cx=e.clientX-r.left,cy=e.clientY-r.top;
    drawRect={{x:Math.min(cx,drawStart.x),y:Math.min(cy,drawStart.y),w:Math.abs(cx-drawStart.x),h:Math.abs(cy-drawStart.y)}};
    render();
  }}
}});

document.addEventListener('mouseup',e=>{{
  if(dragging){{dragging=false;dragBox=null;render();return;}}
  if(resizing){{resizing=false;resizeBox=null;render();return;}}
  if(drawing&&drawRect&&drawRect.w>15&&drawRect.h>15){{
    document.getElementById('add-panel').style.display='flex';
    document.getElementById('status').textContent='Set class for new box, then click Add';
  }}
  drawing=false;
}});

function startDrag(e,id){{
  if(tool!=='select')return;
  e.stopPropagation();e.preventDefault();
  selectBox(id);
  const b=boxes.find(x=>x.id===id);if(!b)return;
  const r=canvasWrap.getBoundingClientRect();
  dragging=true;dragBox=b;
  dragStart={{x:e.clientX-r.left,y:e.clientY-r.top}};
  dragOrigBbox=[...b.bbox];
}}

function startResize(e,id){{
  e.stopPropagation();e.preventDefault();
  const b=boxes.find(x=>x.id===id);if(!b)return;
  resizing=true;resizeBox=b;resizeOrigBbox=[...b.bbox];
}}

function confirmNewBox(){{
  if(!drawRect)return;
  const[bx1,by1]=toBbox(drawRect.x,drawRect.y);
  const[bx2,by2]=toBbox(drawRect.x+drawRect.w,drawRect.y+drawRect.h);
  const maxIdx=Math.max(0,...boxes.map(b=>b.index||0));
  boxes.push({{
    id:'human_'+Date.now(), index:maxIdx+1,
    bbox:[bx1,by1,bx2,by2],
    damage:document.getElementById('new-class').value,
    part:document.getElementById('new-part').value,
    severity:document.getElementById('new-severity').value,
    confidence:1.0, source:'human', cost_min:0, cost_max:0,
    _selected:false, _removed:false,
  }});
  document.getElementById('add-panel').style.display='none';
  drawRect=null;drawStart=null;
  setTool('select'); render();
  document.getElementById('status').textContent='New box added (use Add panel below to register in Streamlit)';
}}

function cancelNewBox(){{
  drawRect=null;drawStart=null;
  document.getElementById('add-panel').style.display='none';
  render();
}}

function setTool(t){{
  tool=t;
  document.getElementById('btn-select').classList.toggle('active',t==='select');
  document.getElementById('btn-draw').classList.toggle('active',t==='draw');
  canvasWrap.style.cursor=t==='draw'?'crosshair':'default';
  if(t==='draw'){{
    selId=null;
    document.getElementById('edit-panel').style.display='none';
    document.getElementById('btn-delete').style.display='none';
    document.getElementById('status').textContent='Click and drag to draw a new box';
  }}
}}
</script>
</body>
</html>"""

    components.html(html, height=canvas_height, scrolling=False)
    return None
