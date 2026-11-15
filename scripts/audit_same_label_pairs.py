#!/usr/bin/env python3
"""Find same-label / same-box suspicious pairs per series and make review sheets."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import math

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path("/home/wbk/ultralytics_2027")
SRC = ROOT / "datasets/schemeC_pose23_full_v5"
OUT = ROOT / "runs/audit/label_pairs"
NAMES = ["blue_sentry","blue_hero","blue_engineer","blue_3","blue_4","blue_outpost","blue_base",
         "red_sentry","red_hero","red_engineer","red_3","red_4","red_outpost","red_base"]


def iou(a, b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix=max(0,min(ax2,bx2)-max(ax1,bx1)); iy=max(0,min(ay2,by2)-max(ay1,by1))
    inter=ix*iy; ua=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/ua if ua>0 else 0.0


def series_of(unit: str) -> str:
    p=unit.split('/')
    if p[0]=='pose': return f'pose/{p[1]}'
    if p[0]=='workspaces': return f'workspaces/{p[1]}'
    if p[0]=='source_json': return f'source_json/{p[1]}'
    return unit


def rows_of(label: Path):
    out=[]
    for line in label.read_text().splitlines():
        f=line.split()
        if len(f)!=13: continue
        c=int(f[0]); cx,cy,bw,bh=map(float,f[1:5])
        out.append((c,cx,cy,bw,bh))
    return out


def pair_crop(img: Image.Image, box, size=140):
    w,h=img.size; cx,cy,bw,bh=box
    x1=max(0,int((cx-bw/2)*w)); x2=min(w,int((cx+bw/2)*w)+1)
    y1=max(0,int((cy-bh/2)*h)); y2=min(h,int((cy+bh/2)*h)+1)
    if x2-x1<2 or y2-y1<2: return None
    c=img.crop((x1,y1,x2,y2)).convert('RGB')
    side=max(c.size); pad=Image.new('RGB',(side,side),(0,0,0)); pad.paste(c,((side-c.width)//2,(side-c.height)//2))
    pad=pad.resize((size,size),Image.BICUBIC)
    g=np.asarray(pad.convert('L')); m=float(g.mean())
    if m>0:
        gam=float(np.clip(np.log(70/255)/np.log(max(m,1)/255),0.3,3.0))
        arr=np.clip((np.asarray(pad).astype(np.float32)/255.0)**gam*255,0,255).astype(np.uint8)
        pad=Image.fromarray(arr)
    return pad


def main():
    rows=list(csv.DictReader((SRC/'split_manifest.tsv').open(),delimiter='\t'))
    by_series=defaultdict(list)
    for r in rows:
        s=series_of(r['unit'])
        by_series[s].append(r)
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for series, rs in sorted(by_series.items()):
        found=[]
        for r in rs:
            ip=SRC/'images'/r['split']/r['image']; lp=SRC/'labels'/r['split']/r['label']
            objs=rows_of(lp)
            if len(objs)<2: continue
            for i in range(len(objs)):
                for j in range(i+1,len(objs)):
                    ci,cxj,cyj,bwj,bhj=objs[i]; cj,cxk,cyk,bwk,bhk=objs[j]
                    bi=(cxj-bwj/2,cyj-bhj/2,cxj+bwj/2,cyj+bhj/2)
                    bj=(cxk-bwk/2,cyk-bhk/2,cxk+bwk/2,cyk+bhk/2)
                    ov=iou(bi,bj)
                    if ci==cj and ov>0.30:
                        found.append((r,ci,objs[i],objs[j],ov,'same_class_overlap'))
                    elif ci!=cj and ov>0.85:
                        found.append((r,ci,objs[i],objs[j],ov,'cross_class_same_box'))
        # dedup/sort
        found=sorted(found,key=lambda x:(-x[4],x[0]['image']))
        print(series,'suspicious',len(found))
        for k,(r,ci,a,b,ov,kind) in enumerate(found[:48]):
            manifest.append((series,k,r['split'],r['image'],NAMES[ci],NAMES[b[0]],round(ov,3),kind))
        for si in range(0,min(len(found),48),24):
            chunk=found[si:si+24]
            cellw=300; cellh=170; cols=4; rowsn=6
            sheet=Image.new('RGB',(cols*cellw,rowsn*cellh),(20,20,20)); d=ImageDraw.Draw(sheet)
            for n,(r,ci,a,b,ov,kind) in enumerate(chunk):
                ip=SRC/'images'/r['split']/r['image']
                try: im=Image.open(ip).convert('RGB')
                except Exception: continue
                ca=pair_crop(im,a[1:]); cb=pair_crop(im,b[1:])
                if ca is None or cb is None: continue
                x=(n%cols)*cellw; y=(n//cols)*cellh
                sheet.paste(ca,(x+2,y+18)); sheet.paste(cb,(x+146,y+18))
                d.text((x+2,y+2),f"{si+n} {NAMES[ci]} vs {NAMES[b[0]]} iou{ov:.2f}",fill=(235,235,235))
                d.text((x+2,y+148),f"{kind} {r['unit'][:32]}",fill=(180,180,180))
            if chunk:
                p=OUT/f"{series.replace('/','_')}_{si//24:02d}.jpg"
                sheet.save(p,quality=90)
        # write a per-series index
    with (OUT/'manifest.tsv').open('w',newline='') as f:
        w=csv.writer(f,delimiter='\t'); w.writerow(['series','idx','split','image','class','other_class','iou','kind']); w.writerows(manifest)
    print('wrote',OUT,'rows',len(manifest))
    # series summary
    c=Counter(x[0] for x in manifest); print('series counts',dict(c))


if __name__=='__main__':
    main()
