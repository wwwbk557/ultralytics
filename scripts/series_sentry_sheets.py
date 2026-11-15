#!/usr/bin/env python3
"""Per-series sentry crop sheets for manual label review."""
from __future__ import annotations
import csv, random
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path('/home/wbk/ultralytics_2027'); SRC=ROOT/'datasets/schemeC_pose23_full_v5'; OUT=ROOT/'runs/audit/series_sentry'
NAMES=['blue_sentry','blue_hero','blue_engineer','blue_3','blue_4','blue_outpost','blue_base','red_sentry','red_hero','red_engineer','red_3','red_4','red_outpost','red_base']

def series_of(u):
 p=u.split('/')
 if p[0]=='pose': return f'pose/{p[1]}'
 if p[0]=='workspaces': return f'workspaces/{p[1]}'
 if p[0]=='source_json': return f'source_json/{p[1]}'
 return u

def crop(im,box,size=150):
 w,h=im.size; cx,cy,bw,bh=box; x1=max(0,int((cx-bw/2)*w)); x2=min(w,int((cx+bw/2)*w)+1); y1=max(0,int((cy-bh/2)*h)); y2=min(h,int((cy+bh/2)*h)+1)
 c=im.crop((x1,y1,x2,y2)).convert('RGB'); side=max(c.size); pad=Image.new('RGB',(side,side),(0,0,0)); pad.paste(c,((side-c.width)//2,(side-c.height)//2)); pad=pad.resize((size,size),Image.BICUBIC)
 g=np.asarray(pad.convert('L')); m=float(g.mean())
 if m>0:
  gam=float(np.clip(np.log(80/255)/np.log(max(m,1)/255),0.3,3.0)); arr=np.clip((np.asarray(pad).astype(np.float32)/255.0)**gam*255,0,255).astype(np.uint8); pad=Image.fromarray(arr)
 return pad

def main():
 OUT.mkdir(parents=True,exist_ok=True); by=defaultdict(list)
 for r in csv.DictReader((SRC/'split_manifest.tsv').open(),delimiter='\t'):
  for line in (SRC/'labels'/r['split']/r['label']).read_text().splitlines():
   f=line.split()
   if len(f)!=13: continue
   c=int(f[0])
   if c not in (0,7): continue
   by[series_of(r['unit'])].append((r,c,*map(float,f[1:5])))
 man=[]; random.seed(0)
 for series,items in sorted(by.items()):
  items=sorted(items,key=lambda x:-(x[4]*x[5]))
  # up to 96 largest; enough to see numbers
  for si in range(0,min(len(items),96),48):
   chunk=items[si:si+48]; cols=8; rows=6; tw=150; th=170
   sh=Image.new('RGB',(cols*tw,rows*th),(20,20,20)); d=ImageDraw.Draw(sh)
   for k,(r,c,cx,cy,bw,bh) in enumerate(chunk):
    im=Image.open(SRC/'images'/r['split']/r['image']).convert('RGB'); p=crop(im,(cx,cy,bw,bh))
    x=(k%cols)*tw; y=(k//cols)*th; sh.paste(p,(x,y)); d.text((x+2,y+152),f"{si+k} {NAMES[c]} a{bw*bh*307200:.0f}",fill=(230,230,230))
    man.append((series,si+k,r['split'],r['image'],NAMES[c],round(bw*bh*307200,1)))
   fp=OUT/f"{series.replace('/','_')}_{si//48:02d}.jpg"; sh.save(fp,quality=90); print(fp,len(chunk))
 with (OUT/'manifest.tsv').open('w',newline='') as f:
  w=csv.writer(f,delimiter='\t'); w.writerow(['series','idx','split','image','class','area480']); w.writerows(man)

if __name__=='__main__': main()
