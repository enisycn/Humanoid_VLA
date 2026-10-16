"""Side-by-side recorded traces. Finished videos hold their last recorded frame."""
from pathlib import Path
import argparse
import numpy as np
import av
from PIL import Image,ImageDraw,ImageFont
p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--final',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--title',required=True);a=p.parse_args()
with np.load(a.baseline)as z:left=z['observation.images.image'].transpose(0,2,3,1)
with np.load(a.final)as z:right=z['observation.images.image'].transpose(0,2,3,1)
fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
try:font=ImageFont.truetype(fontpath,18);small=ImageFont.truetype(fontpath,14)
except OSError:font=ImageFont.load_default();small=font
writer=av.open(str(a.out.with_suffix('.mp4')),'w');stream=writer.add_stream('libx264',rate=20);stream.width=640;stream.height=420;stream.pix_fmt='yuv420p';stream.options={'crf':'25','preset':'fast'};gif=[]
for t in range(max(len(left),len(right))):
 canvas=Image.new('RGB',(640,420),'white');draw=ImageDraw.Draw(canvas);draw.text((10,3),a.title,font=font,fill='#222')
 for col,(images,label)in enumerate([(left,'Original baseline'),(right,'Post-trained')]):
  i=min(t,len(images)-1);x=320*col;draw.text((x+10,28),label,font=font,fill='#222');canvas.paste(Image.fromarray(images[i]).resize((320,320)),(x,55));draw.text((x+10,380),f'Simulated time: {i*.05:.2f} s',font=small,fill='#333')
  if t>=len(images):draw.text((x+10,400),'Last recorded frame',font=small,fill='#666')
 if t==0:canvas.save(a.out.with_suffix('.jpg'))
 if t%5==0:gif.append(canvas.resize((576,378)).quantize(colors=96))
 for packet in stream.encode(av.VideoFrame.from_image(canvas)):writer.mux(packet)
for packet in stream.encode():writer.mux(packet)
writer.close();gif[0].save(a.out.with_suffix('.gif'),save_all=True,append_images=gif[1:],duration=250,loop=0,optimize=True)
print(a.out.name,len(gif),flush=True)
