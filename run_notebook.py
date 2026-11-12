import os,sys,time,pathlib
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['PYTHONIOENCODING']='utf-8'
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
root=pathlib.Path(__file__).resolve().parent
lab=root/'assignment2'
name=sys.argv[1]
p=lab/name
full_nb=nbformat.read(p,as_version=4)
indices=list(range(len(full_nb.cells)))
if len(sys.argv)>2:
 indices=[int(x) for x in sys.argv[2].split(',')]
nb=nbformat.v4.new_notebook(cells=[full_nb.cells[i] for i in indices], metadata=full_nb.metadata)
def save():
 for index, cell in zip(indices, nb.cells): full_nb.cells[index]=cell
 nbformat.write(full_nb,p)
for c in nb.cells:
 if c.cell_type=='code':c.outputs=[];c.execution_count=None
km=KernelManager(kernel_name='python3')
km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
class StreamingNotebookClient(NotebookClient):
 def process_message(self, msg, cell, cell_index):
  if msg['msg_type'] == 'stream':
   print(msg['content']['text'], end='', flush=True)
  return super().process_message(msg, cell, cell_index)
client=StreamingNotebookClient(nb,timeout=3600,km=km,resources={'metadata':{'path':str(lab)}},allow_errors=False)
start=time.time()
def begin(cell,cell_index,**kwargs):print(f'{name} cell {cell_index} start {time.time()-start:.1f}s',flush=True)
def done(cell,cell_index,**kwargs):
 save()
 for o in cell.get('outputs',[]):
  if o.output_type=='stream':print(o.text[-1500:],flush=True)
 print(f'{name} cell {cell_index} done {time.time()-start:.1f}s',flush=True)
client.on_cell_start=begin;client.on_cell_executed=done
try:client.execute()
finally:save()
print('COMPLETE',name,time.time()-start,flush=True)
