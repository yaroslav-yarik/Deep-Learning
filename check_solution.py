"""Small independent gradient and edge-case checks; no datasets are required."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).parent / 'assignment2'))
from dlcv import layers as L
from dlcv.classifiers.fc_net import FullyConnectedNet
from dlcv.classifiers.cnn import ThreeLayerConvNet

rng = np.random.default_rng(231)

def close(actual, expected, name):
    np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-7, err_msg=name)

def numerical(f, x):
    result = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        old = x[i]
        x[i] = old + 1e-5
        positive = f(x)
        x[i] = old - 1e-5
        negative = f(x)
        x[i] = old
        result[i] = (positive - negative) / 2e-5
    return result

def gradients(forward, backward, args, name):
    out, cache = forward(*args)
    dout = rng.normal(size=out.shape)
    computed = backward(dout, cache)
    if not isinstance(computed, tuple):
        computed = (computed,)
    for i, grad in enumerate(computed):
        expected = numerical(lambda value: np.sum(forward(*(args[:i] + [value] + args[i+1:]))[0] * dout), args[i])
        close(grad, expected, name + ':' + str(i))
    print(name, 'PASS', flush=True)

x, gamma, beta = rng.normal(size=(4, 5)), rng.normal(size=5), rng.normal(size=5)
for backward in [L.batchnorm_backward, L.batchnorm_backward_alt]:
    gradients(lambda x,g,b: L.batchnorm_forward(x,g,b,{'mode':'train'}), backward,
              [x.copy(), gamma.copy(), beta.copy()], backward.__name__)
gradients(lambda x,g,b: L.layernorm_forward(x,g,b,{}), L.layernorm_backward,
          [x.copy(), gamma.copy(), beta.copy()], 'layernorm')
param = {'mode':'train', 'momentum':0.0}
L.batchnorm_forward(x, gamma, beta, param)
param['mode'] = 'test'
close(L.batchnorm_forward(x,gamma,beta,param)[0], gamma*(x-x.mean(0))/np.sqrt(x.var(0)+1e-5)+beta, 'batchnorm inference')
for keep in [1.0,0.5]:
    gradients(lambda x: L.dropout_forward(x,{'mode':'train','p':keep,'seed':123}), L.dropout_backward, [x.copy()], 'dropout '+str(keep))
close(L.dropout_forward(x, {'mode':'test','p':0.2})[0], x, 'dropout inference')

for stride,pad in [(1,0),(2,1)]:
    xx = rng.normal(size=(1,2,4,5))
    w = rng.normal(size=(2,2,2,3)); b = rng.normal(size=2)
    param = {'stride':stride,'pad':pad}
    out,_ = L.conv_forward_naive(xx,w,b,param)
    xp = np.pad(xx,((0,0),(0,0),(pad,pad),(pad,pad)))
    expected = np.zeros_like(out)
    for n,f,i,j in np.ndindex(out.shape):
        expected[n,f,i,j] = sum(xp[n,c,i*stride+h,j*stride+k]*w[f,c,h,k]
                                for c,h,k in np.ndindex(w.shape[1:])) + b[f]
    close(out, expected, 'conv reference')
    gradients(lambda x,w,b: L.conv_forward_naive(x,w,b,param), L.conv_backward_naive,
              [xx,w,b], 'conv stride='+str(stride)+' pad='+str(pad))
for stride in [1,2]:
    param={'pool_height':2,'pool_width':2,'stride':stride}
    gradients(lambda x: L.max_pool_forward_naive(x,param), L.max_pool_backward_naive,
              [rng.normal(size=(1,2,4,4))], 'pool stride='+str(stride))
tied=np.ones((1,1,2,2))
_,cache=L.max_pool_forward_naive(tied,{'pool_height':2,'pool_width':2,'stride':2})
close(L.max_pool_backward_naive(np.ones((1,1,1,1)),cache),tied/4,'pool tied subgradient')
xx=rng.normal(size=(2,4,2,2)); gamma=rng.normal(size=4); beta=rng.normal(size=4)
gradients(lambda x,g,b:L.spatial_batchnorm_forward(x,g,b,{'mode':'train'}),L.spatial_batchnorm_backward,
          [xx,gamma,beta],'spatial batchnorm')
for groups in [1,2,4]:
    gradients(lambda x,g,b:L.spatial_groupnorm_forward(x,g,b,groups,{}),L.spatial_groupnorm_backward,
              [xx,gamma.reshape(1,4,1,1),beta.reshape(1,4,1,1)],'groupnorm G='+str(groups))
try:
    L.spatial_groupnorm_forward(xx,gamma.reshape(1,4,1,1),beta.reshape(1,4,1,1),3,{})
except ValueError:
    pass
else:
    raise AssertionError('groupnorm should reject a nondivisor group count')

def model_gradients(model, xx, yy, name):
    # Keep the check away from ReLU's nondifferentiable zero.
    for key in model.params:
        if key.startswith('b'):
            model.params[key][:]=rng.normal(size=model.params[key].shape)*0.03
    loss,grads=model.loss(xx,yy)
    assert np.isfinite(loss) and set(grads)==set(model.params)
    for key,grad in grads.items():
        expected=numerical(lambda _:model.loss(xx,yy)[0],model.params[key])
        close(grad,expected,name+':'+key)
    assert model.loss(xx).shape==(len(xx),3)
    print(name,'PASS',flush=True)

np.random.seed(231)
for norm in [None,'batchnorm','layernorm']:
    for keep in [1,.5]:
        model=FullyConnectedNet([7,8],input_dim=3,num_classes=3,normalization=norm,
            dropout_keep_ratio=keep,seed=123,weight_scale=.1,reg=.2,dtype=np.float64)
        model_gradients(model,rng.normal(size=(3,3)),np.array([0,1,2]),'FC '+str(norm)+' dropout='+str(keep))
model=ThreeLayerConvNet(input_dim=(1,4,4),num_filters=2,filter_size=3,hidden_dim=4,
                       num_classes=3,weight_scale=.15,reg=.2,dtype=np.float64)
model_gradients(model,rng.normal(size=(2,1,4,4)),np.array([0,2]),'CNN')
print('NumPy checks PASS',flush=True)

try:
    import torch
except ImportError:
    if '--require-torch' in sys.argv:
        raise
    print('Torch checks SKIPPED: torch not installed',flush=True)
else:
    from dlcv import rnn_layers_pytorch as R
    from dlcv.classifiers.rnn_pytorch import CaptioningRNN
    torch.set_num_threads(2)
    torch.manual_seed(231)
    def tensor(*shape):
        return (torch.randn(*shape,dtype=torch.float64)*.2).requires_grad_()
    args=(tensor(2,3),tensor(2,4),tensor(3,4),tensor(4,4),tensor(4))
    assert torch.autograd.gradcheck(R.rnn_step_forward,args)
    args=(tensor(2,3,3),tensor(2,4),tensor(3,4),tensor(4,4),tensor(4))
    assert torch.autograd.gradcheck(R.rnn_forward,args)
    assert R.rnn_forward(args[0][:,:0],*args[1:]).shape==(2,0,4)
    ids=torch.tensor([[0,1,1],[2,0,2]])
    assert torch.autograd.gradcheck(lambda w:R.word_embedding_forward(ids,w),(tensor(3,4),))
    assert torch.autograd.gradcheck(R.temporal_affine_forward,(tensor(2,3,4),tensor(4,5),tensor(5)))
    scores=tensor(2,3,4); labels=torch.tensor([[1,2,3],[3,2,1]]);mask=torch.tensor([[True,False,True],[False,True,False]])
    assert torch.autograd.gradcheck(lambda s:R.temporal_softmax_loss(s,labels,mask),(scores,))
    grad=torch.autograd.grad(R.temporal_softmax_loss(scores,labels,mask),scores)[0]
    assert torch.count_nonzero(grad[~mask])==0
    model=CaptioningRNN({'<NULL>':0,'<START>':1,'<END>':2,'a':3},input_dim=3,wordvec_dim=2,hidden_dim=3,dtype=torch.float64)
    keys=list(model.params)
    params=tuple(v.detach().requires_grad_() for v in model.params.values())
    features=tensor(2,3); captions=torch.tensor([[1,3,2,0],[1,2,0,0]],dtype=torch.int32)
    def caption_loss(*values):
        model.params=dict(zip(keys,values))
        return model.loss(features,captions)
    assert torch.autograd.gradcheck(caption_loss,params)
    assert model.sample(features,max_length=0).shape==(2,0)
    generated=model.sample(features,max_length=4)
    assert generated.shape==(2,4) and generated.dtype==torch.long
    assert torch.all((generated>=0)&(generated<4))
    print('Torch RNN and CaptioningRNN autograd checks PASS',flush=True)
