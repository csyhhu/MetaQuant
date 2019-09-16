# MetaQuant
Codes for Accepted Paper : "MetaQuant: Learning to Quantize by Learning to Penetrate Non-differentiable Quantization" in NeurIPS 2019

## Motivation
Most training-based quantization methods replies on [Straight-Through-Estimator](https://arxiv.org/abs/1602.02830) (STE)
to enable training due to the non-differentiable discrete quantization function <a href="https://www.codecogs.com/eqnedit.php?latex=Q(\cdot)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Q(\cdot)" title="Q(\cdot)" /></a>. Formally:

- Forward:

<a href="https://www.codecogs.com/eqnedit.php?latex=\ell&space;=&space;\text{Loss}(f(Q(\mathbf{W}));&space;\mathbf{x}),&space;y)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\ell&space;=&space;\text{Loss}(f(Q(\mathbf{W}));&space;\mathbf{x}),&space;y)" title="\ell = \text{Loss}(f(Q(\mathbf{W})); \mathbf{x}), y)" /></a>

- Backward:

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;Q(\mathbf{W})}{\partial&space;\mathbf{W}}&space;=\left\{&space;\begin{aligned}&space;1&space;&&space;\qquad&space;\text{if}&space;\qquad&space;|\mathbf{W}|&space;\leq&space;1&space;\nonumber&space;\\&space;0&space;&&space;\qquad&space;\text{else}.&space;\end{aligned}&space;\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;Q(\mathbf{W})}{\partial&space;\mathbf{W}}&space;=\left\{&space;\begin{aligned}&space;1&space;&&space;\qquad&space;\text{if}&space;\qquad&space;|\mathbf{W}|&space;\leq&space;1&space;\nonumber&space;\\&space;0&space;&&space;\qquad&space;\text{else}.&space;\end{aligned}&space;\right." title="\frac{\partial Q(\mathbf{W})}{\partial \mathbf{W}} =\left\{ \begin{aligned} 1 & \qquad \text{if} \qquad |\mathbf{W}| \leq 1 \nonumber \\ 0 & \qquad \text{else}. \end{aligned} \right." /></a>

STE is widely used in training-based quantization
as it provides an approximated gradient for penetration 
of <a href="https://www.codecogs.com/eqnedit.php?latex=Q(\cdot)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Q(\cdot)" title="Q(\cdot)" /></a> 
with an easy implementation. 
However, it inevitably brings the problem of **gradient mismatch**: 
the gradients of the weights are not generated 
using the value of weights, but rather its quantized value.

To overcome the problem of gradient mismatch and explore better gradients in training-based methods, 
we propose to learn <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;Q(\mathbf{W})}{\partial&space;\mathbf{W}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;Q(\mathbf{W})}{\partial&space;\mathbf{W}}" title="\frac{\partial Q(\mathbf{W})}{\partial \mathbf{W}}" /></a> 
by a neural network (<a href="https://www.codecogs.com/eqnedit.php?latex=\mathcal{M}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathcal{M}" title="\mathcal{M}" /></a>) 
during quantization training. 
Such neural network is called **Meta Quantizer** and is trained 
together with the base quantized model.

## Method
![Overflow of MetaQuant](./figs/MetaQuant.png)

![Incorporation of Meta Quantizer into quantization training.](./figs/MetaQuant-Forward.png)