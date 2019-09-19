import torch
from torch.optim.optimizer import Optimizer, required


class SGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.

    This is a modified version by Chen Shangyu.
    Basically, it don't add on gradient to the variable, instead it stores updated gradient (by lr and decay)
    back to gradient for further used.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            # weight_decay = group['weight_decay']
            # momentum = group['momentum']
            # dampening = group['dampening']
            # nesterov = group['nesterov']

            for p in group['params']:
                # if p.grad is None:
                #     continue
                # d_p = p.grad.data
                # if weight_decay != 0:
                #     d_p.add_(weight_decay, p.data)
                # if momentum != 0:
                #     param_state = self.state[p]
                #     if 'momentum_buffer' not in param_state:
                #         buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                #         buf.mul_(momentum).add_(d_p)
                #     else:
                #         buf = param_state['momentum_buffer']
                #         buf.mul_(momentum).add_(1 - dampening, d_p)
                #     if nesterov:
                #         d_p = d_p.add(momentum, buf)
                #     else:
                #         d_p = buf
                #
                # p.data.add_(-group['lr'], d_p)
                # # p.grad.data = d_p
                p.data.add_(-group['lr'], p.grad.data)

        return loss

    def get_refine_gradient(self):
        """Performs a single optimization step.
        """

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # p.grad.data = -group['lr'] * d_p
                # p.refinement = d_p.data.clone() / (p.grad.data.clone() + 1e-10)
                p.grad.data = d_p

        # return refinement

        # return loss


if __name__ == '__main__':

    from models.CIFARNet import CIFARNet2 as NetWork
    import torch
    from torch.autograd.variable import Variable

    net = NetWork()
    optimizer = SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.5)
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.MSELoss()

    # inputs = Variable(torch.randn(1,3,32,32))
    # targets = Variable(torch.LongTensor([3]))
    inputs = torch.rand([1,3,32,32])
    targets = torch.rand([1, 10])
    outputs = net(inputs)

    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()

    print ('Gradient before refine:')
    print (net.conv1.weight.grad[0,0,:,:])
    old_gradient = net.conv1.weight.grad.data.clone()
    optimizer.get_refine_gradient()
    print('Gradient after refine:')
    print (net.conv1.weight.grad[0,0,:,:])
    new_gradient = net.conv1.weight.grad.data.clone()

