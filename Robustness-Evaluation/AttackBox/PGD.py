import torch
import torch
import torch.nn.functional as F
from torch.autograd import Variable

def pgd_attack(model, images, labels, eps=0.1, alpha=0.01, iters=40, randomize=True, norm='inf'):

    # set random seed 
    if randomize:
        torch.manual_seed(torch.randint(1000000, size=(1,)).item())

    images = Variable(images, requires_grad=True)
    ori_images = images.data

    for i in range(iters):

        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)

        model.zero_grad()
        loss.backward()

        # perturb the image

        if norm == 'l1':
            images_grad = alpha * torch.sign(images.grad.data)
        elif norm == 'inf':
            images_grad = alpha * torch.sign(images.grad.data)
        elif norm == 'l2':
            images_grad = alpha * images.grad.data / torch.norm(images.grad.data, p=2)
        else:
            raise ValueError('Invalid norm type selected.')

        images_grad = alpha * torch.sign(images.grad.data)
        images = Variable(torch.clamp(images.data + images_grad, -1, 1), requires_grad=True)

        # project into epsilon ball
        eta = torch.clamp(images.data - ori_images, min=-eps, max=eps)
        images = Variable(torch.clamp((ori_images + eta).data, -1, 1), requires_grad=True)

    return images