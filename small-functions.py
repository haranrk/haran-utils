import sys
import time
import torch

def write_hists_to_tensorboard(model, summary_writer, step, tag=None):
    '''Writes the histograms of all the params of given model to event files for tensorboard'''
    for name, values in model.named_parameters():  
        if tag==None:
            tag_name = model.__class__.__name__+'.'+name
        else:
            tag_name = tag+'.'+name
        summary_writer.add_histogram(tag=tag_name, values=values, global_step=step)

def progressbar(total, progress, additional_string = ""):
    """
    Displays or updates a console progress bar.

    Original source: https://stackoverflow.com/a/15860757/1391441
    """
    barLength, status = 20, ""
    progress = float(progress) / float(total)
    if progress >= 1.:
        progress, status = 1, "\r\n"
    block = int(round(barLength * progress))
    text = "\r[{}] {:.0f}% {} {}".format("#" * block + "-" * (barLength - block), round(progress * 100, 0), additional_string, status)
    sys.stdout.write(text)
    sys.stdout.flush()

def calc_gradient_penalty(netD, real_data, fake_data, BATCH_SIZE, device, LAMBDA = 10):
    # print "real_data: ", real_data.size(), fake_data.size()
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, real_data.nelement()/BATCH_SIZE).contiguous().view(BATCH_SIZE, 3, 32, 32)
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.to(device)
    interpolates.required_grad = True

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty