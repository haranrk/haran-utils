import sys
import time

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