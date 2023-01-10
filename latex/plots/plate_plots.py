import os
import sys
import re
#import daft
#import torch
#import torch.nn.functional as F
#import pyro
#import pyro.distributions as dist
#import pyro.distributions.constraints as constraints
#import matplotlib.pyplot as plt

import graphviz



if __name__ == "__main__":
    """
    run this file from the same folder of the graphviz files (.gv).
    It will save a .png and .pdf version for every .gv file in the folder,
    overwriting older plots.
    """
    for f in os.listdir():
        if re.search("\.gv$", f):
            print("rendering " + f)
            g = graphviz.Source.from_file(f)
            g.render(format='png', )
            g.render(format='pdf', )
    #g = graphviz.Source.from_file('./vae.gv',)
    #g.render(format='png', )
    #g.render(format='pdf', )

    #g = graphviz.Source.from_file('./vae_p.gv',)
    #g.render(format='png', )
    #g.render(format='pdf', )

    #g = graphviz.Source.from_file('./vae_q.gv',)
    #g.render(format='png', )
    #g.render(format='pdf', )

    #g = graphviz.Source.from_file('./dirichlet_gmm.gv',)
    #g.render(format='png', )
    #g.render(format='pdf', )

    #g = graphviz.Source.from_file('./dirichlet_gmm_p.gv',)
    #g.render(format='png', )
    #g.render(format='pdf', )

    #g = graphviz.Source.from_file('./dirichlet_gmm_q.gv',)
    #g.render(format='png', )
    #g.render(format='pdf', )

    #g = graphviz.Source.from_file('./gmm.gv',)
    #g.render(format='png', )
    #g.render(format='pdf', )

    #g = graphviz.Source.from_file('./gmm_vanilla.gv',)
    #g.render(format='png', )
    #g.render(format='pdf', )

    #g = graphviz.Source.from_file('./vaewz.gv',)
    #g.render(format='png', )
    #g.render(format='pdf', )

    print("done")
