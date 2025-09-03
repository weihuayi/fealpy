
import os
import argparse
import imageio.v2 as imageio
parser = argparse.ArgumentParser(description=
        """
        制作动图
        """)

parser.add_argument('--output', default='./', type=str)
parser.add_argument('--name', default='movie.gif', type=str)
parser.add_argument('--path', default='./', type=str)
parser.add_argument('--space', default=10, type=int)

args = parser.parse_args()
output = args.output
name = args.name
path = args.path
space = args.space

path_from = 'ls ' + path + '*.png'
path_out = output + name
fnames = [s[:-1] for s in os.popen(path_from)]

with imageio.get_writer(path_out, mode='I') as writer:
    for name in fnames[::space]:
        image = imageio.imread(name)
        writer.append_data(image)



