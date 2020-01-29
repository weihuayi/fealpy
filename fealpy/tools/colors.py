import matplotlib.cm as cm
import matplotlib.colors as colors

def val_to_color(val, cmap='jet'):
    vmax = val.max()
    vmin = val.min()
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    color = mapper.to_rgba(val)
    return color
