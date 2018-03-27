import matplotlib.pyplot as plt
import numpy as np


def plot_example(data, n=1, keys=None, **kwargs):
    fig, ax = plt.subplots(n, 2, sharex=True, sharey=True, **kwargs)

    if keys is None:
        keys = np.random.choice(list(data.keys()), n, replace=False)

    for i, key in enumerate(keys):
        if n == 1:
            ax_row = ax
        else:
            ax_row = ax[i,:]

        #print(key)
        d = data[key]

        image = np.squeeze(d["image"])
        ax_row[0].imshow(image, vmin=0, vmax=1)

        if "mask" in d:
            mask = np.squeeze(d["mask"])
            ax_row[1].imshow(mask, cmap="Greys_r", vmin=0, vmax=1)
        else:
            mask = np.zeros_like(image)
            ax_row[1].imshow(mask, cmap="Greys_r", vmin=0, vmax=1)
            ax_row[1].set_title("NO LABELS")
    
    plt.show()
