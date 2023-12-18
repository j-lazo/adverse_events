from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os


# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range

def visualize_tsne(tsne, labels, plot_size=1000, max_image_size=100):
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    # scale and move the coordinates so they fit [0; 1] range
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # visualize the plot: samples as colored points
    visualize_tsne_points(tx, ty, labels)

colors_per_class = {
    '0': [0, 0, 0],
    '1': [0, 0, 255]}


def visualize_tsne_points(tx, ty, labels, epoch=0, plot_figure=False, dir_save_fig=''):
    # initialize matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    print('labels', labels)
    # for every class, we'll add a scatter plot separately
    for label in colors_per_class:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]

        # extract the coordinates of the points of this class only
        print('indexes', indices)
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # convert the class color to matplotlib format:
        # BGR -> RGB, divide by 255, convert to np.array
        color = np.array([colors_per_class[label][::-1]], dtype=float) / 255

        # add a scatter plot with the correponding color and label
        print('current tx', current_tx)
        print('current ty', current_ty)
        ax.scatter(current_tx, current_ty, c=color, label=label)

    # build a legend using the labels we set previously
    ax.legend(loc='best')


    if plot_figure is True:
        plt.show()

    if dir_save_fig == '':
        dir_save_figure = os.getcwd() + f'/tsne_points_epoch_{epoch}.png'

    else:
        if not dir_save_fig.endswith('.png'):
            dir_save_figure = dir_save_fig + f'tsne_points_epoch_{epoch}.png'
        else:
            dir_save_figure = dir_save_fig

    print(f'figure saved at: {dir_save_figure}')

    plt.savefig(dir_save_figure)
    plt.close()