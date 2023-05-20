import matplotlib.pyplot as plt


def animate(FEM, a_vecs):
    for j in range(a_vecs.shape[1]):
        FEM.update_animation(a_vecs[:, j])
        plt.plot()
        plt.pause(0.01)
        if j != a_vecs.shape[1] - 1:
            plt.close('all')
    plt.show()
