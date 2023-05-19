import matplotlib.pyplot as plt


def animate(FEM, a_vecs):
    for j in range(len(a_vecs)):
        FEM.update_animation(a_vecs[:, j])
        plt.plot()
        plt.pause(0.01)
        plt.close('all')
    plt.show()
