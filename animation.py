

from plantml import TemperetureFEA
import matplotlib as plt



def animate(FEM, a_vecs):
    for j in range(len(a_vecs)):
        FEM.update_animation(j)
        plt.plot()
        plt.pause(0.005)
    plt.show()