import matplotlib.pyplot as plt


class Visualization:
    """Easy neural network training progress visualization."""

    test = ([0], [0])
    train = ([0], [0])
    cross_entropy = ([], [])

    def __init__(self, epochs=200):
        """Initialize the figure with 2 subplots and 4 line plots."""
        self.fig = plt.figure('Digit Classification')
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)

        self.ax1.set_title('Accuracy')
        self.ax1.set_xlim([0, epochs])
        self.ax1.set_ylim([0, 1.1])
        self.ax2.set_title('Loss')
        self.ax2.set_xlim([0, epochs])
        self.ax2.set_ylim([0, 100])

        self.train_plt, = self.ax1.plot([0], [0])
        self.test_plt, = self.ax1.plot([0], [0])
        self.cross_entropy_plt, = self.ax2.plot([0], [0])
        plt.tight_layout()
        plt.show()

    def __call__(self, i, test=None, train=None, cross_entropy=None):
        """Update the plot by calling the instance. 
        Updates are applied selectively to named arguments.    
        """
        if test is not None:
            self.test[0].append(i)
            self.test[1].append(test)
            self.test_plt.set_data(*self.test)
        if train is not None:
            self.train[0].append(i)
            self.train[1].append(train)
            self.train_plt.set_data(*self.train)
        if cross_entropy is not None:
            self.cross_entropy[0].append(i)
            self.cross_entropy[1].append(cross_entropy)
            self.cross_entropy_plt.set_data(*self.cross_entropy)
        self.ax1.legend(labels=['Train: {:.2f}'.format(self.train[1][-1]),
                                'Test: {:.2f}'.format(self.test[1][-1])])
        self.ax2.legend(
            labels=['Cross Entropy: {:.2f}'.format(self.cross_entropy[1][-1])])
        self.fig.canvas.draw()
