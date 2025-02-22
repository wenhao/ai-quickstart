import numpy as np

class NumpySpikes:

    def print_python_version(self):
        print(np.__version__)
        print(np.show_config())

    def define_zero_vector(self):
        zeros = np.zeros(10)
        print(zeros)

    def find_array_size(self):
        zeros = np.zeros((10, 10))
        print("%d bytes"%(zeros.size*zeros.itemsize))

    def print_method_usage(self):
        print(np.info(np.add))

    def define_arrays(self):
        zeros = np.zeros(10)
        zeros[4] = 1
        print(zeros)

        arange_10_50 = np.arange(10, 50)
        print(arange_10_50)

        arange_50 = np.arange(50)
        print(arange_50[::-1])
        print(np.arange(9).reshape(3,3))
        print(np.nonzero([1, 2, 0, 0, 4, 0]))
        print(np.eye(3))
        print(np.random.rand(3,3,3))
        array_rand_10 = np.random.rand(10, 10)
        print(array_rand_10.min(), array_rand_10.max())
        array_rand_30 = np.random.rand(30)
        print(array_rand_10, array_rand_30.mean())
        array_10 = np.ones((10, 10))
        array_10[1:-1, 1:-1] = 0
        print(array_10)
        array_5 = np.ones((5, 5))
        array_5 = np.pad(array_5, pad_width=1, mode='constant', constant_values=0)
        print(array_5)
        print(np.dot(np.ones((5,3)), np.ones((3,2))))

    def main(self):
        self.define_arrays()

if __name__ == '__main__':
    NumpySpikes().main()
