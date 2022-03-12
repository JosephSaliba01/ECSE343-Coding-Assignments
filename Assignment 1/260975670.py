"""ECSE343 Assignment 1."""
__author__ = 'Joseph Saliba 260975670.'
__copyright__ = 'Own work with base file provided by McGill.'

# import statements
from sys import base_prefix
import matplotlib.pyplot as plt   # plotting
from scipy.linalg import lu_factor, lu_solve  # scipy's LU solver
import numpy as np                # all of numpy, for now...
del np.vander, np.linalg.inv      # ... except for two functions


def LU_Decomposition(A):
    """
    ! Deliverable 1
    Input:
    A: (n,n) numpy array

    Outputs:
    L, U
    L: (n, n) numpy array
    U: (n, n) numpy array
    """
    L, U = np.eye(A.shape[0]), np.zeros(A.shape)
    n = A.shape[1]

    for i in range(n):
        for j in range(n):
            U[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(i))

        for j in range(n):
            L[j, i] = (A[j, i] - sum(L[j, k] * U[k, i]
                       for k in range(i))) / U[i, i]
    return L, U


def ForwardSubstitution(L, b):
    """
    ! Deliverable 2
    Inputs: 
    L: (n, n) numpy array
    b: (n,) numpy array

    Output: 
    y: (n,) numpy array
    """
    y = np.zeros(L.shape[0])

    for i in range(len(y)):
        y[i] = (b[i] - sum(L[i, j] * y[j] for j in range(i))) / L[i, i]
    return y


def BackwardSubstitution(U, y):
    """
    ! Deliverable 3
    Perform back substitution to solve Ux=y where U is an upper triangular matrix

    Parameters:     
    U: (n, n) numpy array
    b: (n,) numpy array

    Returns: 
    x np.array(n): Solution vector
    """
    x = np.zeros(U.shape[0])
    n = U.shape[0]

    for i in range(n-1, -1, -1):
        x[i] = (y[i] - sum(U[i, j] * x[j] for j in range(i+1, n))) / U[i, i]

    return x


def LU_solver(A, b):
    """
    !Deliverable 4
    Inputs:
    A: (n,n) numpy array
    b: (n,) numpy array

    Output:
    x: (n,) numpy array
    """
    x = np.zeros(A.shape[0])
    L, U = LU_Decomposition(A)
    x = BackwardSubstitution(U, ForwardSubstitution(L, b))
    return x


def GetVandermonde(t, n):
    """
    ! Deliverable 5
    Inputs:
    t: (m,) numpy array
    n: scalar integer

    Output:
    A: (m, n) shaped numpy array
    """
    A = np.zeros((t.shape[0], n))
    m = t.shape[0]
    for i in range(m):
        for j in range(n):
            A[i, j] = t[i] ** j

    return A


def PolynomialFit(data, n):
    """
    ! Deliverable 6
    Input:
    data: (m,2) numpy array of data points (t_i, b_i)
    n: positive integer for an (n-1) degree polynomial

    Output:
    x: (n, ) numpy array of the polynomial coefficients for the terms t^0, t^1,... t^(n-1)
    """
    x = np.zeros((n,))
    A = GetVandermonde(data.T[0], n)
    x = LU_solver(A.T @ A, A.T @ data.T[1])
    return x


def CreateBlurMatrix(kernel1Dx, width, height):
    """
    ! Deliverable 7
    This method takes in the 1D kernel, the width, and the
    height of the image and constructs the blur matrix A.
    """
    if (kernel1Dx.shape[0] % 2 == 0) or (kernel1Dx.shape[0] > width):
        print("ERROR in inputs.\n")
        return

    n = width * height
    k = kernel1Dx.shape[0]
    A = np.zeros(shape=(n, n))

    for i in range(n):
        for e, j in enumerate(range(i - k // 2, i + k // 2 + 1)):
            A[i, j % n] = kernel1Dx[e]

    return A


def BlurImage(blurMatrix, image):
    """
    ! Deliverable 8
    Given a blurMatrix and an image, this function should return a blurred image

    Parameters:
    blurMatrix: numpy array
    image: numpy array; image to blur

    Returns
    blurred_image: numpy array with same shape as `image`
    """
    blurred_img = np.zeros(image.shape)
    blurred_img = np.reshape(blurMatrix @ image.flatten(), image.shape)
    return blurred_img


def DeblurImage(blurMatrix, blurred_image):
    """
    ! Deliverable 9
    """
    deblurred_img = np.zeros(blurred_image.shape)
    deblurred_img = np.reshape(lu_solve(
        lu_factor(blurMatrix), blurred_image.flatten()), blurred_image.shape)
    return deblurred_img


# Some example test routines for the deliverables.
# Feel free to write and include your own tests here.
# Code in this main block will not count for credit,
# but the collaboration and plagiarism policies still hold.


def main():
    print("\n\n[Deliverable 1] LU Factorization\n\n")
    test_matrix = np.array([[2, -1, -2],
                            [-4, 6, 3],
                            [-4, -2, 8]])

    # test_matrix = np.array([[2, 7, 1],
    #                         [3, -2, 0],
    #                         [1, 5, 3]])

    L, U = LU_Decomposition(test_matrix)

    try:
        assert(np.isclose(test_matrix, L @ U).all())
        assert((np.tril(L) == L).all())
        assert((np.triu(U) == U).all())
        print("LU decomposition passes one test.\n\n")
    except:
        raise Exception("LU decomposition fails one test.\n\n")

    print("[Deliverable 2] Forward Substitution\n\n")
    L = np.tril([[4, 12, -16], [12, 37, -43], [-16, -43, 98]])
    b = np.array([1, 2, 3])
    y = ForwardSubstitution(L, b)

    try:
        assert(np.isclose(L @ y, b).all())
        print("Forward substitution passes one test.\n\n")
    except:
        raise Exception("Forward substitution failed one test.\n\n")

    print("[Deliverable 3] Backward Substitution\n\n")
    U = np.triu([[4., 12., -16.], [12., 37., -43.], [-16., -43., 98.]])
    y = np.array([1., 2., 3.])
    x = BackwardSubstitution(U, y)

    # print(f'Expectation: {U @ x}')

    # breakpoint()

    try:
        assert(np.isclose(U @ x, y).all())
        print("Backward substitution passes one test.\n\n")
    except:
        raise Exception("Backward substitution failed one test.\n\n")

    print("[Deliverable 4] LU Solver\n\n")
    A = np.array([[2, -1, -2],
                  [-4, 6, 3],
                  [-4, -2, 8]])
    x = np.array([1, 2, 3])
    b = A @ x
    lu_solved_x = LU_solver(A, b)

    try:
        assert(np.allclose(x, lu_solved_x))
        print("LU solver passes one test.\n\n")
    except:
        raise Exception("LU solver failed one test.\n\n")

    print("[Deliverable 5] Constructing Vandermonde Matrix\n\n")
    n = 5
    m = 10
    t = np.sort(np.random.randn(m))
    A = GetVandermonde(t, n)

    try:
        from numpy import vander
        try:
            assert(np.isclose(A, np.vander(t, n, True)).all())
            print("GetVandermonde() passes one test.\n\n")
        except:
            raise Exception("GetVandermonde() failed one test.\n\n")
    except:
        print("GetVandermonde() Couldn't be tested due to import error.\n\n")

    print("[Deliverable 6a] Fully Constrained Polynomial Fitting\n\n")
    m = 10
    n = m
    t = np.sort(np.random.randn(m))
    A = GetVandermonde(t, n)
    x = np.random.randn(n).astype('d')
    b = A @ x

    data = np.empty((m, 2))
    data[:, 0] = t
    data[:, 1] = b

    fit_x = PolynomialFit(data, n)
    fit_b = A @ fit_x

    _, plots = plt.subplots(2)
    plots[0].scatter(t, b, color='blue', label='Data')
    plots[0].plot(t, fit_b, color='orange', label='Your curve')
    plots[0].set_title(f'Fully Constrained Polynomial Fit', size=8)
    plots[0].set_ylabel("b -- polynomial points")
    plots[0].legend()

    print("[Deliverable 6b] Overdetermined Polynomial Fitting")
    m = 20
    n = 5
    t = np.sort(np.random.randn(m))
    A = GetVandermonde(t, n)
    assert A.shape[0] > A.shape[1]
    x = np.random.randn(n)
    b = np.random.normal(A @ x, scale=np.ones((m,)))

    data = np.empty((m, 2))
    data[:, 0] = t
    data[:, 1] = b

    fit_x = PolynomialFit(data, n)
    fit_b = A @ fit_x

    plots[1].scatter(t, b, color='blue', label='Data')
    plots[1].plot(t, fit_b, color='orange', label='Your curve')
    plots[1].set_title(f'Overdetermined Least-squares Polynomial Fit', size=8)
    plots[1].set_xlabel("t -- independent polynomial variable")
    plots[1].set_ylabel("b -- polynomial points")
    plots[1].legend()

    plt.show()  # this is a blocking call; kill the plotting window to continue execution

    print("[Deliverable 7 - 9] Blurring and deblurring")
    # Load test data from file
    image, blurred_image, kernel_1d = np.empty(
        (100, 100)), np.empty((100, 100)), np.empty((21,))
    with open('Q3-test-data.npy', 'rb') as f:
        image = np.load(f)
        blurred_image = np.load(f)
        kernel_1d = np.load(f)
        # print(f'{kernel_1d = }, {len(kernel_1d) = }')

    A = CreateBlurMatrix(kernel_1d, image.shape[0], image.shape[1])

    # Display the test images we provide, as well as
    # your (to-be-completed) blurred and deblurred images
    cmap = plt.get_cmap('gray')
    _, plots = plt.subplots(2, 2)
    plt.setp(plots, xticks=[], yticks=[])

    plots[0, 0].set_title('original image - from file', size=8)
    plots[0, 0].imshow(image, cmap, vmin=0, vmax=1)

    plots[0, 1].set_title('blurred image - from file', size=8)
    plots[0, 1].imshow(blurred_image, cmap, vmin=0, vmax=1)

    plots[1, 1].set_title('blurred image - computed', size=8)
    plots[1, 1].imshow(BlurImage(A, image), cmap, vmin=0, vmax=1)

    plots[1, 0].set_title('deblurred image - computed', size=8)
    plots[1, 0].imshow(DeblurImage(A, blurred_image), cmap, vmin=0, vmax=1)

    plt.show()  # this is a blocking call; kill the plotting window to continue execution


if __name__ == "__main__":
    main()
