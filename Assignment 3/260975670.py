"""ECSE343 Assignment 3."""
__author__ = 'Joseph Saliba 260975670.'
__copyright__ = 'Own work with base file provided by McGill.'

# DO NOT EDIT THESE IMPORT STATEMENTS!
import matplotlib.pyplot as plt   # plotting
import numpy as np                # all of numpy
##########################


# [TODO: Deliverable 1] Construct a blur matrix for a 2D blur kernel.
def CreateBlurMatrix(kernel2D, width, height):

    n = width * height
    A = np.zeros(shape=(n, n))
    k = kernel2D.shape[0]

    padded_grid = np.pad(
        np.arange(width * height).reshape(width, height),
        (k//2, k//2),
        mode='wrap'
    )

    strides = np.lib.stride_tricks.sliding_window_view(
        padded_grid, (k, k)).reshape(width * height, k ** 2)

    for i in range(n):
        A[i][strides[i]] = kernel2D.flatten()

    return A


# [TODO: Deliverable 2] Output a discrete k x k 2D Gaussian blur kernel


def Gaussian2D(k, sigma):
    kernel2D = np.zeros(shape=(k, k))
    x_values = np.arange(- (k // 2), k // 2 + 1)
    y_values = np.arange(- (k // 2), k // 2 + 1)

    for i in range(len(x_values)):
        for j in range(len(y_values)):
            x = x_values[i]
            y = y_values[j]
            kernel2D[i, j] = np.exp(- (x ** 2 + y ** 2) /
                                    (2 * sigma ** 2)) / (2 * np.pi * sigma**2)
    return kernel2D


# [TODO: Deliverable 3] Compute and return the deblurred image with a Tikhonov-regularized linear system solve
def DeblurImage(blurMatrix, blurred_image, lambda_=0.0):
    n = blurMatrix.shape[0]

    deblurred_img = np.linalg.solve(
        blurMatrix.T @ blurMatrix + (lambda_ ** 2) * np.identity(n),
        blurMatrix.T @ blurred_image.flatten()
    ).reshape(blurred_image.shape)

    return deblurred_img


# [TODO: Deliverable 4] Compute and return your highest quality deblurred image using whatever linear system formulation and solver you like
def DeblurImageCompetition(blurMatrix, blurred_image):
    # I optimized lambda of the thikonov matrix manually by
    # applying multiple iterations with different lambda values.
    lambda_ = 0.1125
    deblurred_img = DeblurImage(blurMatrix, blurred_image, lambda_)

    return deblurred_img


# [TODO: Deliverable 5] Return the left and right singular vectors associated to the largest singular value of the input matrix
def PowerMiniSVD(matrix, num_iterations=10**3):
    u0, v0 = np.random.rand(matrix.shape[0]), np.random.rand(matrix.shape[1])

    u_0_mult = matrix @ matrix.T
    v_0_mult = matrix.T @ matrix

    for _ in range(num_iterations):
        p_u0 = u_0_mult @ u0
        u0 = p_u0 / np.linalg.norm(p_u0)

        p_v0 = v_0_mult @ v0
        v0 = p_v0 / np.linalg.norm(p_v0)

    return u0, v0


# [TODO: Deliverable 6] Given the SVD of a blur matrix, deblur the input image using a rank-reduced approximation of the blur matrix. 0 <= R <= 1 denotes the amount of energy to retain in your rank-reduced approximation.
def DeblurImageSVD(U, s, Vt, blurred_image, R):

    values = np.cumsum(s ** 2) / sum(s ** 2)
    L = np.where(values <= R, s, 0)

    sigma_hat_inverse = np.diag(
        np.divide(1, L, where=L != 0, out=np.zeros_like(L))
    )

    deblurred_img = np.linalg.multi_dot(
        [Vt.T, sigma_hat_inverse, U.T, blurred_image.flatten()]
    ).reshape(blurred_image.shape)

    return deblurred_img


# Some example test routines for the deliverables.
# Feel free to write and include your own tests here.
# Code in this main block will not count for credit,
# but the collaboration and plagiarism policies still hold.

def main():
    # Load test data from file
    image, blurred_image, blurred_noisy_image = None, None, None
    with open('A3-test-data.npy', 'rb') as f:
        image = np.load(f)
        blurred_image = np.load(f)
        blurred_noisy_image = np.load(f)

    print("[Deliverables 1 and 2] Blur Matrix Construction with 2D Gaussian kernel")

    # Gaussian kernel 21 x 21, sigma = 1.0: this is the kernel we used to generate blurred_image
    sigma, k = 1.0, 21
    kernel2d = Gaussian2D(k, sigma)
    A = CreateBlurMatrix(kernel2d, image.shape[1], image.shape[0])
    # TODO: test your blurring results

    print("[Deliverables 3 and 4] Regularized Deblurring")
    # Generate and display a handful of deblurred images

    cmap = plt.get_cmap('gray')
    _, plots = plt.subplots(2, 3, figsize=(10, 7))
    plt.setp(plots, xticks=[], yticks=[])

    plots[0, 0].set_title('original image - from file', size=8)
    plots[0, 0].imshow(image, cmap, vmin=0, vmax=1)

    plots[0, 1].set_title('blurred image - from file', size=8)
    plots[0, 1].imshow(blurred_image, cmap, vmin=0, vmax=1)

    plots[0, 2].set_title('blurred image + noise - from file', size=8)
    plots[0, 2].imshow(blurred_noisy_image, cmap, vmin=0, vmax=1)

    plots[1, 0].set_title('Deblurred (no regularization)', size=8)
    plots[1, 0].imshow(DeblurImage(A, blurred_noisy_image),
                       cmap, vmin=0, vmax=1)
    # ########
    # for lambda_var in np.arange(0.1125, 0.114, 0.0001):
    #     temp_img_var = DeblurImage(A, blurred_noisy_image, lambda_var)
    #     print(lambda_var, np.linalg.norm(temp_img_var - image))
    # breakpoint()
    # ########

    temp_img = DeblurImage(A, blurred_noisy_image, 0.05)
    plots[1, 1].set_title("Deblurred (Tikonov with $\lambda = 0.05$; error = " +
                          "{:.2f}".format(np.linalg.norm(temp_img - image)) + ")", size=8)
    plots[1, 1].imshow(temp_img, cmap, vmin=0, vmax=1)

    temp_img = DeblurImageCompetition(A, blurred_noisy_image)
    plots[1, 2].set_title("Deblurred (your best; error = " +
                          "{:.2f}".format(np.linalg.norm(temp_img - image)) + ")", size=8)
    plots[1, 2].imshow(temp_img, cmap, vmin=0, vmax=1)

    plt.show()  # this is a blocking call; kill the plotting window to continue execution

    print("[Deliverables 5] Power Iteration for Mini-SVD")
    # Test for accuracy with a liberal floating-point threshold
    M = np.random.rand(20, 20)
    M_u, M_s, M_Vt = np.linalg.svd(M)
    u0, v0 = PowerMiniSVD(M)
    print("Do left singular vectors match? " +
          str(np.all(np.abs(np.abs(M_u[:, 0]) - np.abs(u0)) < 1e-4)))
    print("Do right singular vectors match? " +
          str(np.all(np.abs(np.abs(M_Vt[0, :]) - np.abs(v0)) < 1e-4)))

    print("[Deliverables 6] SVD Deblurring")
    # U, s, Vt = np.linalg.svd(A)  # This is expensive...
    # np.save("del6_U", U)
    # np.save("del6_s", s)
    # np.save("del6_Vt", Vt)

    U = np.load('del6_U.npy')
    s = np.load('del6_s.npy')
    Vt = np.load('del6_Vt.npy')

    cmap = plt.get_cmap('gray')
    _, plots = plt.subplots(1, 3, figsize=(10, 4))
    plt.setp(plots, xticks=[], yticks=[])

    temp_img = DeblurImageSVD(U, s, Vt, blurred_noisy_image, 0.9)
    plots[0].set_title("Deblurred (90% SVD; error = " +
                       "{:.2f}".format(np.linalg.norm(temp_img - image)) + ")", size=8)
    plots[0].imshow(temp_img, cmap, vmin=0, vmax=1)

    temp_img = DeblurImageSVD(U, s, Vt, blurred_noisy_image, 0.95)
    plots[1].set_title("Deblurred (95% SVD; error = " +
                       "{:.2f}".format(np.linalg.norm(temp_img - image)) + ")", size=8)
    plots[1].imshow(temp_img, cmap, vmin=0, vmax=1)

    temp_img = DeblurImageSVD(U, s, Vt, blurred_noisy_image, 0.99)
    plots[2].set_title("Deblurred (99% SVD; error = " +
                       "{:.2f}".format(np.linalg.norm(temp_img - image)) + ")", size=8)
    plots[2].imshow(temp_img, cmap, vmin=0, vmax=1)

    plt.show()  # this is a blocking call; kill the plotting window to continue execution


if __name__ == "__main__":
    main()
