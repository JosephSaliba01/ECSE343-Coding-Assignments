# [TODO] Rename this file to [your student ID].py

# DO NOT EDIT THESE IMPORT STATEMENTS!
import matplotlib.pyplot as plt   # plotting 
import numpy as np                # all of numpy, for now...
##########################

# [TODO: Deliverable 1] Complete this function. It must return a numpy.array of size (1,)
def FloatSystem(beta, t, L, U):
  ### BEGIN SOLUTION

  ### END SOLUTION



# [TODO: Deliverable 2] Write down your answers in the comments, below:
# 1. 
# 2. 
# 3. 
# 4. 
# 5. 



# [TODO: Deliverable 3] Implement a vector dot product using for loops
# Assume x and y are 1D numpy.arrays with the same shape
def SlowDotProduct(x, y):
  ### BEGIN SOLUTION

  ### END SOLUTION


# [TODO: Deliverable 3] Implement a vector dot product using numpy
# Assume x and y are 1D numpy.arrays with the same shape
def FastDotProduct(x, y):
  ### BEGIN SOLUTION

  ### END SOLUTION



# [TODO: Deliverable 4] Return an np.array of shape (2,) according to the specification of the handout
def CatchTheNaN(M):
  idx = np.array([-1,-1])
  ### BEGIN SOLUTION

  ### END SOLUTION
  return idx



# [TODO: Deliverable 5] Implement the naive formulation
# x is an np.array of shape (N,) of the x_i in the equations in section 4.2 of the handout
def LogSumExpNaive(x):
  ### BEGIN SOLUTION

  ### END SOLUTION


# [TODO: Deliverable 5] Implement the robust formulation and include a brief explanation as to why it's more robust
def LogSumExpRobust(x):
  ### BEGIN SOLUTION
  # [Include your brief (1-2 sentences) explanation, here

  ### END SOLUTION


if __name__ == "__main__":
  # Here are some example test routines for the deliverables. 
  # Feel free to write and include your own tests here.
  # None of the code in this main block will count for any credit, i.e., 
  # it won't be graded but you need always follow the collaboration and plagiarism policies.

  # [Deliverable 1 example test]
  print("\n\n[Deliverable 1]\n")
  tmp = FloatSystem(2, 2, -1, 2)
  
  # Here's a concrete solution for a (2,2,-1,2) floating-point system to test against
  test_solution = np.array([ 0.,  0.5, -0.5,  1., -1.,  2., -2.,  4., -4., 0.75, -0.75,  1.5, -1.5,  3., -3.,  6., -6. ])
  test_solution = np.sort(test_solution) # sorting our solution to simplify comparisons

  print("Does your (2,2,-1,2) example system match our test solution? " + str( np.allclose(np.sort(tmp), test_solution) ) )
  
  # Example visualization code: plot an asterisk at each perfectly representable value along the real line
  plt.title('Perfectly representable numbers for the $(2,2,-1,2)$ floating point system')
  plt.plot(tmp, np.zeros(tmp.shape[0]), '*')
  plt.yticks([])
  plt.show() # this is a blocking call; kill the plotting window to continue execution
  
  

  # [Deliverable 2 example tests]
  print("\n\n[Deliverable 2]\n")
  print(" Not applicable (text answers, only); however, you may want to use a breakpoint() here to tinker with code to help answer these questions...")



  # [Deliverable 3 example tests]
  print("\n\n[Deliverable 3]\n")
  x = np.random.rand(100)
  y = np.random.rand(100)
  
  # Test correctness...
  slowResult = SlowDotProduct(x, y)
  fastResult = FastDotProduct(x, y)
  print( "Matching result: " + str(np.isclose(slowResult, fastResult)) )
  
  # Test performance...
  import timeit # Yes, you can import any modules (including "illegal" ones) in __main__ for testing purposes!
  print( "Running SlowDotProduct 10000 times takes " + str( timeit.timeit("SlowDotProduct(x, y)", number = 10000, globals = globals() ) ) + " seconds." )
  print( "Running FastDotProduct 10000 times takes " + str( timeit.timeit("FastDotProduct(x, y)", number = 10000, globals = globals() ) ) + " seconds." )
  print( "Running SlowDotProduct 10000 times with different random inputs takes " + str( timeit.timeit("SlowDotProduct(np.random.rand(1000), np.random.rand(1000))", number = 10000, globals = globals() ) ) + " seconds." )
  print( "Running FastDotProduct 10000 times with different random inputs takes " + str( timeit.timeit("FastDotProduct(np.random.rand(1000), np.random.rand(1000))", number = 10000, globals = globals() ) ) + " seconds." )



  # [Deliverable 4 example test]
  print("\n\n[Deliverable 4]\n")
  N = 30
  x = np.random.rand(N)
  y = np.random.rand(N)
  NaN_idx = np.array([np.random.randint(N-1), np.random.randint(N-1)])
  x[NaN_idx[0]] = np.float64("nan")
  y[NaN_idx[1]] = np.float64("nan")

  # Create our matrix M of shape (N,N) as the outer product of x and y 
  # Note: we could've used, e.g., np.outer or np.einsum, here; 
  # instead, we'll show off some broadcasting and numpy's built-in product operator @
  M = x[:,np.newaxis] @ y[np.newaxis,:] 
  print( "Did you find the NaN? " + str( np.allclose(NaN_idx, CatchTheNaN(M)) ) )



  # [Deliverable 5 example test]
  print("\n\n[Deliverable 5]\n")
  # seed a random floating point vector of shape (N,) with values between a fraction of the min and max signed integer range
  N = 1000
  z = np.random.rand(N) * (np.iinfo(np.int16).max/100 - np.iinfo(np.int16).min/100) + np.iinfo(np.int16).min/100

  naive = LogSumExpNaive(z)
  robust = LogSumExpRobust(z)
  print( "Do the results match within floating point precision? " + str(np.isclose(naive, robust)) )
