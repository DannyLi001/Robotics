import pickle
import torch
import sys
import time 
import numpy as np
import transforms3d as t3d
import matplotlib.pyplot as plt

def tic():
  return time.time()
def toc(tstart, nm=""):
  print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))

def read_data(fname):
  d = []
  with open(fname, 'rb') as f:
    if sys.version_info[0] < 3:
      d = pickle.load(f)
    else:
      d = pickle.load(f, encoding='latin1')  # needed for python 3
  return d

dataset="1"
ifile = "data/imu/imuRaw" + dataset + ".p"
vfile = "data/vicon/viconRot" + dataset + ".p"

ts = tic()
imud = read_data(ifile)
vicd = read_data(vfile)
toc(ts,"Data import")

ts = imud[0].copy()
tau = np.zeros(len(ts) - 1)
acc = imud[1:4].copy()
w = imud[4:].copy()

acc_sens = 300
gyro_sens = 3.33

def calibrate(ts, acc, w, vicd):
  acc_factor = 3300 / 1023 / acc_sens
  gyro_factor = 3300 / 1023 * np.pi / 180 / gyro_sens

  gyro_bias = np.mean(w[:, :10], axis=1)

  acc = acc * acc_factor 
  w = (w - gyro_bias[:, np.newaxis]) * gyro_factor

  tau = ts[1:] - ts[:-1]
  return acc, w, tau

acc, w, tau = calibrate(ts, acc, w, vicd)

q0 = np.array([1.0, 0.0, 0.0, 0.0])
qt = []
qt.append(q0)

euler = []
euler.append(np.array(t3d.euler.quat2euler(qt[-1])))

def fun_f(qt, tau_wt):
  return mul(qt, exp([0, tau_wt / 2]))

def exp(q):
  return np.exp(q[0]) * np.append(np.cos(np.linalg.norm(q[1:])), 
                                  (q[1:] / np.linalg.norm(q[1:]) * np.sin(np.linalg.norm(q[1:]))).flatten())

def mul(q, p):
  q = np.array(q)
  return np.concatenate(([q[0] * p[0] - q[1:] @ p[1:].T], q[0] * p[1:] + p[0] * q[1:] + np.cross(q[1:], p[1:])))

for i in range(len(tau)):
  qt.append(fun_f(qt[-1], tau[i] * w[:, i]))
  euler.append(np.array(t3d.euler.quat2euler(qt[-1])))

qt = np.array(qt)
euler = np.array(euler)

ground_truth_R = vicd['rots'].copy()
ts_truth = vicd['ts'][0]
truth_euler = []
for i in range(ground_truth_R.shape[2]):
  truth_euler.append(np.array(t3d.euler.mat2euler(ground_truth_R[:,:,i])))
truth_euler = np.array(truth_euler)


fig, ax = plt.subplots(3, 1, figsize=(5, 15))
# Plot X-axis values
ax[0].plot(ts - ts[0], euler[:,0], label='MEASURED', color='blue')
ax[0].plot(ts_truth - ts_truth[0], truth_euler[:,0], label='TRUTH', color='red')
ax[0].set_title('X-axis comparison')
ax[0].legend()

# Plot Y-axis values
ax[1].plot(ts - ts[0], euler[:,1], label='MEASURED', color='blue')
ax[1].plot(ts_truth - ts_truth[0], truth_euler[:,1], label='TRUTH', color='red')
ax[1].set_title('Y-axis comparison')
ax[1].legend()

# Plot Z-axis values
ax[2].plot(ts - ts[0], euler[:,2], label='MEASURED', color='blue')
ax[2].plot(ts_truth - ts_truth[0], truth_euler[:,2], label='TRUTH', color='red')
ax[2].set_title('Z-axis comparison')
ax[2].legend()

plt.tight_layout()
# plt.show()

def inv(q):
  tmp = q.copy()
  tmp[1:] *= -1
  return tmp / np.linalg.norm(q) ** 2

at = []
g = 9.8
for q in qt:
  a = mul(inv(q), mul(np.array([0,0,0,-g]), q))
  at.append(a)
  
at = np.array(at)

def cost(qt):
  first = 0
  second = 0
  for i in range(len(qt) - 1):
    first += 1/2 * np.linalg.norm(2 * np.nan_to_num(np.log(mul(inv(qt[i + 1]), qt[i])), nan=1e-6)) ** 2
  for i in range(1, len(qt)):
    second += 1/2 * np.linalg.norm(np.append(np.array([0]), acc[:,i]) - at[i]) ** 2
  
  return first + second

c = cost(qt)


# Convert numpy arrays to PyTorch tensors
qt_torch = torch.tensor(qt, dtype=torch.float32, requires_grad=True)
tau_torch = torch.tensor(tau, dtype=torch.float32)
w_torch = torch.tensor(w.T, dtype=torch.float32)  # Transpose to match dimensions
at_torch = torch.tensor(at, dtype=torch.float32)

# Define the cost function in PyTorch
def cost_function(qt_torch):
    first_term = 0
    second_term = 0
    
    for i in range(len(qt_torch) - 1):
        # Motion model error
        q_pred = fun_f(qt_torch[i], tau_torch[i] * w_torch[i])
        rel_rot = mul(inv(qt_torch[i + 1]), q_pred)
        first_term += 0.5 * torch.norm(2 * log(rel_rot)) ** 2
    
    for i in range(1, len(qt_torch)):
        # Observation model error
        a_pred = mul(inv(qt_torch[i]), mul(torch.tensor([0, 0, 0, -9.8]), qt_torch[i]))
        second_term += 0.5 * torch.norm(torch.tensor([0, at_torch[i]]) - a_pred) ** 2
    
    return first_term + second_term

# Define quaternion operations in PyTorch
def mul(q, p):
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    p0, p1, p2, p3 = p[0], p[1], p[2], p[3]
    return torch.tensor([
        q0 * p0 - q1 * p1 - q2 * p2 - q3 * p3,
        q0 * p1 + q1 * p0 + q2 * p3 - q3 * p2,
        q0 * p2 - q1 * p3 + q2 * p0 + q3 * p1,
        q0 * p3 + q1 * p2 - q2 * p1 + q3 * p0
    ])

def inv(q):
    return torch.tensor([q[0], -q[1], -q[2], -q[3]]) / torch.norm(q) ** 2

def log(q):
    norm_q = torch.norm(q[1:])
    return torch.tensor([0, q[1] / norm_q * torch.atan2(norm_q, q[0]), 
                         q[2] / norm_q * torch.atan2(norm_q, q[0]), 
                         q[3] / norm_q * torch.atan2(norm_q, q[0])])

def fun_f(qt, tau_wt):
    return mul(qt, exp(torch.tensor([0, tau_wt[0] / 2, tau_wt[1] / 2, tau_wt[2] / 2])))

def exp(q):
    norm_q = torch.norm(q[1:])
    return torch.exp(q[0]) * torch.tensor([torch.cos(norm_q), 
                                           q[1] / norm_q * torch.sin(norm_q), 
                                           q[2] / norm_q * torch.sin(norm_q), 
                                           q[3] / norm_q * torch.sin(norm_q)])

# Projection onto unit quaternion space
def project_to_unit_quaternion(q):
    return q / torch.norm(q)

# Optimization loop
learning_rate = 0.01
num_iterations = 100

for iteration in range(num_iterations):
    # Zero the gradients
    if qt_torch.grad is not None:
        qt_torch.grad.zero_()
    
    # Compute the cost
    cost = cost_function(qt_torch)
    
    # Backpropagation
    cost.backward()
    
    # Gradient descent step
    with torch.no_grad():
        qt_torch -= learning_rate * qt_torch.grad
    
    # Project back to unit quaternion space
    with torch.no_grad():
        for i in range(len(qt_torch)):
            qt_torch[i] = project_to_unit_quaternion(qt_torch[i])
    
    # Print the cost every 10 iterations
    if iteration % 10 == 0:
        print(f"Iteration {iteration}, Cost: {cost.item()}")

# Convert the optimized quaternions back to numpy
qt_optimized = qt_torch.detach().numpy()

# Plot the results
euler_optimized = []
for q in qt_optimized:
    euler_optimized.append(np.array(t3d.euler.quat2euler(q)))
euler_optimized = np.array(euler_optimized)

fig, ax = plt.subplots(3, 1, figsize=(5, 15))
ax[0].plot(ts - ts[0], euler_optimized[:, 0], label='OPTIMIZED', color='green')
ax[0].plot(ts_truth - ts_truth[0], truth_euler[:, 0], label='TRUTH', color='red')
ax[0].set_title('X-axis comparison')
ax[0].legend()

ax[1].plot(ts - ts[0], euler_optimized[:, 1], label='OPTIMIZED', color='green')
ax[1].plot(ts_truth - ts_truth[0], truth_euler[:, 1], label='TRUTH', color='red')
ax[1].set_title('Y-axis comparison')
ax[1].legend()

ax[2].plot(ts - ts[0], euler_optimized[:, 2], label='OPTIMIZED', color='green')
ax[2].plot(ts_truth - ts_truth[0], truth_euler[:, 2], label='TRUTH', color='red')
ax[2].set_title('Z-axis comparison')
ax[2].legend()

plt.tight_layout()
plt.show()