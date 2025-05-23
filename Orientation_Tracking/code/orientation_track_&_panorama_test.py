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

dataset="11"
ifile = "data/imu/imuRaw" + dataset + ".p"
vfile = "data/vicon/viconRot" + dataset + ".p"

ts = tic()
imud = read_data(ifile)
toc(ts,"Data import")

ts = torch.tensor(imud[0].copy())
tau = torch.zeros(len(ts) - 1)
acc = torch.tensor(imud[1:4].copy())
w = torch.tensor(imud[4:].copy())

acc_sens = 300
gyro_sens = 3.33

def calibrate(ts, acc, w):
  acc_factor = 3300 / 1023 / acc_sens
  gyro_factor = 3300 / 1023 * torch.pi / 180 / gyro_sens

  gyro_bias = torch.mean(w[:, :10], axis=1)
  acc_bias = torch.mean(acc[:, :10], axis=1)

  acc = (acc - acc_bias[:, torch.newaxis]) * acc_factor 
  acc[2,:] += 1
  w = (w - gyro_bias[:, torch.newaxis]) * gyro_factor

  tau = ts[1:] - ts[:-1]
  return acc, w, tau

acc, w, tau = calibrate(ts, acc, w)

q0 = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=float)
qt = []
qt.append(q0)

euler = []
euler.append(torch.tensor(t3d.euler.quat2euler(qt[-1])))

def fun_f(qt, tau_wt):
  return mul(qt, exp(torch.cat([torch.tensor([0]), tau_wt / 2])))

def exp(q):
  return torch.exp(q[0]) * torch.cat([torch.tensor([torch.cos(torch.linalg.norm(q[1:]))]), (q[1:] / torch.linalg.norm(q[1:]) * torch.sin(torch.linalg.norm(q[1:])))])

def mul(q, p):
  return torch.cat([torch.tensor([q[0] * p[0] - q[1:] @ p[1:].T]), q[0] * p[1:] + p[0] * q[1:] + torch.cross(q[1:], p[1:])])

for i in range(len(tau)):
  qt.append(fun_f(qt[-1], tau[i] * w[:, i]))
  euler.append(torch.tensor(t3d.euler.quat2euler(qt[-1])))

qt = torch.stack(qt)
qt.requires_grad = True
euler = torch.stack(euler)



fig, ax = plt.subplots(3, 1, figsize=(5, 15))
# Plot X-axis values
ax[0].plot(ts - ts[0], euler[:,0], label='MEASURED', color='blue')
ax[0].set_title('Roll (X-axis) comparison')
ax[0].legend()

# Plot Y-axis values
ax[1].plot(ts - ts[0], euler[:,1], label='MEASURED', color='blue')
ax[1].set_title('Pitch (Y-axis) comparison')
ax[1].legend()

# Plot Z-axis values
ax[2].plot(ts - ts[0], euler[:,2], label='MEASURED', color='blue')
ax[2].set_title('Yaw (Z-axis) comparison')
ax[2].legend()

plt.tight_layout()
plt.savefig("fig/Measure_Euler_" + dataset + ".png")
plt.show()

def inv(q):
  tmp = q.clone()
  tmp[1:] *= -1
  return tmp / torch.linalg.norm(q) ** 2

at = []
g = 9.8
for q in qt:
  a = mul(inv(q), mul(torch.tensor([0,0,0,1], dtype=float), q))
  at.append(a)

at = torch.stack(at)

fig, ax = plt.subplots(3, 1, figsize=(5, 15))
# Plot X-axis values
ax[0].plot(ts - ts[0], at[:,1].detach().numpy(), label='MEASURED', color='blue')
ax[0].plot(ts - ts[0], acc[0,:], label='TRUTH', color='red')
ax[0].set_title('Acc X-axis comparison')
ax[0].legend()

# Plot Y-axis values
ax[1].plot(ts - ts[0], at[:,2].detach().numpy(), label='MEASURED', color='blue')
ax[1].plot(ts - ts[0], acc[1,:], label='TRUTH', color='red')
ax[1].set_title('Acc Y-axis comparison')
ax[1].legend()

# Plot Z-axis values
ax[2].plot(ts - ts[0], at[:,3].detach().numpy(), label='MEASURED', color='blue')
ax[2].plot(ts - ts[0], acc[2,:], label='TRUTH', color='red')
ax[2].set_title('Acc Z-axis comparison')
ax[2].legend()

plt.tight_layout()
plt.savefig("fig/Measure_vs_Truth_Acc_" + dataset + ".png")
plt.show()

learning_rate = 0.008
num_iterations = 150
# num_iterations = 0

def mat_mul(a,b):    
  w1, x1, y1, z1 = a[:,0], a[:,1], a[:,2], a[:,3]
  w2, x2, y2, z2 = b[:,0], b[:,1], b[:,2], b[:,3]
  
  w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
  x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
  y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
  z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
  
  return torch.vstack([w, x, y, z]).T

def mat_inv(mat):
    q_conj = mat.clone()
    q_conj[:, 1:] = -q_conj[:, 1:]  
    norm_squared = torch.sum(mat**2, dim=1, keepdim=True)
    
    return q_conj / norm_squared

def mat_exp(a):
  return torch.exp(a[0]) * torch.cat([torch.cos(torch.norm(a[1:], dim=0)).unsqueeze(0), 
                                      a[1:] / torch.norm(a[1:], dim=0) * torch.sin(torch.norm(a[1:], dim=0))])

def mat_f(a,b):
  zeros_col = torch.zeros(1, b.size(1), dtype=torch.float32)
  b = torch.cat((zeros_col, b), dim=0)
  tmp = mat_exp(b/2)
  return mat_mul(a, tmp.T)

def mat_log(q):
  norm = torch.norm(q, dim=1, keepdim=True)
  q_normalized = q / norm
  
  w = q_normalized[:, 0]
  w = torch.clamp(w, -1 + 1e-8, 1 - 1e-8)
  theta = torch.acos(w)
  theta = torch.where(torch.abs(theta) < 1e-8, torch.zeros_like(theta), theta)
  
  qv_norm = torch.norm(q[:, 1:], dim=1, keepdim=True)
  qv_norm = torch.where(qv_norm == 0, torch.tensor(1e-8), qv_norm)
  log_qv = q[:, 1:] / qv_norm  # Unit vector part of the rotation axis
  log_q = theta.unsqueeze(1) * log_qv  # Element-wise multiplication
  
  return torch.cat((torch.log(norm), log_q), dim=1)

def mat_h(qt):
  temp = torch.zeros_like(qt)
  temp[:,-1] += 1
  return mat_mul(mat_inv(qt), mat_mul(temp, qt))

def cost(qt):
  first = mat_mul(mat_inv(qt[1:]),mat_f(qt[:-1], torch.mul(tau, w[:,:-1])))
  first = 1/2 * torch.sum(torch.norm(mat_log(first) * 2, dim=1)**2)
  
  zeros_col = torch.zeros(1, acc.size(1), dtype=torch.float32)
  tmp = torch.cat((zeros_col, acc), dim=0)
  second = 1/2 * torch.sum(torch.norm(tmp.T[1:] - mat_h(qt[1:]), dim=1)**2)
  return first + second


def project_to_unit_quaternion(q):
    return q / torch.norm(q)

for iteration in range(num_iterations):

  if qt.grad is not None:
    qt.grad.zero_()

  c = cost(qt)

  c.backward(retain_graph=True)

  with torch.no_grad():
    qt = qt - learning_rate * qt.grad  # Simple gradient descent step
    for i in range(len(qt)):
        qt[i] = project_to_unit_quaternion(qt[i])

  qt = qt.detach().clone().requires_grad_(True)

  if iteration % 10 == 0:
    print(f"Iteration {iteration}, Cost: {c.item()}")



qt_optimized = qt.detach().numpy()

euler_optimized = []
rot_mat_optimized = []
for q in qt_optimized:
  euler_optimized.append(torch.tensor(t3d.euler.quat2euler(q)))
  rot_mat_optimized.append(torch.tensor(t3d.euler.quat2mat(q)))
euler_optimized = torch.vstack(euler_optimized)
rot_mat_optimized = torch.stack(rot_mat_optimized)

fig, ax = plt.subplots(3, 1, figsize=(5, 15))
ax[0].plot(ts - ts[0], euler[:,0], label='MEASURED', color='blue')
ax[0].plot(ts - ts[0], euler_optimized[:, 0], label='OPTIMIZED', color='green')
ax[0].set_title('Roll (X-axis) comparison')
ax[0].legend()

ax[1].plot(ts - ts[0], euler[:,1], label='MEASURED', color='blue')
ax[1].plot(ts - ts[0], euler_optimized[:, 1], label='OPTIMIZED', color='green')
ax[1].set_title('Pitch (Y-axis) comparison')
ax[1].legend()

ax[2].plot(ts - ts[0], euler[:,2], label='MEASURED', color='blue')
ax[2].plot(ts - ts[0], euler_optimized[:, 2], label='OPTIMIZED', color='green')
ax[2].set_title('Yaw (Z-axis) comparison')
ax[2].legend()

plt.tight_layout()
plt.savefig("fig/Measure_vs_Optm_Euler_" + dataset + ".png")
plt.show()






cfile = "data/cam/cam" + dataset + ".p"
camd = read_data(cfile)

images = camd['cam'].T
img_ts = camd['ts'][0]


def create_panorama(images, image_timestamps, orientations, orientation_timestamps, h_fov, v_fov):

    panorama_width = 2048  # Width of the unwrapped cylinder (2π radians)
    panorama_height = 1024  # Height of the unwrapped cylinder (π radians)
    panorama = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)

    h_fov_rad = np.radians(h_fov)
    v_fov_rad = np.radians(v_fov)

    for idx, image in enumerate(images):
        if idx > images.shape[0]/4*3:
          break
        image = image.T
        image_time = image_timestamps[idx]
        orientation_idx = np.searchsorted(orientation_timestamps, image_time, side='right') - 1
        if orientation_idx < 0:
            orientation_idx = 0

        R = orientations[orientation_idx]

        H, W = image.shape[:2]

        # Create a grid of pixel coordinates
        i, j = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

        # Compute longitude (λ) and latitude (ϕ) for each pixel
        lambda_ = (j / W - 0.5) * h_fov_rad
        phi = (0.5 - i / H) * v_fov_rad

        # Convert spherical coordinates to Cartesian coordinates
        x = np.cos(phi) * np.cos(lambda_)
        y = np.cos(phi) * np.sin(lambda_)
        z = np.sin(phi)

        # Stack into a (3, H*W) matrix for vectorized rotation
        xyz = np.stack([x, y, z], axis=0).reshape(3, -1)

        # Rotate to the world frame using the rotation matrix R
        adjustment_R = np.array([
            [1, 0, 0],  
            [0, -1, 0],  
            [0, 0, 1]   
        ])
        xyz_world = R @ (adjustment_R @ xyz)

        # Convert back to spherical coordinates
        x_w, y_w, z_w = xyz_world
        lambda_w = np.arctan2(y_w, x_w).numpy()
        phi_w = np.arcsin(z_w).numpy()

        # Map spherical coordinates to panorama coordinates
        panorama_x = ((lambda_w + np.pi) / (2 * np.pi) * panorama_width).astype(int)
        panorama_y = ((phi_w + np.pi / 2) / np.pi * panorama_height).astype(int)

        # Clip coordinates to ensure they are within the panorama bounds
        panorama_x = np.clip(panorama_x, 0, panorama_width - 1)
        panorama_y = np.clip(panorama_y, 0, panorama_height - 1)

        # Flatten the image and map pixels to the panorama
        image_flat = image.reshape(-1, 3)
        panorama[panorama_y, panorama_x] = image_flat

    return panorama

h_fov = 60  # Horizontal field of view in degrees
v_fov = 45  # Vertical field of view in degrees

panorama = create_panorama(images, img_ts, rot_mat_optimized, ts, h_fov, v_fov)

# Save or display the panorama
plt.imshow(panorama, origin='lower')
plt.title("Panorama for dataset " + str(dataset))
plt.savefig("fig/Panorama_" + dataset + ".png")
plt.show()