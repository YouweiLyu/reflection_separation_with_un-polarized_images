import os 
import numpy as np

def makeFile(f):
    if not os.path.exists(f):
        os.makedirs(f)
    #else:  raise Exception('Rendered image directory %s is already existed!!!' % directory)

def makeFiles(f_list):
    for f in f_list:
        makeFile(f)

def emptyFile(name):
    with open(name, 'w') as f:
        f.write(' ')

def dictToString(dicts, start='\t', end='\n'):
    strs = '' 
    for k, v in sorted(dicts.items()):
        strs += '%s%s: %s%s' % (start, str(k), str(v), end) 
    return strs

def checkIfInList(list1, list2):
    contains = []
    for l1 in list1:
        for l2 in list2:
            if l1 in l2.lower():
                contains.append(l1)
                break
    return contains

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def readList(list_path,ignore_head=False, sort=False):
    lists = []
    with open(list_path) as f:
        lists = f.read().splitlines()
    if ignore_head:
        lists = lists[1:]
    if sort:
        lists.sort(key=natural_keys)
    return lists

def xi(x):
    def theta_t(xxx):
        return np.arcsin(np.sin(xxx) / 1.474)
    theta_plus = x + theta_t(x)
    theta_minus = x - theta_t(x)
    return 2*(np.sin(theta_minus) ** 2)/(np.sin(theta_minus)**2 + np.sin(theta_plus)**2) + \
        2*(np.tan(theta_minus) ** 2)/(np.tan(theta_minus)**2 + np.tan(theta_plus)**2)

def zeta(x, phi_perp, phi_0):
    def theta_t(xxx):
        return np.arcsin(np.sin(xxx) / 1.474)
    theta_plus = x + theta_t(x)
    theta_minus = x - theta_t(x)
    return 2 * (np.sin(theta_minus) ** 2) / (np.sin(theta_minus) ** 2 + np.sin(theta_plus) ** 2) * (np.cos(phi_0 - phi_perp))**2 + \
        2 * (np.tan(theta_minus) ** 2) / (np.tan(theta_minus) ** 2 + np.tan(theta_plus) ** 2) * (np.sin(phi_0 - phi_perp))**2

def para_mesh_matrix_former(h, w, alpha, beta):
    z0 = 1
    c_x, c_y = w / 2, h / 2
    f_x, f_y = w * 1.4, w * 1.4
    kk = np.tan(alpha)
    xx, yy = np.meshgrid(np.linspace(0, w - 1, w), np.linspace(0, h - 1, h))
    denominator = (f_x * np.cos(beta) + kk * (xx - c_x) - np.sin(beta) * (yy - c_y)) / z0 / np.cos(beta)
    denominator[denominator == 0] = 0.000001
    glass_x = (xx - c_x) / denominator
    glass_y = (yy - c_y) / denominator
    glass_z = f_x / denominator
    normal_x, normal_y, normal_z = kk, -np.sin(beta), np.cos(beta)
    AOI = cal_aoi(glass_x, glass_y, glass_z, normal_x, normal_y, normal_z)
    AOI[AOI == 0] = 0.000001
    phi = cal_phi(glass_x, glass_y, glass_z, normal_x, normal_y, normal_z)
    return AOI, phi

def cal_aoi(x, y, z, nx, ny, nz):
    aoi_rad = np.arccos((np.abs(x*nx+y*ny+z*nz)/(np.sqrt(x**2+y**2+z**2)*np.sqrt(nx**2+ny**2+nz**2))).clip(-1, 1))
    return aoi_rad

def cal_phi(x, y, z, nx, ny, nz):
    poi_normal_x, poi_normal_y = y * nz - ny * z, z * nx - x * nz
    phi_rad = np.arctan2(poi_normal_y, poi_normal_x)
    return phi_rad
