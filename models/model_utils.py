import os
import torch
import torch.nn as nn

def getRefInput(args, para, pre_input):
    phi_0 = args.phi_0
    h, w = pre_input.shape[2], pre_input.shape[3]
    alpha, beta = para[:, 0], para[:, 1]

    if args.grayscale:
        unpol, pol = pre_input[:, :1, :, :], pre_input[:, 1:2, :, :]
    else:
        unpol, pol = pre_input[:, :3, :, :], pre_input[:, 3:6, :, :]

    xi, zeta = para_module(h, w, alpha, beta, phi_0)
    est_r = (2 * ((2 - xi) * pol + (zeta - 1) * unpol) / (2 * zeta - xi)).clamp(0, 1)
    est_b = (2 * (zeta * unpol - xi * pol) / (2 * zeta - xi)).clamp(0, 1)
    input_list = [pre_input]
    if args.xi_zeta:
        input_list.append(xi)
        input_list.append(zeta)
    if args.est_r_b:
        input_list.append(est_r)
        input_list.append(est_b)
    input = torch.cat(input_list, 1)
    return input

def getRealRefInput(args, para, sample):
    unpol, pol00 = sample[0].cuda(non_blocking=True), sample[1].cuda(non_blocking=True) 
    phi_0 = args.phi_0
    h, w = unpol.shape[2], unpol.shape[3]
    alpha, beta = para[:, 0], para[:, 1]
    xi, zeta = para_module(h, w, alpha, beta, phi_0)
    xi, zeta = xi.type(torch.cuda.DoubleTensor), zeta.type(torch.cuda.DoubleTensor)
    est_r = (2 * ((2 - xi) * pol00 + (zeta - 1) * unpol) / (2 * zeta - xi)).clamp(0, 1)
    est_b = (2 * (zeta * unpol - xi * pol00) / (2 * zeta - xi)).clamp(0, 1)
    input_list = [unpol, pol00]
    if args.xi_zeta:
        input_list.append(xi)
        input_list.append(zeta)
    if args.est_r_b:
        input_list.append(est_r)
        input_list.append(est_b)
    input = torch.cat(input_list, 1).type(torch.cuda.FloatTensor)
    return input

def parseEstData(args, sample, timer=None, split='train'):
    if args.cuda:
        input_tensor = torch.cat([sample[0].cuda(non_blocking=True), sample[1].cuda(non_blocking=True)], 1)
        para_gt = sample[2].cuda(non_blocking=True)[:, :, 0]
        if timer: timer.updateTime('ToGPU')
    else:
        input_tensor = torch.cat([sample[0], sample[1]], 1)
        para_gt = sample[2][:, :, 0]
        if timer: timer.updateTime('ToCPU')
    # data = {'input': input_tensor, 'gt': para_gt}
    return input_tensor, para_gt

def parseRefData(args, sample, timer=None, split='train'):
    if args.cuda:
        input_tensor = torch.cat([sample[0].cuda(non_blocking=True), sample[1].cuda(non_blocking=True)], 1)
        rb_gt = torch.cat((sample[2].cuda(non_blocking=True), sample[3].cuda(non_blocking=True)), 1)
        if timer: timer.updateTime('ToGPU')
    else:
        input_tensor = torch.cat([sample[0], sample[1]], 1)
        rb_gt = torch.cat((sample[2], sample[3]), 1)
        if timer: timer.updateTime('ToCPU')
    return input_tensor, rb_gt

def parseRealData(args, sample):
    c_h, c_w = args.fc_height/2, args.fc_width/2
    if args.cuda:
        unpol, pol00 = sample[0].cuda(non_blocking=True), sample[1].cuda(non_blocking=True) 
        unpol_est, pol00_est = unpol.clone(), pol00.clone()
        h, w = unpol_est.shape[2], unpol_est.shape[3]
        unpol_est = unpol_est[:, :, int(h/2-c_h):int(h/2+c_h), int(w/2-c_w):int(w/2+c_w)]
        pol00_est = pol00_est[:, :, int(h/2-c_h):int(h/2+c_h), int(w/2-c_w):int(w/2+c_w)]
        pol00_est /= unpol_est.max()
        unpol_est /= unpol_est.max()
        input_tensor = torch.cat([unpol_est, pol00_est], 1).type(torch.cuda.FloatTensor)
    else:
        unpol, pol00 = sample[0], sample[1] 
        unpol_est, pol00_est = unpol.copy(), pol00.copy()
        h, w = unpol.shape[2], unpol.shape[3]
        unpol_est = unpol_est[int(h/2-c_h):int(h/2+c_h), int(w/2-c_w):int(w/2+c_w), :]
        pol00_est = pol00_est[int(h/2-c_h):int(h/2+c_h), int(w/2-c_w):int(w/2+c_w), :]
        pol00_est /= unpol_est.max()
        unpol_est /= unpol_est.max()
        input_tensor = torch.cat([unpol_est, pol00_est], 1).type(torch.FloatTensor)
    return input_tensor
    
        
def RefInputChanel(args):
    if args.grayscale:
        in_c = 2
        in_c += 2 if args.xi_zeta else in_c == in_c
        in_c += 2 if args.est_r_b else in_c == in_c
    else:
        in_c = 6
        in_c += 2 if args.xi_zeta else in_c == in_c
        in_c += 6 if args.est_r_b else in_c == in_c

    return in_c

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp

def loadCheckpoint(path, model, cuda=True):
    if cuda:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])

def saveCheckpoint(save_path, model_name, epoch=-1, model=None, optimizer=None, records=None, args=None):
    state   = {'state_dict': model.state_dict(), 'model': model_name}
    records = {'epoch': epoch, 'optimizer':optimizer.state_dict(), 'records': records} # 'args': args}
    torch.save(state,   os.path.join(save_path, 'checkpoint_EPC({}).pth.tar'.format(epoch)))
    torch.save(records, os.path.join(save_path, 'checkpoint_EPC({})_rec.pth.tar'.format(epoch)))

def para_module(h, w, alpha, beta, phi_0):
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    batch_num = alpha.size(0)
    
    alpha = torch.stack([torch.stack([alpha]*h, dim=1)]*w, dim=2)
    beta = torch.stack([torch.stack([beta]*h, dim=1)]*w, dim=2)

    c_x, c_y = w / 2, h / 2
    f_x, f_y = w * 1.4, w * 1.4
    kk = torch.tan(alpha)
    xx, yy = torch.meshgrid([torch.linspace(0, w - 1, steps=w), torch.linspace(0, h - 1, steps=h)])
    xx, yy = torch.stack([xx.transpose(0, 1)]*batch_num), torch.stack([yy.transpose(0, 1)]*batch_num)

    glass_x = (xx - c_x)
    glass_y = (yy - c_y)
    glass_z = f_x
    
    normal_x, normal_y, normal_z = kk, -torch.sin(beta), torch.cos(beta)

    AOI = cal_tensor_aoi_3d(glass_x, glass_y, glass_z, normal_x, normal_y, normal_z)
    AOI[AOI == 0] = 0.000001
    PHI_PERP = cal_tensor_phi(glass_x, glass_y, glass_z, normal_x, normal_y, normal_z)
    # phi_out = torch.atan(torch.tan(phi_out))
    xi, zeta = tensor_xi(AOI), tensor_zeta(AOI, PHI_PERP, phi_0)
    zeta[zeta == xi / 2] += 0.000001
    xi, zeta = torch.unsqueeze(xi, 1), torch.unsqueeze(zeta, 1)
    return xi, zeta

def tensor_xi(x):
    def theta_t(xxx):
        return torch.asin(torch.sin(xxx) / 1.474)
    theta_plus = x + theta_t(x)
    theta_minus = x - theta_t(x)
    return 2*(torch.sin(theta_minus) ** 2)/(torch.sin(theta_minus)**2 + torch.sin(theta_plus)**2) + \
        2*(torch.tan(theta_minus) ** 2)/(torch.tan(theta_minus)**2 + torch.tan(theta_plus)**2)

def tensor_zeta(x, phi_perp, phi_0):
    def theta_t(xxx):
        return torch.asin(torch.sin(xxx) / 1.474)
    theta_plus = x + theta_t(x)
    theta_minus = x - theta_t(x)
    return 2 * (torch.sin(theta_minus) ** 2) / (torch.sin(theta_minus) ** 2 + torch.sin(theta_plus) ** 2) * (torch.cos(phi_0 - phi_perp))**2 + \
        2 * (torch.tan(theta_minus) ** 2) / (torch.tan(theta_minus) ** 2 + torch.tan(theta_plus) ** 2) * (torch.sin(phi_0 - phi_perp))**2

def cal_tensor_aoi_3d(x, y, z, nx, ny, nz):
    aoi_rad = torch.acos((torch.abs(x * nx + y * ny + z * nz) / (torch.sqrt(x ** 2 + y ** 2 + z ** 2) * torch.sqrt(nx ** 2 + ny ** 2 + nz ** 2))).clamp(-1, 1))
    return aoi_rad

def cal_tensor_phi(x, y, z, nx, ny, nz):
    poi_normal_x, poi_normal_y = y * nz - ny * z, z * nx - x * nz
    phi_rad = torch.atan2(poi_normal_y, poi_normal_x)
    return phi_rad

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )
def fully_con(batchNorm, in_neurons, out_neurons):    
    return nn.Sequential(
        nn.Linear(in_neurons, int(in_neurons/4)),
        nn.LeakyReLU(0.1,inplace=True),
        nn.Linear(int(in_neurons/4), int(in_neurons/16)),
        nn.LeakyReLU(0.1,inplace=True),
        nn.Linear(int(in_neurons/16), out_neurons),
    )

def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1,inplace=True)
    )