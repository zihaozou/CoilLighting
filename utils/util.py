import torch
import h5py
import math
import shutil
import os
import numpy as np
import torch.nn as nn

evaluateSnr = lambda x, xhat: 20*np.log10(np.linalg.norm(x.flatten('F'))/np.linalg.norm(x.flatten('F')-xhat.flatten('F')))

def h5py2mat(data):
    result = np.array(data)
    print(result.shape)

    if len(result.shape) == 3 and result.shape[0] > result.shape[1]:
        result = result.transpose([1,0,2])
    elif len(result.shape) == 3 and result.shape[1] < result.shape[2]:
        result = result.transpose([2,1,0])
    elif len(result.shape) == 3 and result.shape[1] > result.shape[2]:
        result = result.transpose([2,1,0])        
    print(result.shape)    
    return result

def complex_multiple_torch(x: torch.Tensor, y: torch.Tensor):
    x_real, x_imag = torch.unbind(x, -1)
    y_real, y_imag = torch.unbind(y, -1)

    res_real = torch.mul(x_real, y_real) - torch.mul(x_imag, y_imag)
    res_imag = torch.mul(x_real, y_imag) + torch.mul(x_imag, y_real)

    return torch.stack([res_real, res_imag], -1)

###################
# Read Images
###################
def np2torch_complex(array: np.ndarray):
    return torch.stack([torch.from_numpy(array.real), torch.from_numpy(array.imag)], -1)


def addwgn_torch(x: torch.Tensor, inputSnr,minV,maxV):
    noiseNorm = torch.norm(x.flatten() * 10 ** (-inputSnr / 20))

    # xBool = np.isreal(x)
    # real = True
    # for e in np.nditer(xBool):
    #     if not e:
    #         real = False
    # if real:
    #     noise = np.random.randn(np.shape(x)[0], np.shape(x)[1])
    # else:
    #     noise = np.random.randn(np.shape(x)[0], np.shape(x)[1]) + \
    #         1j * np.random.randn(np.shape(x)[0], np.shape(x)[1])

    noise = torch.randn(x.shape[-2], x.shape[-1]).to(x.device)
    noise = noise / torch.norm(noise.flatten()) * noiseNorm
    
    rec_y = x + noise

    return rec_y,torch.clamp(rec_y,min=minV,max=maxV)

def compare_snr(img_test, img_true):
    return 20 * torch.log10(torch.norm(img_true.flatten()) / torch.norm(img_true.flatten() - img_test.flatten()))



def rsnr_cal(rec,oracle):
    "regressed SNR"
    sumP    =        sum(oracle.reshape(-1))
    sumI    =        sum(rec.reshape(-1))
    sumIP   =        sum( oracle.reshape(-1) * rec.reshape(-1) )
    sumI2   =        sum(rec.reshape(-1)**2)
    A       =        np.matrix([[sumI2, sumI],[sumI, oracle.size]])
    b       =        np.matrix([[sumIP],[sumP]])
    c       =        np.linalg.inv(A)*b #(A)\b
    rec     =        c[0,0]*rec+c[1,0]
    err     =        sum((oracle.reshape(-1)-rec.reshape(-1))**2)
    SNR     =        10.0*np.log10(sum(oracle.reshape(-1)**2)/err)

    if np.isnan(SNR):
        SNR=0.0
    return SNR

def compute_rsnr(x, xhat):
    if len(x.shape) == 2:
        avg_rsnr = rsnr_cal(xhat, x)
    elif len(x.shape) == 3 and x.shape[0] < x.shape[1]:   
        rsnr = np.zeros([1,x.shape[0]])
        for num_imgs in range(0,x.shape[0]):
            rsnr[:,num_imgs] = rsnr_cal(xhat, x)
        avg_rsnr = np.mean(rsnr)
    return avg_rsnr


def get_vars(network):
    lst_vars = []
    num_count = 0
    for para in network.parameters():
        num_count += 1
        print('Layer %d' % num_count)
        print(para.size())
        lst_vars.append(para)
    return lst_vars 

# def compute_rsnr(x, xhat):

#     evaluateSnr = lambda x, xhat: 20*np.log10(np.linalg.norm(x.flatten('F'))/np.linalg.norm(x.flatten('F')-xhat.flatten('F')))
    
#     if len(x.shape) == 2:
#         A = np.zeros((2, 2))
#         A[0, 0] = np.sum(xhat.flatten('F')**2)
#         A[0, 1] = np.sum(xhat.flatten('F'))
#         A[1, 0] = A[0, 1]
#         A[1, 1] = x.size

#         b = np.zeros((2, 1))
#         b[0] = np.sum(x.flatten('F') * xhat.flatten('F'))
#         b[1] = np.sum(x.flatten('F'))
#         try:
#             c = np.matmul(np.linalg.inv(A), b)
#         except np.linalg.LinAlgError:
#             c = [0, 0]
#             print('xhat is all zeros.')
#         avg_rsnr = evaluateSnr(x, c[0]*xhat+c[1])
#     elif len(x.shape) == 3 and x.shape[0] < x.shape[1]:
#         rsnr = np.zeros([1,x.shape[0]])
#         for num_imgs in range(0,x.shape[0]):
#             A = np.zeros((2, 2))
#             A[0, 0] = np.sum(xhat.flatten('F')**2)
#             A[0, 1] = np.sum(xhat.flatten('F'))
#             A[1, 0] = A[0, 1]
#             A[1, 1] = x.size
#             b = np.zeros((2, 1))
#             b[0] = np.sum(x.flatten('F') * xhat.flatten('F'))
#             b[1] = np.sum(x.flatten('F'))
#             try:
#                 c = np.matmul(np.linalg.inv(A), b)
#             except np.linalg.LinAlgError:  
#                 c = [0, 0]
#                 print('xhat is all zeros.')
#             rsnr[:,num_imgs] = evaluateSnr(x, c[0]*xhat+c[1])
#         avg_rsnr = np.mean(rsnr)
#     else:
#         avg_rsnr = np.zeros([1,1])
#     return avg_rsnr

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant_(m.bias.data, 0.0)
        
def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant_(m.bias.data, 0.0)


def copytree(src=None, dst=None, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)
                
def data_augmentation(image, mode):
    out = image
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return out                
              

def powerIter(A, imgSize, iters=100, tol=1e-6, verbose=False):
    # compute singular value for A'*A
    # A should be a function (lambda:x)
    x = np.random.randn(imgSize[0], imgSize[1])
    x = x / np.linalg.norm(x.flatten('F'))
    lam = 1
    for i in range(iters):
        # apply Ax
        xnext = A(x)
        # xnext' * x / norm(x)^2
        lamNext = np.dot(xnext.flatten('F'), x.flatten('F')) / np.linalg.norm(x.flatten('F')) ** 2
        # only take the real part
        lamNext = lamNext.real
        # normalize xnext
        xnext = xnext / np.linalg.norm(xnext.flatten('F'))
        # compute relative difference
        relDiff = np.abs(lamNext - lam) / np.abs(lam)
        x = xnext
        lam = lamNext
        # verbose
        if verbose:
            print('[{}/{}] lam = {}, relative Diff = {:0.4f}'.format(i, iter, lam, relDiff))
        # stopping criterion
        if relDiff < tol:
            break
    return lam

def powerIter_torch(A, imgSize, iters=100, tol=1e-6, verbose=False):
    # compute singular value for A'*A
    # A should be a function (lambda:x)
    x = np.random.randn(imgSize[0], imgSize[1])
    x = x / np.linalg.norm(x.flatten('F'))
    lam = 1
    for i in range(iters):
        # apply Ax
        xnext = A(x)
        # xnext' * x / norm(x)^2
        lamNext = np.dot(xnext.flatten('F'), x.flatten('F')) / np.linalg.norm(x.flatten('F')) ** 2
        # only take the real part
        lamNext = lamNext.real
        # normalize xnext
        xnext = xnext / np.linalg.norm(xnext.flatten('F'))
        # compute relative difference
        relDiff = np.abs(lamNext - lam) / np.abs(lam)
        x = xnext
        lam = lamNext
        # verbose
        if verbose:
            print('[{}/{}] lam = {}, relative Diff = {:0.4f}'.format(i, iter, lam, relDiff))
        # stopping criterion
        if relDiff < tol:
            break
    return lam