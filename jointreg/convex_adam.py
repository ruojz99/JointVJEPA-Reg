import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LR = 3
ITER_SMOOTH_KERNEL = 7
ITER_SMOOTH_NUM = 5
GRID_SP_CONVEX = 6
DISP_HW = 4


def correlate(feat_fix, feat_mov, disp_hw, grid_sp, shape, ch):
    H, W, D = (int(s) for s in shape)
    with torch.no_grad():
        unfold = F.unfold(F.pad(feat_mov, (disp_hw,) * 6).squeeze(0), disp_hw * 2 + 1)
        unfold = unfold.view(ch, -1, (disp_hw * 2 + 1) ** 2, W // grid_sp, D // grid_sp)
    ssd = torch.zeros((disp_hw * 2 + 1) ** 3, H // grid_sp, W // grid_sp, D // grid_sp,
                      dtype=feat_fix.dtype, device=feat_fix.device)
    with torch.no_grad():
        for i in range(disp_hw * 2 + 1):
            s = (feat_fix.permute(1, 2, 0, 3, 4) - unfold[:, i:i + H // grid_sp]).pow(2).sum(0, keepdim=True)
            ssd[i::(disp_hw * 2 + 1)] = F.avg_pool3d(F.avg_pool3d(s.transpose(2, 1), 3, stride=1, padding=1),
                                                     3, stride=1, padding=1).squeeze(1)
        ssd = ssd.view(disp_hw * 2 + 1, disp_hw * 2 + 1, disp_hw * 2 + 1, H // grid_sp, W // grid_sp, D // grid_sp) \
                 .transpose(1, 0).reshape((disp_hw * 2 + 1) ** 3, H // grid_sp, W // grid_sp, D // grid_sp)
        ssd_argmin = torch.argmin(ssd, 0)
    return ssd, ssd_argmin


def coupled_convex(ssd, ssd_argmin, disp_mesh_t, grid_sp, shape):
    H, W, D = (int(s) for s in shape)
    size = (1, 3, H // grid_sp, W // grid_sp, D // grid_sp)
    disp_soft = F.avg_pool3d(disp_mesh_t.view(3, -1)[:, ssd_argmin.view(-1)].reshape(size), 3, padding=1, stride=1)
    for c in torch.tensor([0.003, 0.01, 0.03, 0.1, 0.3, 1]):
        argmin = torch.zeros_like(ssd_argmin)
        with torch.no_grad():
            for i in range(H // grid_sp):
                coupled = ssd[:, i] + c * (disp_mesh_t - disp_soft[:, :, i].view(3, 1, -1)).pow(2).sum(0) \
                    .view(-1, W // grid_sp, D // grid_sp)
                argmin[i] = torch.argmin(coupled, 0)
        disp_soft = F.avg_pool3d(disp_mesh_t.view(3, -1)[:, argmin.view(-1)].reshape(size), 3, padding=1, stride=1)
    return disp_soft


def inverse_consistency(disp1, disp2, iters=20):
    _, _, H, W, D = disp1.size()
    with torch.no_grad():
        d1i, d2i = disp1.clone(), disp2.clone()
        identity = F.affine_grid(torch.eye(3, 4).unsqueeze(0), (1, 1, H, W, D)).permute(0, 4, 1, 2, 3) \
            .to(disp1.device).to(disp1.dtype)
        for _ in range(iters):
            d1, d2 = d1i.clone(), d2i.clone()
            d1i = 0.5 * (d1 - F.grid_sample(d2, (identity + d1).permute(0, 2, 3, 4, 1)))
            d2i = 0.5 * (d2 - F.grid_sample(d1, (identity + d2).permute(0, 2, 3, 4, 1)))
    return d1i, d2i


def convex_adam(feat_fix, feat_mov, lambda_weight, grid_sp_adam, num_iter=100):
    """(X, Y, Z, C) fixed and moving features -> (3, X, Y, Z) displacement in voxels on the fixed grid,
    warped(x) = moving(x + u(x))."""
    fix = torch.from_numpy(feat_fix).float().permute(3, 0, 1, 2)[None].cuda()
    mov = torch.from_numpy(feat_mov).float().permute(3, 0, 1, 2)[None].cuda()
    H, W, D = fix.shape[2:]
    n_ch = fix.shape[1]


    gs, hw = GRID_SP_CONVEX, DISP_HW
    fix_s, mov_s = F.avg_pool3d(fix, gs, stride=gs), F.avg_pool3d(mov, gs, stride=gs)
    ssd, ssd_argmin = correlate(fix_s, mov_s, hw, gs, (H, W, D), n_ch)
    disp_mesh_t = F.affine_grid(hw * torch.eye(3, 4).cuda().half().unsqueeze(0),
                                (1, 1, hw * 2 + 1, hw * 2 + 1, hw * 2 + 1), align_corners=True) \
        .permute(0, 4, 1, 2, 3).reshape(3, -1, 1)
    disp_soft = coupled_convex(ssd, ssd_argmin, disp_mesh_t, gs, (H, W, D))
    scale = torch.tensor([H // gs - 1, W // gs - 1, D // gs - 1]).view(1, 3, 1, 1, 1).cuda().half() / 2
    ssd_, ssd_argmin_ = correlate(mov_s, fix_s, hw, gs, (H, W, D), n_ch)
    disp_soft_ = coupled_convex(ssd_, ssd_argmin_, disp_mesh_t, gs, (H, W, D))
    disp_ice, _ = inverse_consistency((disp_soft / scale).flip(1), (disp_soft_ / scale).flip(1), iters=15)
    disp_hr = F.interpolate(disp_ice.flip(1) * scale * gs, size=(H, W, D), mode='trilinear', align_corners=False)
    del ssd, ssd_, fix_s, mov_s


    g = grid_sp_adam
    size = (H // g, W // g, D // g)
    with torch.no_grad():
        pf, pm = F.avg_pool3d(fix, g, stride=g), F.avg_pool3d(mov, g, stride=g)
    disp_lr = F.interpolate(disp_hr, size=size, mode='trilinear', align_corners=False)
    net = nn.Sequential(nn.Conv3d(3, 1, size, bias=False))
    net[0].weight.data[:] = disp_lr.float().cpu().data / g
    net.cuda()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    grid0 = F.affine_grid(torch.eye(3, 4).unsqueeze(0).cuda(), (1, 1) + size, align_corners=False)
    pad = (ITER_SMOOTH_KERNEL - 1) // 2
    for _ in range(num_iter):
        opt.zero_grad()
        d = net[0].weight
        for _ in range(ITER_SMOOTH_NUM):
            d = F.avg_pool3d(d, ITER_SMOOTH_KERNEL, stride=1, padding=pad)
        d = d.permute(0, 2, 3, 4, 1)
        reg = lambda_weight * ((d[0, :, 1:, :] - d[0, :, :-1, :]) ** 2).mean() + \
            lambda_weight * ((d[0, 1:, :, :] - d[0, :-1, :, :]) ** 2).mean() + \
            lambda_weight * ((d[0, :, :, 1:] - d[0, :, :, :-1]) ** 2).mean()
        sc = torch.tensor([(s - 1) / 2 for s in size]).cuda().unsqueeze(0)
        grid = grid0.view(-1, 3).float() + (d.view(-1, 3) / sc).flip(1).float()
        warped = F.grid_sample(pm.float(), grid.view(1, *size, 3), align_corners=False, mode='bilinear')
        loss = (warped - pf).pow(2).mean(1).mean()
        (loss + reg).backward()
        opt.step()
    disp_hr = F.interpolate(d.detach().permute(0, 4, 1, 2, 3) * g, size=(H, W, D), mode='trilinear',
                            align_corners=False)
    out = disp_hr[0].cpu().half().numpy().astype(np.float32)         
    del fix, mov, pf, pm, net, opt
    torch.cuda.empty_cache()
    return out
