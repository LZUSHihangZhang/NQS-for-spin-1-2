import torch
import math
from torch.autograd.functional import jacobian

device = torch.device('cuda')
dtype1 = torch.float32
dtype2 = torch.complex64


def Neel_state(Lx, Ly):
    state = torch.ones((Lx, Ly), device=device, dtype=dtype1)
    for i in range(Lx):
        for j in range(Ly):
            if (i + j) % 2 == 0:
                state[i, j] = 1
            else:
                state[i, j] = -1
    return state

D6 = torch.tensor([
    [0, 1, 2, 3, 4, 5, 6],  # e->e:恒等变换
    [0, 6, 1, 2, 3, 4, 5],  # r
    [0, 5, 6, 1, 2, 3, 4],  # r2
    [0, 4, 5, 6, 1, 2, 3],  # r3
    [0, 3, 4, 5, 6, 1, 2],  # r4
    [0, 2, 3, 4, 5, 6, 1],  # r5
    [0, 1, 6, 5, 4, 3, 2],  # s
    [0, 2, 1, 6, 5, 4, 3],  # sr
    [0, 3, 2, 1, 6, 5, 4],  # sr2
    [0, 4, 3, 2, 1, 6, 5],  # sr3
    [0, 5, 4, 3, 2, 1, 6],  # sr4
    [0, 6, 5, 4, 3, 2, 1]  # sr5
])

class Triangle:
    def __init__(self, lx, ly):
        """b：patch大小（你的情况是7，即自身+6个邻居）
           r：每个头的输出维度（你设为11），满足 r = d / h
           h：注意力头数（你设为200）
           d：embedding维度，满足 d = h * r = 200 * 11 = 2200
           N：序列长度（即格点数，lx*ly）
           K0：复数层隐藏单元数（你设为100）"""
        self.lx = lx
        self.ly = ly
        '''定义卷积网络的超参数'''
        self.nl = 4  # 网络的深度
        self.d = 12  # 卷积是融合维度
        self.h = 3     # 注意力的头数
        self.groups = 12
        self.r = self.d // self.h
        self.N = self.lx * self.ly  # batch的个数
        self.b = 7
        self.K0 = 30
        self.beta0 = 0.995
        self.mu = 0.95
        '''求解逆矩阵的正则化系数'''
        self.lam = 0.001

        """定义卷积层的参数"""
        '''定义虚数卷积层的参数'''
        self.V = torch.randn(self.nl, self.h, self.d, self.r, device=device, dtype=dtype1)
        self.K = torch.randn(self.h, self.b, self.d, device=device, dtype=dtype1)
        self.Q = torch.randn(self.h, self.b, self.d, device=device, dtype=dtype1)
        self.emb_matrix = torch.randn(self.b, self.d, device=device, dtype=dtype1)
        self.concate = torch.randn(self.nl,self.h, self.r, self.d, device=device, dtype=dtype1)
        self.W0 = torch.randn(self.nl, self.groups,self.d, self.d, device=device, dtype=dtype1)
        self.beta = torch.randn(self.nl, self.d, device=device, dtype=dtype1)
        self.gamma = torch.randn(self.nl, self.d, device=device, dtype=dtype1)

        self.omega_re = torch.randn(self.K0,self.d, device=device, dtype=dtype1)
        self.B_re = torch.randn(1,self.K0, device=device, dtype=dtype1)
        self.omega_im = torch.randn(self.K0, self.d, device=device, dtype=dtype1)
        self.B_im = torch.randn(1, self.K0, device=device, dtype=dtype1)

        '''总参数量'''
        self.parameter_size =(self.V.numel() + self.emb_matrix.numel()
                                   + self.concate.numel() + self.K.numel() + self.Q.numel()
                                   + self.W0.numel() + self.beta.numel()
                                   + self.gamma.numel() ) + 2*(self.omega_im.numel() + self.B_im.numel())
        self.step = 500
        self.warm_up = 100
        '''设置虚时间步'''
        self.tau = 0.05
        self.ReLU = torch.nn.ReLU()
        self.Y = 0

    def mapping(self, state):
        mapping_state = torch.zeros((state.shape[0], 7, self.lx, self.ly),dtype=dtype1,device=device)
        mapping_state[:, 0, :, :] = state
        mapping_state[:, 1, :, :] = torch.roll(state, dims=2, shifts=-1)
        mapping_state[:, 2, :, :] = torch.roll(state, dims=1, shifts=1)
        mapping_state[:, 3, :, :] = torch.roll(state, dims=(1, 2), shifts=(1,1))
        mapping_state[:, 4, :, :] = torch.roll(state, dims=2, shifts=1)
        mapping_state[:, 5, :, :] = torch.roll(state, dims=1, shifts=-1)
        mapping_state[:, 6, :, :] = torch.roll(state, dims=(1, 2), shifts=(-1,-1))
        return mapping_state.permute(0,2,3,1)

    def ViT(self, states):
        batch_size = states.shape[0]
        X = self.mapping(states)[:, :, :, D6].permute(0, 3, 1, 2, 4).reshape(batch_size * self.groups, self.N, self.b)
        '''X[batch_size,N,b=7]*emb_matrix[b=7,d]->y[batch_size,N,d]'''
        y = (torch.tensordot(X, self.emb_matrix, dims=([2], [0]))).reshape(batch_size * self.groups, self.N, self.d)
        '''计算注意力矩阵 Attn'''
        '''X[batch_size,N,b]Q[h,b,d]->Q[batch_size,h,N,d]'''
        '''X[batch_size,N,b]K0[h,b,d]->K0[batch_size,h,d,N]'''
        Q = torch.tensordot(X, self.Q, dims=([2], [1])).permute(0, 2, 1, 3)
        K = torch.tensordot(X, self.K, dims=([2], [1])).permute(0, 2, 3, 1)
        '''Q[batch_size,h,N,d]K0[batch_size,h,d,N]Attn[batch_size,h,N,N]'''
        Attn = torch.softmax(torch.matmul(Q, K / math.sqrt(self.d)), dim=-1)
        for i in range(self.nl):
            '''y[batch_size,N,d]*self.V[h,d,r]=V[batch_size,h,N,r]'''
            V = torch.tensordot(y, self.V[i, :, :, :], dims=([2], [1])).permute(0, 2, 1, 3)
            '''A[batch_size,N,h,r]=alpha_im[batch_size,h,N,N]*V[batch_size,h,N,r]'''
            A = torch.matmul(Attn, V).permute(0, 2, 1, 3)
            '''A[batch_size,N,h,r],concate[h,r,d]->A[batch_size,N,d=h*r]'''
            A = torch.tensordot(A, self.concate[i, :, :, :], dims=([2, 3], [0, 1]))
            '''加入残差和ReLU,A[batch_size,N,d=h*r]'''
            A = self.ReLU(A) + y
            '''层归一化'''
            mu = torch.mean(A, dim=-1, keepdim=True)
            sigma = torch.var(A, dim=-1, keepdim=True)
            A = self.gamma[i, :] * ((A - mu) / torch.sqrt(sigma + 1e-8)) + self.beta[i, :]
            '''y[batch_size,N,d=h*r],加入残差和ReLU'''
            '''self.W0[d,d]'''
            A = A.reshape(batch_size,self.groups,self.N,self.d)
            '''A[batch_size,groups,N,d]*W0[groups,d,d]=A[batch_size,groups,N,groups,d]'''
            y = self.ReLU(torch.tensordot(A, self.W0[i,:, :, :], dims=([3], [1]))).permute(0,1,3,2,4)
            y = torch.mean(y,dim=1).reshape(batch_size*self.groups,self.N,self.d)
            y = self.ReLU(y)+A.reshape(batch_size*self.groups,self.N,self.d)
        y = y.reshape(batch_size, self.groups, self.N, self.d)
        y = torch.mean(y, dim=2)
        '''y[batch_size,groups,d]*omega[K0,d]->y[batch_size,groups,K0]'''
        y = torch.tensordot(y.to(dtype2), (self.omega_re + 1j * self.omega_im), dims=([2], [1])) / self.K0 + (
                self.B_re + 1j * self.B_im)
        '''ln_psi[batch_size,groups,K0]->ln_psi[batch_size,groups]'''
        ln_psi = torch.sum(torch.log(torch.cosh(y)), dim=2)
        ln_psi = torch.logsumexp(ln_psi, dim=1)

        return ln_psi

    def select_sample(self, state):
        with torch.no_grad():
            """我们得到抽样之前的样本ln_psi_old"""
            ln_psi_old = self.ViT(state)
            '''对于二维的模型,我们依然使用U(1)对称性,我们从mz=0的态出发,每次翻转两个'''
            '''我们得到一批第一个格点要翻转的坐标'''
            Rx_1 = torch.randint(0, self.lx, (self.step,), device=device)
            Ry_1 = torch.randint(0, self.ly, (self.step,), device=device)
            '''我们得到一批第二个格点要翻转的坐标'''
            Rx_2 = torch.randint(0, self.lx, (self.step,), device=device)
            Ry_2 = torch.randint(0, self.ly, (self.step,), device=device)
            '''为了满足U(1)对称性我们要选择两个相反自旋的格点'''
            spin1 = state[torch.arange(self.step, device=device), Rx_1, Ry_1]
            spin2 = state[torch.arange(self.step, device=device), Rx_2, Ry_2]
            # 获取需要更新的样本索引
            update_indices = torch.where(spin1 * spin2 == -1)[0]
            num_updates = len(update_indices)

            # 提取需要更新的状态
            states_to_update = state[update_indices]  # [num_updates, L]
            Rx_1_updates = Rx_1[update_indices]
            Ry_1_updates = Ry_1[update_indices]
            Rx_2_updates = Rx_2[update_indices]
            Ry_2_updates = Ry_2[update_indices]

            # 创建新状态并翻转
            new_states_updates = states_to_update.clone()
            batch_idx = torch.arange(num_updates, device=device)
            new_states_updates[batch_idx, Rx_1_updates, Ry_1_updates] = -new_states_updates[
                batch_idx, Rx_1_updates, Ry_1_updates]
            new_states_updates[batch_idx, Rx_2_updates, Ry_2_updates] = -new_states_updates[
                batch_idx, Rx_2_updates, Ry_2_updates]
            # 计算新状态的概率
            ln_psi_new_updates = self.ViT(new_states_updates)
            ln_psi_old_updates = ln_psi_old[update_indices]

            # Metropolis准则
            prob_ratio = torch.abs(torch.exp(ln_psi_new_updates)) ** 2 / torch.abs(torch.exp(ln_psi_old_updates)) ** 2
            acceptance = torch.clamp(prob_ratio, max=1.0)
            rand_nums = torch.rand(num_updates, device=device)
            accept_mask_updates = rand_nums < acceptance

            # 更新接受的状态
            final_states = state.clone()
            final_lnP = ln_psi_old.clone()

            # 只有被接受的状态才更新
            accepted_indices = update_indices[accept_mask_updates]
            if len(accepted_indices) > 0:
                final_states[accepted_indices] = new_states_updates[accept_mask_updates]
                final_lnP[accepted_indices] = ln_psi_new_updates[accept_mask_updates]

        return final_lnP, final_states

    '''局域能量的计算'''
    '''我们局域能量的计算:分为对角的Ising型和非对角的量子涨落型'''

    def E_loc(self, states, ln_psi):
        with torch.no_grad():
            E = torch.zeros(self.step, dtype=dtype2, device=device)
            batch_size = states.shape[0]
            """最近邻相互作用"""
            '''加上非对角项的贡献'''
            '''将自旋反号'''
            # 创建行索引数组
            states_new = torch.roll(states, shifts=1, dims=1)
            E += torch.sum(states * states_new, dim=(1, 2)).reshape(batch_size) / 4
            diff = torch.stack(torch.where(states != states_new))
            states_new = states[diff[0], :, :]
            rows = torch.arange(states_new.shape[0])
            states_new[rows, diff[1], diff[2]] = -states_new[rows, diff[1], diff[2]]
            states_new[rows, diff[1] - 1, diff[2]] = - states_new[rows, diff[1] - 1, diff[2]]
            ln_psi_new = self.ViT(states_new)
            values = torch.exp(ln_psi_new - ln_psi[diff[0]]) * 0.5
            E.index_add_(0, diff[0], values)

            states_new = torch.roll(states, shifts=1, dims=2)
            E += torch.sum(states * states_new, dim=(1, 2)).reshape(batch_size) / 4
            diff = torch.stack(torch.where(states != states_new))
            states_new = states[diff[0], :, :]
            rows = torch.arange(states_new.shape[0])
            states_new[rows, diff[1], diff[2]] = -states_new[rows, diff[1], diff[2]]
            states_new[rows, diff[1], diff[2] - 1] = - states_new[rows, diff[1], diff[2] - 1]
            ln_psi_new = self.ViT(states_new)
            values = torch.exp(ln_psi_new - ln_psi[diff[0]]) * 0.5
            E.index_add_(0, diff[0], values)

            states_new = torch.roll(states, shifts=(1, 1), dims=(1, 2))
            E += torch.sum(states * states_new, dim=(1, 2)).reshape(batch_size) / 4
            diff = torch.stack(torch.where(states != states_new))
            states_new = states[diff[0], :, :]
            rows = torch.arange(states_new.shape[0])
            states_new[rows, diff[1], diff[2]] = -states_new[rows, diff[1], diff[2]]
            states_new[rows, diff[1] - 1, diff[2] - 1] = - states_new[rows, diff[1] - 1, diff[2] - 1]
            ln_psi_new = self.ViT(states_new)
            values = torch.exp(ln_psi_new - ln_psi[diff[0]]) * 0.5
            E.index_add_(0, diff[0], values)
        torch.cuda.empty_cache()
        return E

    def mag_z(self, states, i, j):
        """用于实现计算[i,j]格点的磁矩Sz的期望值"""
        return states[:, i, j]

    def learning(self):
        print(self.parameter_size)
        # 我们的初始状态从Neel态出发
        states = Neel_state(self.lx, self.ly)
        '''我们复制Neel态,得到一批Neel态'''
        states = states.repeat(self.step, 1, 1)

        """初始化参数的梯度,使其梯度为True"""
        self.omega_re.requires_grad_(True)
        self.B_re.requires_grad_(True)

        self.V.requires_grad_(True)
        self.Q.requires_grad_(True)
        self.K.requires_grad_(True)
        self.emb_matrix.requires_grad_(True)
        self.concate.requires_grad_(True)
        self.omega_im.requires_grad_(True)
        self.B_im.requires_grad_(True)
        self.W0.requires_grad_(True)
        self.beta.requires_grad_(True)
        self.gamma.requires_grad_(True)

        parameter = torch.zeros(self.parameter_size, device=device, dtype=dtype1)
        nu = torch.ones(self.parameter_size, device=device, dtype=dtype1)
        for i in range(10000):
            self.Y = i
            '''热化'''
            for _ in range(100):
                ln_psi, states = self.select_sample(states)
            # 抽样
            ln_psi, states = self.select_sample(states)
            params_list = [
                self.omega_re, self.B_re,
                self.V, self.Q, self.K, self.emb_matrix, self.W0, self.omega_im, self.B_im,
                self.gamma, self.beta, self.concate
            ]
            O_re = torch.zeros((self.step, self.parameter_size), device=device, dtype=dtype1)
            O_im = torch.zeros((self.step, self.parameter_size), device=device, dtype=dtype1)
            chunk_size = self.step // 4

            for u in range(4):
                start = u * chunk_size
                end = (u + 1) * chunk_size if u < 3 else self.step
                states_batch = states[start:end, :, :]

                parameter_flatten = torch.cat([p.flatten() for p in params_list]).detach()

                def f_re(p):
                    idx = 0
                    n = 0
                    self.omega_re = p[idx:idx + (n := self.omega_re.numel())].view(self.omega_re.shape)
                    idx += n
                    self.B_re = p[idx:idx + (n := self.B_re.numel())].view(self.B_re.shape)
                    idx += n

                    self.V = p[idx:idx + (n := self.V.numel())].view(self.V.shape)
                    idx += n
                    self.Q = p[idx:idx + (n := self.Q.numel())].view(self.Q.shape)
                    idx += n
                    self.K = p[idx:idx + (n := self.K.numel())].view(self.K.shape)
                    idx += n
                    self.emb_matrix = p[idx:idx + (n := self.emb_matrix.numel())].view(self.emb_matrix.shape)
                    idx += n
                    self.W0 = p[idx:idx + (n := self.W0.numel())].view(self.W0.shape)
                    idx += n
                    self.omega_im = p[idx:idx + (n := self.omega_im.numel())].view(self.omega_im.shape)
                    idx += n
                    self.B_im = p[idx:idx + (n := self.B_im.numel())].view(self.B_im.shape)
                    idx += n
                    self.gamma = p[idx:idx + (n := self.gamma.numel())].view(self.gamma.shape)
                    idx += n
                    self.beta = p[idx:idx + (n := self.beta.numel())].view(self.beta.shape)
                    idx += n
                    self.concate = p[idx:idx + (n := self.concate.numel())].view(self.concate.shape)
                    idx += n


                    return self.ViT(states_batch).real

                O_batch_re = jacobian(f_re, parameter_flatten, vectorize=True)
                O_re[start:end, :] = O_batch_re  # O_batch 形状 [actual_batch_size, parameter_size]
                parameter_flatten = torch.cat([p.flatten() for p in params_list]).detach()

                def f_im(p):
                    idx = 0
                    n = 0
                    self.omega_re = p[idx:idx + (n := self.omega_re.numel())].view(self.omega_re.shape)
                    idx += n
                    self.B_re = p[idx:idx + (n := self.B_re.numel())].view(self.B_re.shape)
                    idx += n

                    self.V = p[idx:idx + (n := self.V.numel())].view(self.V.shape)
                    idx += n
                    self.Q = p[idx:idx + (n := self.Q.numel())].view(self.Q.shape)
                    idx += n
                    self.K = p[idx:idx + (n := self.K.numel())].view(self.K.shape)
                    idx += n
                    self.emb_matrix = p[idx:idx + (n := self.emb_matrix.numel())].view(self.emb_matrix.shape)
                    idx += n
                    self.W0 = p[idx:idx + (n := self.W0.numel())].view(self.W0.shape)
                    idx += n
                    self.omega_im = p[idx:idx + (n := self.omega_im.numel())].view(self.omega_im.shape)
                    idx += n
                    self.B_im = p[idx:idx + (n := self.B_im.numel())].view(self.B_im.shape)
                    idx += n
                    self.gamma = p[idx:idx + (n := self.gamma.numel())].view(self.gamma.shape)
                    idx += n
                    self.beta = p[idx:idx + (n := self.beta.numel())].view(self.beta.shape)
                    idx += n
                    self.concate = p[idx:idx + (n := self.concate.numel())].view(self.concate.shape)
                    idx += n


                    return self.ViT(states_batch).imag

                O_batch_im = jacobian(f_im, parameter_flatten, vectorize=True)
                O_im[start:end, :] = O_batch_im
            O = torch.cat([O_re, O_im], dim=0)
            O_mean = torch.mean(O, dim=0)
            O = (O - O_mean) / math.sqrt(2*self.step)
            E = self.E_loc(states, ln_psi)
            E_mean = torch.mean(E)
            print('*************************************************')
            print('第', i, '步')
            print('能量为', E_mean.item() / self.lx / self.ly)
            E = -self.tau * (E - E_mean)
            E = torch.cat([E.real, E.imag], dim=0)
            L = torch.linalg.cholesky(torch.mm(O.to(torch.float64),
                                               torch.mm(torch.sqrt(torch.diag((1 / nu).to(torch.float64))),
                                                        O.to(torch.float64).T)) + self.lam * torch.eye(2 * self.step,
                                                                                                       dtype=torch.float64,
                                                                                                       device=device)).to(
                dtype1)
            parameter_k = torch.mv(torch.sqrt(torch.diag(1 / nu)), torch.mv(O.T, torch.cholesky_solve(
                (E - torch.mv(O, self.mu * parameter.detach())).reshape(2 * self.step, 1), L).reshape(
                2 * self.step))) + self.mu * parameter.detach()
            nu = self.beta0 * nu + (parameter_k.detach() - parameter.detach()) ** 2
            parameter = parameter_k.detach()
            parameter_k *= 0.01 / (1 + max(i - 8000, 0) / 8000)
            with torch.no_grad():
                idx = 0

                omega_re_size = self.omega_re.numel()
                self.omega_re += parameter_k[idx:idx + omega_re_size].reshape(self.omega_re.shape)
                idx += omega_re_size

                B_re_size = self.B_re.numel()
                self.B_re += parameter_k[idx:idx + B_re_size].reshape(self.B_re.shape)
                idx += B_re_size

                V_im_size = self.V.numel()
                self.V += parameter_k[idx:idx + V_im_size].reshape(self.V.shape)
                idx += V_im_size

                Q_im_size = self.Q.numel()
                self.Q += parameter_k[idx:idx + Q_im_size].reshape(self.Q.shape)
                idx += Q_im_size

                K_im_size = self.K.numel()
                self.K += parameter_k[idx:idx + K_im_size].reshape(self.K.shape)
                idx += K_im_size

                emb_matrix_im_size = self.emb_matrix.numel()
                self.emb_matrix += parameter_k[idx:idx + emb_matrix_im_size].reshape(self.emb_matrix.shape)
                idx += emb_matrix_im_size

                W0_im_size = self.W0.numel()
                self.W0 += parameter_k[idx:idx + W0_im_size].reshape(self.W0.shape)
                idx += W0_im_size

                omega_im_size = self.omega_im.numel()
                self.omega_im += parameter_k[idx:idx + omega_im_size].reshape(self.omega_im.shape)
                idx += omega_im_size

                B_im_size = self.B_im.numel()
                self.B_im += parameter_k[idx:idx + B_im_size].reshape(self.B_im.shape)
                idx += B_im_size

                gamma_im_size = self.gamma.numel()
                self.gamma += parameter_k[idx:idx + gamma_im_size].reshape(self.gamma.shape)
                idx += gamma_im_size

                beta_im_size = self.beta.numel()
                self.beta += parameter_k[idx:idx + beta_im_size].reshape(self.beta.shape)
                idx += beta_im_size

                concate_im_size = self.concate.numel()
                self.concate += parameter_k[idx:idx + concate_im_size].reshape(self.concate.shape)
                idx += concate_im_size

            torch.cuda.empty_cache()


Lx = 4
Ly = 4
Triangle = Triangle(lx=Lx, ly=Ly)
Triangle.learning()
'''batch_size = 1
states = torch.arange(Lx*Ly).reshape(batch_size,Lx,Ly)
print(states)
states_mapping = Kagome.mapping(states)
print(states_mapping)
states_D6 = states[:,:,:,D6]
print(states_D6.sjape)'''
