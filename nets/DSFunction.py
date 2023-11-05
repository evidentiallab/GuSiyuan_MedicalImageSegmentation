import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class DsFunction1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, W, BETA, alpha, gamma):
        class_dim = 2
        prototype_dim = 10
        [batch_size, in_channel, height, weight, depth] = input.size()

        BETA2 = BETA * BETA
        beta2 = BETA2.t().sum(0)
        U = BETA2 / (beta2.unsqueeze(1) * torch.ones(1, class_dim, device=input.device))
        alphap = 0.99 / (1 + torch.exp(-alpha))  # sigmoid

        d = torch.zeros(prototype_dim, batch_size, height, weight, depth, device=input.device)
        s = torch.zeros(prototype_dim, batch_size, height, weight, depth, device=input.device)
        expo = torch.zeros(prototype_dim, batch_size, height, weight, depth, device=input.device)

        mk = torch.cat((torch.zeros(class_dim, batch_size, height, weight, depth, device=input.device), torch.ones(1, batch_size, height, weight, depth, device=input.device)), 0)

        for k in range(prototype_dim):
            temp = input.permute(1, 0, 2, 3, 4) - torch.mm(W[k, :].unsqueeze(1), torch.ones(1, batch_size, device=input.device)).unsqueeze(2).unsqueeze(3).unsqueeze(4)
            d[k, :] = 0.5 * (temp * temp).sum(0)
            expo[k, :] = torch.exp(-gamma[k] ** 2 * d[k, :])
            s[k, :] = alphap[k] * expo[k, :]
            m = torch.cat((U[k, :].unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4) * s[k, :], torch.ones(1, batch_size, height, weight, depth, device=input.device) - s[k, :]), 0)

            t2 = mk[:class_dim, :, :, :, :] * (m[:class_dim, :, :, :, :] + torch.ones(class_dim, 1, height, weight, depth, device=input.device)* m[class_dim, :, :, :, :])
            t3 = m[:class_dim, :, :, :, :] * (torch.ones(class_dim, 1, height, weight, depth, device=input.device) * mk[class_dim, :, :, :, :])
            t4 = (mk[class_dim, :, :, :, :]) * (m[class_dim, :, :, :, :]).unsqueeze(0)
            mk = torch.cat((t2 + t3, t4), 0)

        K = mk.sum(0)
        mk_n = (mk / (torch.ones(class_dim + 1, 1, height, weight, depth, device=input.device) * K)).permute(1, 0, 2, 3, 4)
        #mass_b = mk_n[:, :class_dim,:,:,:] + 1 / class_dim * mk_n[:, class_dim:,:,:,:] * torch.ones(1, class_dim,height,weight,depth,device=input.device)
        ctx.save_for_backward(input, W, BETA, alpha, gamma, mk, d)
        return mk_n

    @staticmethod
    def backward(ctx, grad_output):

        input, W, BETA, alpha, gamma, mk, d = ctx.saved_tensors
        grad_input = grad_W = grad_BETA = grad_alpha = grad_gamma = None

        M = 2
        prototype_dim = 10
        [batch_size, in_channel, height, weight, depth] = input.size()
        mu = 0  # regularization parameter (default=0)
        iw = 1  # 1 if optimization of prototype centers, 0 otherwise (default=1)
        grad_output_ = grad_output[:, :2, :, :, :]*batch_size*M*height*weight*depth
        K = mk.sum(0).unsqueeze(0)
        K2 = K ** 2
        BETA2 = BETA * BETA
        beta2 = BETA2.t().sum(0).unsqueeze(1)
        U = BETA2 / (beta2 * torch.ones(1, M, device=input.device))
        alphap = 0.99 / (1 + torch.exp(-alpha))  # 200*1
        I = torch.eye(M, device=grad_output.device)

        s = torch.zeros(prototype_dim, batch_size, height, weight, depth, device=input.device)
        expo = torch.zeros(prototype_dim, batch_size, height, weight, depth, device=input.device)
        mm = torch.cat((torch.zeros(M, batch_size, height, weight, depth, device=input.device), torch.ones(1, batch_size, height, weight, depth, device=input.device)), 0)

        dEdm = torch.zeros(M + 1, batch_size, height, weight, depth, device=input.device)
        # dEdm_test = torch.zeros(M + 1, batch_size, height, weight, depth,device=input.device)
        dU = torch.zeros(prototype_dim, M, device=input.device)
        Ds = torch.zeros(prototype_dim, batch_size, height, weight, depth, device=input.device)
        DW = torch.zeros(prototype_dim, in_channel, device=input.device)

        for p in range(M):

            dEdm[p, :] = (grad_output_.permute(1, 0, 2, 3, 4) * (
                        I[:, p].unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4) * K - mk[:M, :] - 1 / M * (
                            torch.ones(M, 1, height, weight, depth, device=input.device) * mk[M, :]))).sum(0) / K2

        dEdm[M, :] = ((grad_output_.permute(1, 0, 2, 3, 4) * (
                    - mk[:M, :] + 1 / M * torch.ones(M, 1, height, weight, depth, device=input.device) * (K - mk[M, :]))).sum(
            0)) / K2
        # dEdm_t[M, :] = ((grad_output_t.t() * (- mk_t[:M, :] + 1 / 3 * torch.ones(3, 1, device=input.device) * (K_t - mk_t[M, :]))).sum(0)) / K2_t

        for k in range(prototype_dim):
            expo[k, :] = torch.exp(-gamma[k] ** 2 * d[k, :])
            s[k] = alphap[k] * expo[k, :]
            m = torch.cat((U[k, :].unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4) * s[k, :],torch.ones(1, batch_size, height, weight,depth, device=input.device) - s[k, :]), 0)
            mm[M, :] = mk[M, :] / m[M, :]
            L = torch.ones(M, 1, height, weight, depth, device=input.device) * mm[M, :]    # L:m_M+1
            mm[:M, :] = (mk[:M, :] - L * m[:M, :]) / (m[:M, :] + torch.ones(M, 1, height, weight, depth, device=input.device) * m[M, :])  # m_j
            R = mm[:M, :] + L     # function 97,
            A = R * torch.ones(M, 1, height, weight, depth, device=input.device) * s[k, :]  # function 97, s
            B = U[k, :].unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4) * torch.ones(1, batch_size, height, weight, depth, device=input.device) * R - mm[:M, :]
            # tet=torch.mean((A * dEdm[:M, :]).view(3,-1).permute(1,0),0)
            dU[k, :] = torch.mean((A * dEdm[:M, :]).view(M,-1).permute(1,0),0)
            Ds[k, :] = (dEdm[:M, :] * B).sum(0) - (dEdm[M, :] * mm[M, :])
            # test=Ds[k, :] * gamma[k] ** 2 * s[k, :]
            tt1 = Ds[k, :] * (gamma[k] ** 2 * torch.ones(1, batch_size, height, weight, depth, device=input.device)) * s[k, :]
            tt2 = (torch.ones(batch_size, 1, device=input.device) * W[k, :]).unsqueeze(2).unsqueeze(
                3).unsqueeze(4) - input  # - input
            tt1 = tt1.view(1, -1)
            tt2 = tt2.permute(1, 0, 2, 3, 4).reshape(in_channel, batch_size*height*weight*depth).permute(1, 0)
            DW[k, :] = -torch.mm(tt1, tt2)



        DW = iw * DW / (batch_size*height*weight*depth)
        T = beta2 * torch.ones(1, M, device=input.device)
        Dbeta = (2 * BETA / T ** 2) * (dU * (T - BETA2) - (dU * BETA2).sum(1).unsqueeze(1) * torch.ones(1, M,
                                                                                                        device=input.device) + dU * BETA2)
        Dgamma = - 2 * torch.mean(((Ds * d * s).view(prototype_dim, -1)).t(), 0).unsqueeze(1) * gamma
        Dalpha = (torch.mean(((Ds * expo).view(prototype_dim, -1)).t(), 0).unsqueeze(1) + mu) * (
                    0.99 * (1 - alphap) * alphap)
        Dinput = torch.zeros(batch_size, in_channel, height, weight, depth, device=input.device)
        temp2 = torch.zeros(prototype_dim, in_channel, height, weight, depth, device=input.device)

        for n in range(batch_size):
            for k in range(prototype_dim):
                test7 = input[n, :] - W[k, :].unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)
                test9 = (Ds[k, n, :, :] * (gamma[k] ** 2) * s[k, n, :, :]).unsqueeze(0).unsqueeze(1)
                temp2[k] = -prototype_dim*test9*test7
                Dinput[n, :] = temp2.mean(0)


        if ctx.needs_input_grad[0]:
            grad_input = Dinput
        if ctx.needs_input_grad[1]:
            grad_W = DW
        if ctx.needs_input_grad[2]:
            grad_BETA = Dbeta
        if ctx.needs_input_grad[3]:
            grad_alpha = Dalpha
        if ctx.needs_input_grad[4]:
            grad_gamma = Dgamma

        return grad_input, grad_W, grad_BETA, grad_alpha, grad_gamma


class Ds1(nn.Module):
    def __init__(self, input_dim, prototpye_dim, class_dim):
        super(Ds1, self).__init__()
        self.input_dim = input_dim
        self.class_dim = class_dim
        self.prototype_dim = prototpye_dim
        self.BETA = Parameter(torch.Tensor(self.prototype_dim, self.class_dim))
        self.alpha = Parameter(torch.Tensor(self.prototype_dim, 1))
        self.gamma = Parameter(torch.Tensor(self.prototype_dim, 1))
        self.W = Parameter(torch.Tensor(self.prototype_dim, self.input_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.W)
        nn.init.normal_(self.BETA)
        nn.init.constant_(self.gamma, 0.1)
        nn.init.constant_(self.alpha, 0)

    def forward(self, input):
        return DsFunction1.apply(input, self.W, self.BETA, self.alpha, self.gamma)
