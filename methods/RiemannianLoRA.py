import torch
from torch.optim import Optimizer
import torch.nn as nn


class FixedRank:
    def __init__(self, r: int):
        self.r = r

        """
        point = A @ B ~ (A, B) 
        A \in St(m, r), left orthogonalization
        
        vec = A_l @ dot_B + dot_A @ B_r
        A_l \in St(m, r), B_r \in St(n, r) 
        """
        
    def norm2(self, point, vec):
        dot_A, dot_B = vec
        _, B = point 
        return [torch.norm(dot_A)**2, torch.norm(dot_B)**2]

    def retraction(self, point, tangent_vec):
        A_part, B_part = self.embedding(point, tangent_vec, ambient=False)
        Q_l, R_l = torch.linalg.qr(A_part)
        Q_r, R_r = torch.linalg.qr(B_part.T)
        T = R_l @ R_r.T # (r, r)

        U, S, Vt = torch.linalg.svd(T)
        U = U[:, :self.r]
        S = S[:self.r]
        Vt = Vt[:self.r, :]
        
        new_A = Q_l @ U
        new_B = torch.diag(S) @ Vt @ Q_r.T
        return [new_A, new_B]

    def embedding(self, point: list, vec: list, ambient: bool = True) -> torch.Tensor: 
        U_l, _ = point
        _, Vt_r = self.to_right(point)

        # return (m, n) matrix
        if ambient:
            return U_l @ vec[1] + vec[0] @ Vt_r
        
        left  = torch.cat([U_l, vec[0]],  dim=1)
        right = torch.cat([vec[1], Vt_r], dim=0)
        return [left, right]

    def to_right(self, point: list):
        A, B = point 
        # A \in St(m, r)

        Q, R = torch.linalg.qr(B.T)
        B = Q.T
        A = A @ R.T

        return [A, B]

    def euclidian_to_reimanian(self, point: list, vec: list) -> list:
        A, B = point
        dot_A, dot_B = vec

        # project dot_A on orhogonal complement to span(A)
        dot_A -= A @ (A.T @ dot_A)

        return dot_A, dot_B

    def transport(self, point_X: list, point_Y: list, vec_X: list):
        U_l, _ = point_Y
        _, Vt_r = self.to_right(point_Y)

        A_part, B_part = self.embedding(point_X, vec_X, ambient=False)
        # Z = A_part @ B_part
        dot_U = (A_part - U_l @ (U_l.T @ A_part)) @ (B_part @ Vt_r.T)
        dot_V = (U_l.T @ A_part) @ B_part
        return [dot_U, dot_V]


class RiemannianLora(Optimizer):
    def __init__(self, params, **defaults):
        
        lr = defaults.get('lr', 0.001)
        betas = defaults.get('betas', [0.9, 0.999])
        super().__init__(params, {'lr' : lr, 'betas': betas})
    
    @torch.no_grad()
    def apply_second_momentum(self, point: list, grad_vec: list, momentum_vec: list, state: dict, beta: list, manifold:FixedRank):
        # bias_factor1 = (1 - beta[0] ** (state['step'] + 1))
        # bias_factor2 = (1 - beta[1] ** (state['step'] + 1))
        
        norm = manifold.norm2(point=point, vec=grad_vec)
        
        if 'second_momentum' not in state:
            state['second_momentum'] = [norm[0] * (1-beta[1]), norm[1] * (1-beta[1])]
            
            addm = [1 / (torch.sqrt(state['second_momentum'][0]) + 1e-8), 1 / (torch.sqrt(state['second_momentum'][1]) + 1e-8)]
            dot_A, dot_B = momentum_vec[0] * addm[0], momentum_vec[1] * addm[1]
            
            return dot_A, dot_B
        
        state['second_momentum'] = [beta[1] * state['second_momentum'][0] + (1-beta[1]) * norm[0], beta[1] * state['second_momentum'][1] + (1-beta[1]) * norm[1]]

        addm = [1 / (torch.sqrt(state['second_momentum'][0]) + 1e-8), 1 / (torch.sqrt(state['second_momentum'][1]) + 1e-8)]
        dot_A, dot_B = momentum_vec[0] * addm[0], momentum_vec[1] * addm[1]
        
        return dot_A, dot_B 
    
    @torch.no_grad()
    def get_deltas(self, A, B, manifold: FixedRank) -> tuple[list, list]:
        # A = (A_l, 0)
        # B = (B, B_r)
        
        dot_A = A.grad.T  # (m, 2*r)
        dot_B = B.grad.T  # (2*r, n)

        r = manifold.r
        dot_A = dot_A[:, r:]
        dot_B = dot_B[:r, :]
        
        A = A.T[:, :r]
        B = B.T[:r, :]
        point = [A, B]
        
        vec = manifold.euclidian_to_reimanian(
            point=point,
            vec=[dot_A, dot_B]
        )

        return point, vec

    @torch.no_grad()
    def apply_momentum(self, point: list, vec: list, state: dict, manifold: FixedRank, beta: float) -> list:
        
        if 'momentum' not in state:
            state['momentum'] = vec[0] * (1-beta), vec[1] * (1-beta)
            state['prev_point'] = point
            
            return vec
        
        m_vec = manifold.transport(
            point_X=state['prev_point'], # source tangent space
            point_Y=point,               # destination tangent space
            vec_X=state['momentum']      # momentum vector from source tangent space
        )
        dot_A = beta * m_vec[0] + (1 - beta) * vec[0]
        dot_B = beta * m_vec[1] + (1 - beta) * vec[1]

        state['prev_point'] = point
        state['momentum'] = [dot_A, dot_B]

        return dot_A, dot_B

    @torch.no_grad()
    def step(self, closure=None):
        for group_idx, group in enumerate(self.param_groups):
            A_idx = group['param_names'].index('lora_A.weight')
            B_idx = group['param_names'].index('lora_B.weight')
            
            A = group['params'][A_idx]  ## is orhogonal
            B = group['params'][B_idx]
            r = A.shape[0] // 2

            lr = group['lr']
            beta = group['betas']
            
            state = self.state[A]
            if len(state) == 0:
                state['step'] = 0
                state['manifold'] = FixedRank(r=r)

            manifold = state['manifold']
            point, grad_vec = self.get_deltas(A, B, manifold=manifold)

            momentum_vec = self.apply_momentum(
                point=point, vec=grad_vec,
                state=state,
                manifold=manifold,
                beta=beta[0],
            )
            
            dot_A, dot_B = self.apply_second_momentum(
                grad_vec=grad_vec,
                momentum_vec=momentum_vec,
                state=state,
                beta=beta,
                manifold=manifold
            )

            new_A, new_B = manifold.retraction(
                point=point,
                tangent_vec=[-lr * dot_A, point[1] - lr * dot_B]
            )

            A_zero = torch.zeros_like(new_A)
            B_orth = torch.linalg.qr(new_B.T).Q.T

            new_A = torch.cat([new_A, A_zero], dim=1)
            new_B = torch.cat([new_B, B_orth], dim=0)

            group['params'][A_idx].copy_(new_A.T)
            group['params'][B_idx].copy_(new_B.T)
            state['step'] += 1


class RiemannianSGD(RiemannianLora):
    def __init__(self, params, **defaults):
        
        lr = defaults.get('lr', 0.001)
        momentum = defaults.get('momentum', 0.0)

        betas = [momentum, 0.0]
        super().__init__(params, **{'lr' : lr, 'betas': betas})

    @torch.no_grad()
    def apply_second_momentum(self, point: list, grad_vec: list, momentum_vec: list, state: dict, beta: list, manifold: FixedRank):
        # does nothing

        return momentum_vec



