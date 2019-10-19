class BCG():
    
    def __init__(self, A, b, prior_mean, prior_cov, eps, m_max):
        
        self.A = A
        self.b = b
        self.x0 = prior_mean
        self.sigma0 = prior_cov
        self.eps = eps
        self.max = m_max

    def bcg(self):
        
        sigmaF = [] #np.concatenate?
        
        A = self.A
        b = self.b
        x0 = self.x0
        sigma0 = self.sigma0
        eps = self.eps
        m_max = self.max
        
        r_m = b - torch.mm(A, x0)
        r_m_dot_r_m = torch.mm(r_m.t(), r_m)
        s_m = r_m
        x_m = x0
        
        nu_m = 0
        m = 0
        d = b.shape[0]
        
        while True:
            
            sigma_At_s = torch.mm(sigma0, torch.mm(A.t(), s_m))
            A_sigma_A_s = torch.mm(A, sigma_At_s)
            
            E_2 = torch.mm(s_m.t(), A_sigma_A_s)
            alpha_m = r_m_dot_r_m / E_2
            x_m += alpha_m * sigma_At_s
            r_m -= alpha_m * A_sigma_A_s
            nu_m += alpha_m**2 #r_m_dot_r_m * r_m_dot_r_m / E_2
            sigma_m = ((d - 1 - m) * nu_m / (m + 1)).sqrt() ##??
            prev_r_m_dot_r_m = r_m_dot_r_m
            r_m_dot_r_m = torch.mm(r_m.t(), r_m)
            E = E_2.sqrt()
            sigmaF.append(sigma_At_s / E)
            
            m +=1
            
            beta_m = r_m_dot_r_m / prev_r_m_dot_r_m
            s_m = r_m + beta_m *s_m
            
            #add minimal no of iterations
            if sigma_m < eps:
                break
            '''else sqrt(r_m_dot_r_m) < eps: - traditional residual-minimising strategy
                break'''
            if m == m_max or m == d:
                raise
                
            return self.BCGOutput(sigmaF, m, x_m, nu_m/m)
        
    def BCGOutput(self, sigmaF, m, x_m, nu_m):
        
        pass #TODO
