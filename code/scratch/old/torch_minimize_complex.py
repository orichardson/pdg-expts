def _torch_opt_inc(self, gamma=None,    
            extraTemp = 1E-3, iters=350, 
            ret_losses:bool = True,
            ret_iterates:bool = False
            #, tol = 1E-8, max_iters=300 #unsupported
            ):
        """ = min_\mu inc(\mu, gamma) """
        
        if gamma is None: # This is the target gamma
            gamma = self.gamma_default
        γ = gamma + extraTemp        
        
        # uniform starting position
        # μdata = torch.tensor(self.genΔ(RJD.unif).data, requires_grad=True)
        μdata = torch.tensor(self.genΔ(RJD.unif).data, requires_grad=True)
        μ = RJD(μdata, self.varlist, use_torch=True)
        # print(μdata)
        ozr = torch.optim.Adam([μdata], lr=2E-3)
        # ozr = torch.optim.SGD([μdata], lr=1E-3, momentum=0.8, dampening=0.0, nesterov=True)
        
        
        best_state = ozr.state_dict()
        best_μdata = μdata.detach().clone()
        went_up = False
        cooldown = 0
        def custom_lr(_epoch):
            return 0.9999**_epoch
            # relative = 0.1 if went_up else 1.
            # if epoch > iters/4: relative *= 0.999
            # if epoch > 2*iters/3: relative *= 0.99
            # 
            # 
            # if not went_up and epoch % (iters//3) == 0:
            #     relative *= 10
            # 
            # fully_discounted = relative * custom_lr.previous
            # custom_lr.previous = fully_discounted
            # return fully_discounted
        custom_lr.previous = 1
        
        lrsched1 = torch.optim.lr_scheduler.LambdaLR(ozr, custom_lr)
        # lrsched2 = torch.optim.lr_scheduler.ReduceLROnPlateau(ozr, factor=0.1)
        
        bestl = float('inf')
        losses = [ float('inf') ] 
        if ret_iterates: iterates = [ μdata.detach() ]
        
        print()# blank line
        
        for it in range(iters):
            ozr.zero_grad()
            # print(μ.data)     
            temp = 1E-3
            # temp = lrsched1.get_last_lr()[0] * 2
            nnμdata = temp*torch.logsumexp(torch.stack([μdata/temp, torch.zeros(μdata.shape)], dim=μdata.ndim), dim=-1) #soft max for +
            # nnμdata = torch.clip(μdata, min=0) # hard max for positivity

            # print(μdata)
            μ.data = ( nnμdata / nnμdata.sum())
            loss = self.torch_score(μ, γ)            
            l = loss.detach().item()
            went_up = l > losses[-1]            
    
            if True or went_up: # Nice printing.
                numdig = str(len(str(iters)))
                # sys.stdout.write(
                #     ('[{ep:>'+numdig+'}/{its}]  loss:  {ls:.3e};  lr: {lr:.3e}')\
                #         .format(ep=it, ls=loss.detach().item(), lr=lrsched1.get_last_lr()[0], its=iters) )
                # sys.stdout.flush()    
                # if True: sys.stdout.write('\n')
                print(
                    ('[{ep:>'+numdig+'}/{its}]  loss:  {ls:.3e};  lr: {lr:.3e}')\
                        .format(ep=it, ls=loss.detach().item(), lr=lrsched1.get_last_lr()[0], its=iters) )
                    # sys.stdout.flush()
            
            
            # We know this is strictly convex, so if we went up, go back 
                # and make a new optimizer with a smaller learning rate.
            if went_up:
                if cooldown == 0:
                    print('\n', "*"*65)
                    print('|\t current loss: {}, last loss: {}'.format(l, losses[-1]))
                    # print('|\t reverting to state_dict: ', best_state, ' from ', ozr.state_dict())
                    μdata = best_μdata.clone()
                    μdata.requires_grad = True    
                    # rebuild optimizer...
                    ozr = torch.optim.Adam([μdata], lr=lrsched1.get_last_lr()[0]/1000)
                    # ozr.load_state_dict(best_state)
                    lrsched1 = torch.optim.lr_scheduler.LambdaLR(ozr, custom_lr)
                    cooldown = 30
                    continue
                else: cooldown -= 1
            
            elif l <= bestl:
                best_μdata = μdata.detach().clone()
                best_state = ozr.state_dict()
                bestl = l
            
            
            ## The breadcrumbs we leave in case we get lost + a map of where we've been
            if ret_losses: losses.append(l)
            else: losses = [l]
            
            if ret_iterates: iterates.append(μdata.detach().clone())
            else: iterates = [μdata.detach().clone()]


            # if : # update the optimizer unless we just reset it
            loss.backward()
            ozr.step()
            # if it % 10 == 0:
            lrsched1.step()
            # lrsched2.step(loss)
            
            γ += (gamma-γ) / 3. # anneal to closest
        
        μ.data = best_μdata
        
        to_ret = ()
        if ret_iterates: to_ret += (iterates,)
        if ret_losses: to_ret += (losses,)
        return (μ,)+to_ret if len(to_ret) else μ
