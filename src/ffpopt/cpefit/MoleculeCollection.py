#!/usr/bin/env python3


class ParamListType(object):
    
    def __init__(self,npar):
        import numpy as np
        self.q =  np.zeros( (npar,) )
        self.chempot =  np.zeros( (npar,) )
        self.hardness =  np.zeros( (npar,) )
        self.zetascl =  1.
        self.opt_q = True
        self.opt_chempot = True
        self.opt_hardness = True
        self.opt_zetascl = True
        self.resp_a = 0.001
        self.resp_b = 0.1

        #self.lb_q = np.array([-np.inf]*npar)
        #self.ub_q = np.array([ np.inf]*npar)
        self.lb_chempot = np.array([-2]*npar)
        self.ub_chempot = np.array([0]*npar)
        self.lb_hardness = np.array([0.1]*npar)
        self.ub_hardness = np.array([50.]*npar)
        self.lb_zetascl = 0.1
        self.ub_zetascl = 1.0

    # off = 0
    # if optq:
    #     lb[off:off+nfree] = -np.inf
    #     ub[off:off+nfree] =  np.inf
    #     off += nfree
    # if optchempot:
    #     lb[off:off+npar] = -2.
    #     ub[off:off+npar] =  0.
    #     off += npar
    # if opthardness:
    #     lb[off:off+npar] =  0.1
    #     ub[off:off+npar] = 50.00
    #     xfree[off:off+npar] = mc.GetHardnessFreeParams() * 2
    #     for i in range(npar):
    #         xfree[off+i] = min(ub[off+i],max(lb[off+i],xfree[off+i]))
    #     off += npar
    # if optzscl:
    #     lb[off] = 0.1
    #     ub[off] = 1.
    #     xfree[off] = 1.
    #     off += 1
        
    # xfree[-1] = 1.
        
        
    def SetOptParamsFromArray(self,x,mc):
        nfree = mc.nfree
        npar = mc.npar
        off = 0
        if self.opt_q:
            self.q = mc.GetAtomParamsFromFreeParams(x[off:off+nfree])
            off += nfree
        if self.opt_chempot:
            self.chempot = x[off:off+npar]
            off += npar
        if self.opt_hardness:
            self.hardness = x[off:off+npar]
            off += npar
        if self.opt_zetascl:
            self.zetascl = x[off]
            off += 1
        #self.SetMoleculeParams(mc)

        
    def SetOptParamsFromMolecules(self,mc):
        for m in mc.mols:
            for i in range(len(m.paridxs)):
                k = m.paridxs[i]
                self.q[k] = m.parm.atoms[i].charge
        self.hardness[:] = mc.hardness[:]
        self.chempot[:] = mc.chempot[:]
        self.zetascl = mc.zetascl

            
    def SetMoleculeParams(self,mc):
        for m in mc.mols:
            for i in range(len(m.paridxs)):
                k = m.paridxs[i]
                m.parm.atoms[i].charge = self.q[k]
        mc.hardness[:] = self.hardness[:]
        mc.chempot[:] = self.chempot[:]
        mc.zetascl = self.zetascl

            
    def GetOptParams(self,mc):
        import numpy as np
        x  = []
        lb = []
        ub = []
        if self.opt_q:
            qfree = mc.GetFreeParamsFromAtomParams(self.q)
            x.extend( [ q for q in qfree ] )
            lb.extend( [-np.inf]*len(qfree) )
            ub.extend( [np.inf]*len(qfree) )
            #lb.extend( self.lb_q[:len(qfree)] )
            #ub.extend( self.lb_q[:len(qfree)] )

        if self.opt_chempot:
            x.extend( [ mu for mu in self.chempot ] )
            #lb.extend( [-2]*len(self.chempot) )
            #ub.extend( [0]*len(self.chempot) )
            lb.extend( self.lb_chempot[:len(self.chempot)] )
            ub.extend( self.ub_chempot[:len(self.chempot)] )
        if self.opt_hardness:
            x.extend( [ h for h in self.hardness ] )
            #lb.extend( [0.1]*len(self.hardness) )
            #ub.extend( [50.]*len(self.hardness) )
            lb.extend( self.lb_hardness[:len(self.hardness)] )
            ub.extend( self.ub_hardness[:len(self.hardness)] )
        if self.opt_zetascl:
            x.append( self.zetascl )
            #lb.append( 0.1 )
            #ub.append( 1.0 )
            lb.append( self.lb_zetascl )
            ub.append( self.ub_zetascl )
        for i in range(len(x)):
            x[i] = max(lb[i],min(ub[i],x[i]))
        return np.array(x),np.array(lb),np.array(ub)

    


def HardnessObjective(xfree,mc,params,return_grad):
    import numpy as np
    import copy

    #mc = copy.deepcopy(inpmc)
    params.SetOptParamsFromArray(xfree,mc)
    
    pen=0.

    nfree = mc.nfree
    npar = mc.npar

    q0 = params.q
    glb_dpendq = np.zeros( (npar,) )
    glb_dpendu = np.zeros( (npar,) )
    glb_dpendh = np.zeros( (npar,) )
    glb_dpendz = 0

    for m in mc.mols:
        nat = len(m.paridxs)
        myq = q0[ m.paridxs ]
        myz = params.zetascl
        myh = params.hardness[ m.paridxs ]
        myu = params.chempot[ m.paridxs ]
        mydh = np.zeros((nat,))
        mydq = np.zeros((nat,))
        mydu = np.zeros((nat,))
        for c in m.conformers:
            
            atomzs      = myz * myh**2 * np.pi * 0.5
            datomzsdscl =       myh**2 * np.pi * 0.5
            datomzsdh   = myz * myh    * np.pi
            gauB,gauBdzeta = c.extsurf.CptGaussianInteractionMatrixAndGrd(atomzs)
            ptESP,gauESP,gauESPdzeta = c.espsurf.CptESPMatricesAndGrd(atomzs)
            gauESPdscl = datomzsdscl[:,np.newaxis] * gauESPdzeta
            gauESPdh = datomzsdh[:,np.newaxis] * gauESPdzeta

            for idesp in range(len(c.desps)):
                desp = c.desps[idesp]
                
                surfqs = np.array(desp.extvals_pos,copy=True)
                myb = myu + gauB @ surfqs
                E,dq_pos,dqdb_pos,dqdh_pos,dqdz_pos = c.SolveInterCPE(myb,myh,myz)
                dqdh_pos += (dqdb_pos*datomzsdh) * (gauBdzeta @ surfqs)
                dqdz_pos += (dqdb_pos*datomzsdscl) @ (gauBdzeta @ surfqs)
            
                mdlesp_pos = myq @ ptESP + dq_pos @ gauESP
                
                surfqs = np.array(desp.extvals_neg,copy=True)
                myb = myu + gauB @ surfqs
                E,dq_neg,dqdb_neg,dqdh_neg,dqdz_neg = c.SolveInterCPE(myb,myh,myz)
                dqdh_neg += (dqdb_neg*datomzsdh) * (gauBdzeta @ surfqs)
                dqdz_neg += (dqdb_neg*datomzsdscl) @ (gauBdzeta @ surfqs)
            
                mdlesp_neg = myq @ ptESP + dq_neg @ gauESP

                mdlesp = mdlesp_pos - mdlesp_neg
                tgtesp = desp.despvals                
                wts = c.espwts

                dq   = dq_pos - dq_neg
                dqdh = dqdh_pos - dqdh_neg
                dqdz = dqdz_pos - dqdz_neg
                dqdb = dqdb_pos - dqdb_neg

                #for x,y in zip(mdlesp,tgtesp):
                #    print("%12.4e %12.4e %12.4e"%(x,y,x-y))
                
                deltaesp = mdlesp - tgtesp
                pen += 0.5 * np.dot( wts, deltaesp**2 )

                #
                # X = (1/2) \sum_{e} w_e (p_{e}-p_{0,e})^2
                #
                # p_e = \sum_{a} q_a A_{a,e} + \sum_a c_a B_{a,e}(h,s)
                #
                
                # dX/dp_e
                dpen = wts * deltaesp
                
                # dX/dq_a = \sum_e dX/dp_e dp_e/dq_a
                #dpendq = 0. # ptESP @ dpen
                
                # dX/ds = \sum_e dX/dp_e \sum_a dp_e/dc_a|_{fixed s} dc_a/ds
                #       + \sum_e dX/dp_e \sum_a dp_e/ds|_{fixed c}
                #
                # dp_e/dc_a = B_{a,e}
                # dp_e/ds = \sum_a c_a dB/ds
                dpendz = dqdz @ gauESP @ dpen + dq @ gauESPdscl @ dpen
                dpendh = dqdh.T @ gauESP @ dpen + dq[:,np.newaxis] * gauESPdh @ dpen
                dpendu = dqdb @ gauESP @ dpen
            
                glb_dpendz += dpendz
                mydh += dpendh[:]
                #mydq += dpendq[:]
                mydu += dpendu[:]
        for i in range(nat):
            glb_dpendh[m.paridxs[i]] += mydh[i]
            glb_dpendq[m.paridxs[i]] += mydq[i]
            glb_dpendu[m.paridxs[i]] += mydu[i]


    retvals = []

    if params.opt_q:
        gfreeq = mc.GetFreeParamsFromAtomParams(glb_dpendq)
        retvals.append(gfreeq)

    if params.opt_chempot:
        retvals.append(glb_dpendu)
        
    if params.opt_hardness:
        retvals.append(glb_dpendh)

    if params.opt_zetascl:
        retvals.append([glb_dpendz])
    
    gfree = np.concatenate( retvals )
    
    print("objfcn: %15.7e"%(pen))

    if return_grad:
        return pen,gfree
    else:
        return pen




def FixedChargeObjective(xfree,mc,params,penfc):
    import numpy as np
    import copy
    
    pen=0.

    #mc = copy.deepcopy(inpmc)
    params.SetOptParamsFromArray(xfree,mc)
    
    #p = mc.GetAtomParamsFromFreeParams(xfree)
    p = params.q
    g = np.zeros( params.q.shape )
    gfree = np.zeros( xfree.shape )

    penfcs = np.zeros( (mc.npar,) )
    for m in mc.mols:
        penfcs[m.paridxs] = penfc * m.refhardness[:]
    
    for m in mc.mols:
        q = p[ m.paridxs ]
        myg = np.zeros( (len(q),) )
        for c in m.conformers:
            pot = q @ c.Bmat
            dp = pot-c.espvals
            #wts = np.array([ elem.quadwt * elem.switchwt
            #                 for elem in c.espsurf.elems ])
            #print(wts)
            pen += 0.5 * np.dot(dp,c.espwts * dp)
            myg += c.Bmat @ (c.espwts * dp)
        for i in range(len(myg)):
            g[ m.paridxs[i] ] += myg[i]

    for i in range(len(p)):
        eta = penfcs[i]
        pen += 0.5 * eta * p[i]**2
        g[i] += eta * p[i]
        if p[i] > 2.:
            d = p[i]-2.
            pen += 0.5 * 10. * d*d
            g[i] += 10.*d
        if p[i] < -1:
            d = p[i]+1.
            pen += 0.5 * 10. * d*d
            g[i] += 10.*d
            
    h = mc.GetFreeParamsFromAtomParams(g)
    for i in range(len(gfree)):
        gfree[i] = h[i]

    #print("%15.7e  %s"%(pen, " ".join(["%12.3f"%(x) for x in p])))
    #print("%15s  %s"%(""," ".join(["%12s"%(x) for x in mc.glbparams])))

    print("objfcn: %15.7e"%(pen))

    return pen,gfree



def FixedChargeAndCPEObjective(xfree,mc,params,penfc,skipcpe,return_grad):
    import numpy as np
    from scipy.special import erf

    #mc = copy.deepcopy(inpmc)
    params.SetOptParamsFromArray(xfree,mc)
    
    pen=0.

    nfree = mc.nfree
    npar = mc.npar

    if params.opt_q:
        q0 = np.array(params.q,copy=True)
    else:
        q0 = np.zeros( params.q.shape )
    
    glb_dpendq = np.zeros( (npar,) )
    glb_dpendu = np.zeros( (npar,) )
    glb_dpendh = np.zeros( (npar,) )
    glb_dpendz = 0

    for m in mc.mols:
        nat = len(m.paridxs)
        myq = q0[ m.paridxs ]
        mydq = np.zeros((nat,))
        if not skipcpe:
            myz = params.zetascl
            myh = params.hardness[ m.paridxs ]
            myu = params.chempot[ m.paridxs ]
            mydh = np.zeros((nat,))
            mydu = np.zeros((nat,))
        for c in m.conformers:

            E=0.
            if not skipcpe:
                surfqs = np.array([ elem.q
                                    for elem in c.extsurf.elems ])

                atomzs      = myz * myh**2 * np.pi * 0.5
                datomzsdscl =       myh**2 * np.pi * 0.5
                datomzsdh   = myz * myh    * np.pi

                gauB,gauBdzeta = c.extsurf.CptGaussianInteractionMatrixAndGrd(atomzs)
                myb = myu + gauB @ surfqs

                E,dq,dqdb,dqdh,dqdz = c.SolveInterCPE(myb,myh,myz)
                dqdh += (dqdb*datomzsdh) * (gauBdzeta @ surfqs)
                dqdz += (dqdb*datomzsdscl) @ (gauBdzeta @ surfqs)

                ptESP,gauESP,gauESPdzeta = c.espsurf.CptESPMatricesAndGrd(atomzs)

                mdlesp = myq @ ptESP + dq @ gauESP
                gauESPdscl = datomzsdscl[:,np.newaxis] * gauESPdzeta
                gauESPdh = datomzsdh[:,np.newaxis] * gauESPdzeta

                dqabs = np.mean(np.abs(dq))
            else:
                ptESP = c.Bmat
                mdlesp = myq @ c.Bmat
                dqabs = 0.

            tgtesp = c.espvals
            wts = c.espwts
            
            deltaesp = mdlesp - tgtesp
            mypen = 0.5 * np.dot( wts, deltaesp**2 )
            pen += mypen
            
            #print("%20.10e %20.10e %20.10e"%(mypen,E,dqabs))
            
            #
            # X = (1/2) \sum_{e} w_e (p_{e}-p_{0,e})^2
            #
            # p_e = \sum_{a} q_a A_{a,e} + \sum_a c_a B_{a,e}(h,s)
            #

            # dX/dp_e
            dpen = wts * deltaesp

            # dX/dq_a = \sum_e dX/dp_e dp_e/dq_a
            dpendq = ptESP @ dpen
            mydq += dpendq[:]


            
            if not skipcpe:
                # dX/ds = \sum_e dX/dp_e \sum_a dp_e/dc_a|_{fixed s} dc_a/ds
                #       + \sum_e dX/dp_e \sum_a dp_e/ds|_{fixed c}
                #
                # dp_e/dc_a = B_{a,e}
                # dp_e/ds = \sum_a c_a dB/ds
                dpendz = dqdz @ gauESP @ dpen + dq @ gauESPdscl @ dpen
                dpendh = dqdh.T @ gauESP @ dpen + dq[:,np.newaxis] * gauESPdh @ dpen
                dpendu = dqdb @ gauESP @ dpen
                glb_dpendz += dpendz
                mydh += dpendh[:]
                mydu += dpendu[:]

        if params.opt_q:
            # resp penalty
            xpen = 0
            for iat in range(nat):
                if m.atnums[iat] > 1:
                    w = np.sqrt( myq[iat]**2 + params.resp_b**2 )
                    mypen = params.resp_a * (w-params.resp_b)
                    ddq = (params.resp_a/w) * myq[iat]
                    xpen += mypen
                    pen += mypen
                    mydq[iat] += ddq
            #print("xpen=",xpen,params.resp_a,params.resp_b)
                
        for i in range(nat):
            glb_dpendq[m.paridxs[i]] += mydq[i]

        if not skipcpe:
            for i in range(nat):
                glb_dpendh[m.paridxs[i]] += mydh[i]
                glb_dpendu[m.paridxs[i]] += mydu[i]

    # if penfc > 0 and params.opt_q:
    #     refhardness = np.zeros( (mc.npar,) )
    #     for m in mc.mols:
    #         for i,k in enumerate(m.paridxs):
    #             refhardness[k] = m.refhardness[i]
    #     for i in range(mc.npar):
    #         q2             = q0[i]**2
    #         pen           += 0.5 * penfc * refhardness[i] * q2
    #         glb_dpendq[i] +=       penfc * refhardness[i] * q0[i]
    #         if q0[i] > 2.:
    #             d = q0[i]-2.
    #             pen += 0.5 * 10. * d*d
    #             glb_dpendq[i] += 10.*d
    #         if q0[i] < -1.:
    #             d = q0[i]+1.
    #             pen += 0.5 * 10. * d*d
    #             glb_dpendq[i] += 10.*d
        
                

    retvals = []

    if params.opt_q:
        gfreeq = mc.GetFreeParamsFromAtomParams(glb_dpendq)
        retvals.append(gfreeq)

    if params.opt_chempot:
        retvals.append(glb_dpendu)
        
    if params.opt_hardness:
        retvals.append(glb_dpendh)

    if params.opt_zetascl:
        retvals.append([glb_dpendz])
    
    gfree = np.concatenate( retvals )
    
    #print(xfree.shape,gfree.shape)
    #print("%15.7e  %s"%(pen, " ".join(["%12.3f"%(x) for x in q0])))
    #print("%15s  %s"%(""," ".join(["%12s"%(x) for x in mc.glbparams])))
    print("objfcn: %15.7e"%(pen))
    #print("gfree=",gfree)
    
    if return_grad:
        return pen,gfree
    else:
        return pen








def OptimizeHardness(mc,params):
    import numpy as np
    from scipy.optimize import minimize
    from scipy.optimize import Bounds

    # nfree = mc.nfree
    # npar = mc.npar

    # optchempot = False
    # optzscl = True
    # opthardness = True
    # optq = False
    
    # nparam = 0
    # if optq:
    #     nparam += nfree
    # if optchempot:
    #     nparam += npar
    # if opthardness:
    #     nparam += npar
    # if optzscl:
    #     nparam += 1
    
    # xfree = np.zeros( (nparam,) )

    # lb = np.zeros( (nparam,) )
    # ub = np.zeros( (nparam,) )

    # off = 0
    # if optq:
    #     lb[off:off+nfree] = -np.inf
    #     ub[off:off+nfree] =  np.inf
    #     off += nfree
    # if optchempot:
    #     lb[off:off+npar] = -2.
    #     ub[off:off+npar] =  0.
    #     off += npar
    # if opthardness:
    #     lb[off:off+npar] =  0.1
    #     ub[off:off+npar] = 50.00
    #     xfree[off:off+npar] = mc.GetHardnessFreeParams() * 2
    #     for i in range(npar):
    #         xfree[off+i] = min(ub[off+i],max(lb[off+i],xfree[off+i]))
    #     off += npar
    # if optzscl:
    #     lb[off] = 0.1
    #     ub[off] = 1.
    #     xfree[off] = 1.
    #     off += 1
        
    # xfree[-1] = 1.

    xfree,lb,ub = params.GetOptParams(mc)

    xfree[:mc.npar] *= 2
    for i in range(mc.npar):
        xfree[i] = min(ub[i],max(lb[i],xfree[i]))
    
    if True:
        res = minimize(HardnessObjective, xfree,
                       args=(mc,params,False),
                       method="COBYLA",
                       jac=False,
                       bounds=Bounds(lb,ub),
                       options={"maxiter":250,
                                "disp":2,
                                "rhobeg": 0.025,
                                "tol": 1.e-3})
        xfree = res.x

    
    res = minimize(HardnessObjective, xfree,
                   args=(mc,params,True),
                   method="L-BFGS-B",
                   jac=True,
                   bounds=Bounds(lb,ub),
                   options={"maxiter":10000,
                            "disp":True,
                            "ftol":1.e-15,
                            "gtol":1.e-10})
    print(res)
    return res.x



def OptimizeCPE(mc,params,penfc,skipcpe):
    import numpy as np
    from scipy.optimize import minimize
    from scipy.optimize import Bounds

    # nfree = mc.nfree
    # npar = mc.npar

    # optchempot = True
    # optzscl = True
    # opthardness = True
    # optq = False
    
    # nparam = 0
    # if optq:
    #     nparam += nfree
    # if optchempot:
    #     nparam += npar
    # if opthardness:
    #     nparam += npar
    # if optzscl:
    #     nparam += 1
    
    # xfree = np.zeros( (nparam,) )

    # lb = np.zeros( (nparam,) )
    # ub = np.zeros( (nparam,) )

    # off = 0
    # if optq:
    #     lb[off:off+nfree] = -np.inf
    #     ub[off:off+nfree] =  np.inf
    #     off += nfree
    # if optchempot:
    #     lb[off:off+npar] = -2.
    #     ub[off:off+npar] =  0.
    #     off += npar
    # if opthardness:
    #     lb[off:off+npar] =  0.1
    #     ub[off:off+npar] = 50.0
    #     xfree[off:off+npar] = mc.GetHardnessFreeParams() * 2
    #     for i in range(npar):
    #         xfree[off+i] = min(ub[off+i],max(lb[off+i],xfree[off+i]))
    #     off += npar
    # if optzscl:
    #     lb[off] = 0.01
    #     ub[off] = 1.
    #     xfree[off] = 1.
    #     off += 1
        
    # xfree[-1] = 1.

    xfree,lb,ub = params.GetOptParams(mc)
    
    
    if params.opt_chempot or params.opt_hardness:
        res = minimize(FixedChargeAndCPEObjective, xfree,
                       (mc,params,penfc,skipcpe,False),
                       method="COBYLA",
                       jac=False,
                       bounds=Bounds(lb,ub),
                       options={"maxiter":250,
                                "disp":2,
                                "rhobeg": 0.025,
                                "tol": 1.e-3})
        xfree = res.x

    res = minimize(FixedChargeAndCPEObjective, xfree,
                   args=(mc,params,penfc,skipcpe,True),
                   method="L-BFGS-B",
                   jac=True,
                   bounds=Bounds(lb,ub),
                   options={"maxiter":10000,
                            "disp":True,
                            "ftol":1.e-15,
                            "gtol":1.e-10})
    print(res)
    return res.x



def OptimizeFixedCharge(mc,params,penfc):
    import numpy as np
    from scipy.optimize import minimize
    from scipy.optimize import Bounds

    xfree,lb,ub = params.GetOptParams(mc)
    
    # nfree = mc.nfree
    # npar = mc.npar
    # nparam = nfree
    # xfree = np.zeros( (nparam,) )
    # lb = np.zeros( (nparam,) )
    # ub = np.zeros( (nparam,) )

    # off = 0
    # lb[off:off+nfree] = -np.inf
    # ub[off:off+nfree] =  np.inf

    res = minimize(FixedChargeObjective, xfree,
                   args=(mc,params,penfc),
                   method="L-BFGS-B",
                   jac=True,
                   bounds=Bounds(lb,ub),
                   options={"maxiter":10000,
                            "disp":True,
                            "ftol":1.e-15,
                            "gtol":1.e-10})
    print(res)
    return res.x




def AssignUniqueParams(mols,digits=3,verbose=True):
    from collections import OrderedDict as ddict
    import numpy as np
    
    allatomnames = []
    allchargestr = []
    allatomnums  = []

    fmtstr = "%%.%if"%(digits)
    for imol,mol in enumerate(mols):
        allatomnames.extend( mol.atnames )
        allchargestr.extend( [ fmtstr%(a.charge) for a in mol.parm.atoms ] )
        atnums = mol.get_group_shifted_ids() # 
        offset = 1000 * ( imol + 1 )
        glbids = []
        for i in range(len(atnums)):
            # if an atom is part of a group, then its id has been shifted
            # so it is > 100. If it is part of a group, then we prevent it
            # from being constrained to charges in other molecules.
            #
            # Only atoms not part of a group can share parameters with
            # atoms in other molecules
            #if atnums[i] > 100:
            #    atnums[i] += offset
            #
            # I am changing the output of get_group_shifted_ids()
            # so it is no longer an integer; it is now a list of integers,
            # where the first element is the atomic number, and the
            # remaining integers flag the group occupancies for that molecule
            #
            # I will create a global id string here.
            # If all group occupancies are zero, then the string will
            # simply be the atomic number (but as a string)
            # If at least one occupancy is nonzero, then the molecule
            # offset is added to the nonzero occupancy, and the string
            # concatenates the id list separated by underscores.
            #
            occsum = 0
            if len(atnums[i]) > 1:
                occsum = sum(atnums[i][1:])
            if occsum == 0:
                glbids.append( str(atnums[i][0]) )
            else:
                for j in range(len(atnums[i])-1):
                    if atnums[i][j+1] > 0:
                        atnums[i][j+1] += offset
                glbids.append( "_".join( ["%s"%(x) for x in atnums[i]] ) )
        allatomnums.extend(glbids)

    parmap = ddict()
    for name,zq in zip(allatomnames,zip(allatomnums,allchargestr)):
        if zq not in parmap:
            parmap[zq] = [name]
        else:
            parmap[zq].append(name)

    glbparams = [ names[0] for zq,names in parmap.items() ]

    #print(glbparams)
    
    for zq,names in parmap.items():
        refname = names[0]
        for mol in mols:
            for i in range(len(mol.atnames)):
                name = mol.atnames[i]
                if name in names:
                    mol.parnames[i] = refname
                    mol.paridxs[i] = glbparams.index(refname)

    if verbose:
        for mol in mols:
            for i in range(len(mol.atnames)):
                name = mol.atnames[i]
                refname = mol.parnames[i]
                print("Atom: %-9s  => Param: %-9s (%i)"%(name,refname,mol.paridxs[i]))
                    
    return np.array(glbparams)



def SetUniqueParams(mols,parammap):
    #
    # parammap is a list of list
    # Each row in the list is a unique parameter
    # Each column in the sublist is a residue:atom name
    # The first residue:atom name in the sublist is the
    # name of the unique parameter
    #
    # The residue:atom name is stored in the mol.atnames attribute
    #
    # The unique global parameter name is stnored in mol.parnames
    #
    from collections import defaultdict as ddict
    import numpy as np
    
    glbparams = [ names[0] for names in parmmap ]

    found_params = ddict(bool)
    for mol in mols:
        for i in range(len(mol.atnames)):
            name = mol.atnames[i]
            found_params[name] = False
    
    for names in parmmap:
        refname = names[0]
        for mol in mols:
            for i in range(len(mol.atnames)):
                name = mol.atnames[i]
                if name in names:
                    mol.parnames[i] = refname
                    mol.paridxs[i] = glbparams.index(refname)
                    found_params[name] = True
                
    missing_params = [ name for name in found_params
                       if not found_params[name] ]

    if len(missing_params) > 0:
        raise Exception("Not all atoms are associated with "
                        +"a unique parameter: %s"%(" ".join(missing_params)))
    
    return np.array(glbparams)



class MoleculeCollection(object):
    def __init__(self,mols,digits=3,parammap=None,verbose=True):
        import copy
        import numpy as np
        
        
        self.mols = [ copy.copy(m) for m in mols ]
        if parammap is None:
            self.glbparams = AssignUniqueParams(self.mols,digits=digits,verbose=verbose)
        else:
            self.glbparams = SetUniqueParams(self.mols,parammap)
            
        self.convals = None
        self.conmat = None
        npar = len(self.glbparams)
        self.nfree = npar
        self.npar = npar
        
        for mol in self.mols:
            m,v = mol.get_glb_constraints(self.glbparams)
            ncon = len(v)
            for icon in range(ncon):
                if self.convals is None:
                    self.convals = np.array([v[icon]])
                    self.conmat = np.array([m[icon,:]])
                else:
                    idx_exists = None
                    for j in range(self.conmat.shape[0]):
                        if np.linalg.norm( m[icon,:]-self.conmat[j,:] ) < 1.e-8:
                            idx_exists = j
                    if idx_exists is not None:
                        if abs(self.convals[idx_exists]-v[icon]) > 1.e-8:
                            msg="Attempted to push an existing constraint "\
                                +"with a different constraint value"
                            raise Exception(msg)
                        else:
                            continue
                    else:
                        self.convals = np.concatenate( (self.convals,[v[icon]]) )
                        self.conmat = np.concatenate( (self.conmat,m[icon:icon+1,:]),axis=0 )
        ncon = self.conmat.shape[0]
        if len(self.convals) > 0:
            U,s,Vt = np.linalg.svd(self.conmat,
                                   full_matrices=True,
                                   compute_uv=True,
                                   hermitian=False)
            
            
            nsingular = (self.conmat.shape[1] - self.conmat.shape[0]) \
                + sum( [ 1 for x in s if abs(x) < 1.e-8 ] )
            
            if nsingular != npar - ncon:
                raise Exception(f"Found {nsingular} singular values; "
                                +f"expected {npar-ncon}")
            sidxs = []
            smat = np.zeros( (ncon,npar) )
            for i in range(npar):
                if i < len(s):
                    if abs(s[i]) > 1.e-8:
                        smat[i,i] = 1./s[i]
                    else:
                        smat[i,i] = 0.
                        sidxs.append(i)
                else:
                    sidxs.append(i)
            Cinv = np.dot(Vt.T, np.dot(smat.T, U.T))
            self.free2atom_vec = Cinv @ self.convals
            self.free2atom_mat = Vt[sidxs,:]
            self.nfree = self.free2atom_mat.shape[0]

        self.hardness = np.zeros( (self.npar,) )
        self.chempot = np.zeros( (self.npar,) )
        self.zetascl = 1.
        for m in self.mols:
            self.hardness[ m.paridxs ] = m.refhardness
            
    # def GetHardnessFreeParams(self):
    #     import numpy as np
    #     hfree = np.zeros( (self.npar,) )
    #     for m in self.mols:
    #         for a in range(len(m.paridxs)):
    #             k = m.paridxs[a]
    #             hfree[k] = m.hardness[a]
    #     return hfree

    
    def GetAtomParamsFromFreeParams(self,q):
        import numpy as np
        return self.free2atom_vec + np.dot(q,self.free2atom_mat)

    
    def GetFreeParamsFromAtomParams(self,p):
        import numpy as np
        return np.dot(self.free2atom_mat,p)

    
    def MakeParams(self):
        p = ParamListType(self.npar)
        p.SetOptParamsFromMolecules(self)
        return p
    
    
    def OptimizeFixedCharge(self,params,penfc=0):
        params.opt_q = True
        params.opt_hardness = False
        params.opt_chempot = False
        params.opt_zetascl = False
        x = OptimizeCPE(self,params,penfc,True)
        params.SetOptParamsFromArray(x,self)
        params.SetMoleculeParams(self)

    
    def OptimizeFixedChargeAndCPE(self,params,penfc=0):
        x = OptimizeCPE(self,params,penfc,False)
        params.SetOptParamsFromArray(x,self)
        params.SetMoleculeParams(self)

    
    def OptimizeCPE(self,params,penfc=0):
        params.opt_q = False
        x = OptimizeCPE(self,params,False)
        params.SetOptParamsFromArray(x,self)
        params.SetMoleculeParams(self)


    def OptimizeHardness(self,params,opt_zetascl=True):
        params.opt_q = False
        params.opt_hardness = True
        params.opt_chempot = False
        params.opt_zetascl = opt_zetascl
        x = OptimizeHardness(self,params)
        params.SetOptParamsFromArray(x,self)
        params.SetMoleculeParams(self)
    
