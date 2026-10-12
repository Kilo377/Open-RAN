"""Gray-QAM BICM mutual information, integrated over AWGN by quadrature.

This is MIESM, not the Sionna default EESM. The constellation and bit labels
come from Sionna. Per-resource SINRs enter before averaging, preserving QAM's
finite information ceiling on strong resources in a selective channel.
"""
from functools import lru_cache
import numpy as np
from scipy.special import logsumexp


@lru_cache(maxsize=4)
def qam_information_table(bits):
    from sionna.phy.mapping import Constellation
    points=Constellation('qam',bits,device='cpu').points.detach().numpy()
    labels=((np.arange(2**bits)[:,None] >> np.arange(bits-1,-1,-1)) & 1)
    levels,indices=np.unique(points.real,return_index=True)
    bit_labels=labels[indices,::2]
    nodes,weights=np.polynomial.hermite.hermgauss(64)
    grid=np.arange(-30.,40.001,.025);information=[]
    for db in grid:
        noise=10**(-db/10)
        received=levels[:,None]+np.sqrt(noise)*nodes[None,:]
        likelihood=-(received[:,:,None]-levels[None,None,:])**2/noise
        total=0.
        for bit in range(bits//2):
            llr=logsumexp(likelihood[:,:,bit_labels[:,bit]==1],axis=-1)-logsumexp(likelihood[:,:,bit_labels[:,bit]==0],axis=-1)
            entropy=np.logaddexp(0,(1-2*bit_labels[:,bit,None])*llr)/np.log(2)
            total+=1-float(np.mean(entropy @ weights/np.sqrt(np.pi)))
        information.append(np.clip(2*total,0,bits))
    information=np.maximum.accumulate(information)
    return grid,np.asarray(information)


def effective_sinr(spec,values):
    values=np.asarray(values,float)
    if values.std()<=values.mean()*1e-12:
        return float(values.mean())
    grid,information=qam_information_table(int(spec['qam_bits']))
    mean_information=float(np.mean(np.interp(10*np.log10(np.maximum(values,1e-300)),grid,information)))
    # Remove saturated knots to keep the inverse unambiguous.
    unique,indices=np.unique(information,return_index=True)
    db=float(np.interp(mean_information,unique,grid[indices]))
    cv=float(values.std()/values.mean())
    shift=float(np.interp(cv,spec.get('cv_knots',[0.]),spec.get('shift_db',[0.])))
    slope=float(np.interp(cv,spec.get('cv_knots',[0.]),spec.get('slope',[1.]*len(spec.get('cv_knots',[0.])))))
    reference=float(spec.get('reference_db',0.))
    return float(10**((reference+slope*(db-reference)+shift)/10))
