#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import LFPy
from pylab import *

cell = LFPy.Cell(morphology='morphologies/L5_Mainen96_LFPy.hoc', passive=True)

synapse = LFPy.Synapse(cell,
                       idx=cell.get_idx("soma[0]"),
                       syntype='Exp2Syn',
                       weight=0.005,
                       e=0,
                       tau1=0.5,
                       tau2=2,
                       record_current=True)

synapse.set_spike_times(array([20., 40]))

cell.simulate()

figure(figsize=(12, 9))
subplot(222)
plot(cell.tvec, synapse.i, 'r'), title('synaptic current (pA)')
subplot(224)
plot(cell.tvec, cell.somav, 'k'), title('somatic voltage (mV)')
subplot(121)
for sec in LFPy.cell.neuron.h.allsec():
    idx = cell.get_idx(sec.name())
    plot(np.r_[cell.xstart[idx], cell.xend[idx][-1]],
         np.r_[cell.zstart[idx], cell.zend[idx][-1]],
         color='k')
plot([cell.synapses[0].x], [cell.synapses[0].z],
     color='r', marker='o', markersize=10)
axis([-500, 500, -400, 1200])

show()
