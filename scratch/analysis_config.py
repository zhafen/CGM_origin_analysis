#!/usr/bin/env python
'''Config file for running an analysis suite.'''

SIM_NAMES = [
    'm10q',
    'm10v',
    'm10y',
    'm10z',
    'm11q',
    'm11v',
    'm11a',
    'm11b',
    'm11c',
    'm12i',
    'm12f',
    'm12m',
    'm11d_md',
    'm11e_md',
    'm11h_md',
    'm11i_md',
    'm12b_md',
    'm12c_md',
    'm12z_md',
    'm12r_md',
    'm12w_md',
]

# Simulations whose first three letters doesn't fit their mass bin.
MASS_BINS = {
    'm11a' : 'm10',
    'm11b' : 'm10',
}

SNUM = 172

GALDEF = '_galdefv3'

# Account for the fact that, at z>2, m10v is rather unreliable
if SNUM <= 172:
    SIM_NAMES.remove( 'm10v' )
