# -*- coding: utf-8 -*-
"""
run_all.py  -- regenerate all 20 analysis figures (A/B/C/D/E x 4 each)
Usage: cd Analysis_Plots/scripts && python run_all.py
"""

print("=== Running all analysis figures ===\n")

print("Metric A: Class composition (count fraction)")
from metric_a_class_composition import fig1A, fig2A, fig3A, fig3A_byday
fig1A(); fig2A(); fig3A(); fig3A_byday()
print()

print("Metric B: Mean radius per class")
from metric_b_mean_radius import fig1B, fig2B, fig3B, fig3B_byday
fig1B(); fig2B(); fig3B(); fig3B_byday()
print()

print("Metric C: Mean volume per class")
from metric_c_mean_volume import fig1C, fig2C, fig3C, fig3C_byday
fig1C(); fig2C(); fig3C(); fig3C_byday()
print()

print("Metric D: Volume fraction per class")
from metric_d_volume_fraction import fig1D, fig2D, fig3D, fig3D_byday
fig1D(); fig2D(); fig3D(); fig3D_byday()
print()

print("Metric E: Droplet density per class (per magnification)")
from metric_e_density import fig1E, fig2E, fig3E, fig3E_byday
fig1E(); fig2E(); fig3E(); fig3E_byday()
print()

print("=== All 20 figures saved to Analysis_Plots/output/ ===")
