#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py
import numpy as np
import xarray as xr
import pandas as pd
import pickle
import os
from pathlib import Path
import argparse

from py_wake.site.xrsite import XRSite
from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine
from py_wake.site import UniformSite  # Changed from GlobalWindAtlasSite
from py_wake.wind_turbines import WindTurbines
from py_wake import Nygaard_2022
from py_wake.wind_turbines.power_ct_functions import SimpleYawModel, PowerCtTabular
from py_wake.wind_turbines._wind_turbines import WindTurbine
from precompute_farm_layouts import get_turbine_types

from topfarm._topfarm import TopFarmProblem
from topfarm.plotting import XYPlotComp
from topfarm.cost_models.py_wake_wrapper import PyWakeAEPCostModelComponent

from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.constraint_components.boundary import XYBoundaryConstraint


def get_layout_from_h5(farm_idx, type_idx, seed):
    """Get turbine layout from HDF5 file based on farm, type, and seed parameters."""
    config_key = f"farm{farm_idx}_t{type_idx}_s{seed}"
    
    with h5py.File(H5_FILE, 'r') as f:
        if config_key in f:
            # Get layout dataset which should be an array of shape (2, n_turbines)
            layout_data = f[config_key]['layout'][:]
            return layout_data[0], layout_data[1]  # x and y coordinates
        else:
            return None, None

def evaluate_sample(sample_no, output_dir, n_points=18, random_pct=50):
    """Evaluate a single sample configuration."""
    
    # Load sample data and time series
    samples = xr.load_dataset('samples.nc')
    ts = pd.read_csv('energy_island_10y_daily_av_wind.csv', sep=';', parse_dates=True)
    
    # Setup simple uniform site with TI=0.1
    site = XRSite.load('ref_site.nc')

    # Get sample data
    sample_data = samples.sel(sample=sample_no)
    rated_powers = sample_data.rated_power.values
    construction_days = sample_data.construction_day.values
    ss_seeds = sample_data.ss_seed.values
    
    # Setup wind turbines
    wt_ref = GenericWindTurbine('WT_ref', 240, 150, 15e3)
    RPs = np.arange(10, 16).astype(int)
    Ds = (240 * np.sqrt(RPs / 15)).astype(int)
    hhs = (Ds / 240 * 150).astype(int)
    wt_list = [
        GenericWindTurbine(f'WT_{n}', d, h, rp * 1e3) 
        for n, d, h, rp in zip(np.arange(6), Ds, hhs, RPs)
    ]
    
    # Add no-wake reference turbine
    # this is used for computing external wake losses
    # since the internal turbines are fixed, we only need this one additional turbine type
    u = wt_ref.powerCtFunction.ws_tab
    p, _ = wt_ref.powerCtFunction.power_ct_tab
    ct = np.zeros_like(p)
    powerCtFunction = PowerCtTabular(u, p, 'w', ct, ws_cutin=None, ws_cutout=None,
                                   power_idle=0, ct_idle=None, method='linear',
                                   additional_models=[SimpleYawModel()])
    wt_ref_no_wake = WindTurbine('WT_ref_no_wake', 240, 150, powerCtFunction)
    wt_list.append(wt_ref_no_wake)
    
    target_layout = get_layout_from_h5(farm_idx=0, type_idx=5, seed=0)
    x_target = target_layout[0]
    y_target = target_layout[1]
    
    # Get neighbor farm layouts
    x_neighbours = []
    y_neighbours = []
    nwt_list = []
    
    # iterate through eafch farm
    for n, (turbid, seed) in enumerate(zip(sample_data.rated_power - 10, ss_seeds)):
        # rp = turbs[turbid]?
        #nwt = int(1000/rp)  # Calculate number of turbines based on rated power
        #if n == 0: continue
        nwt = 1000 // int(turbid + 10)
        nwt_list.append(nwt)
        
        neighbor_layout = get_layout_from_h5(farm_idx=n, type_idx=int(turbid), seed=seed)
        x_neighbours.append(list(neighbor_layout[0]))
        y_neighbours.append(list(neighbor_layout[1]))
        assert(neighbor_layout[0].size == nwt) # todo: wrong size atm

    # Run simulations for each time period
    sequence = np.argsort(construction_days)
    construction_days_sorted = np.sort(construction_days)
    wf_model = Nygaard_2022(site, WindTurbines.from_WindTurbine_lst(wt_list))
    
    for m in range(10):
        wake_path = os.path.join(output_dir, f'wake/res_{sample_no}_{m}.nc')
        no_wake_path = os.path.join(output_dir, f'no_wake/res_{sample_no}_{m}.nc')
        
        os.makedirs(os.path.dirname(wake_path), exist_ok=True)
        os.makedirs(os.path.dirname(no_wake_path), exist_ok=True)
        
        # Get time period data
        if m == 0:
            ts_part = ts.iloc[:construction_days_sorted[0]]
        elif m == 9:
            ts_part = ts.iloc[construction_days_sorted[8]:]
        else:
            ts_part = ts.iloc[construction_days_sorted[m-1]:construction_days_sorted[m]]
        
        if ts_part.size > 0:
            active_farms = sequence[:m]
            x = np.asarray(x_target)
            y = np.asarray(y_target)
            types_active = (5 * np.ones(66)).astype(int)  # Target farm type
            types_active_no_wake = (6 * np.ones(66)).astype(int)  # No-wake reference
            
            # Add active farms to layout
            for af in active_farms:
                x = np.concatenate([x, x_neighbours[af]])
                y = np.concatenate([y, y_neighbours[af]])
                types_active = np.concatenate([
                    types_active, 
                    (rated_powers[af]-10) * np.ones(nwt_list[af])
                ])
                types_active_no_wake = np.concatenate([
                    types_active_no_wake, 
                    (rated_powers[af]-10) * np.ones(nwt_list[af])
                ])
# Run wake simulation
            if not os.path.exists(wake_path):
                sim_res = wf_model(
                    x, y,
                    type=types_active,
                    wd=ts_part.WD_150,
                    ws=ts_part.WS_150,
                    time=ts_part.index,
                    TI=0.1
                )
                sim_res.save(wake_path)
            
            # Run no-wake simulation
            if not os.path.exists(no_wake_path):
                sim_res_no_wake = wf_model(
                    x, y,
                    type=types_active_no_wake,
                    wd=ts_part.WD_150,
                    ws=ts_part.WS_150,
                    time=ts_part.index,
                    TI=0.1
                )
                sim_res_no_wake.save(no_wake_path)

def evaluate_samples(start_sample, end_sample, output_dir):
    """Evaluate a range of samples."""
    os.makedirs(output_dir, exist_ok=True)
    
    for sample_no in range(start_sample, end_sample + 1):
        print(f"Processing sample {sample_no}")
        evaluate_sample(sample_no, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate wind farm samples")
    parser.add_argument("--start", type=int, required=True, help="Starting sample number")
    parser.add_argument("--end", type=int, required=True, help="Ending sample number")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()
    
    H5_FILE = "re_precomputed_layouts.h5"
    evaluate_samples(args.start, args.end, args.output)
