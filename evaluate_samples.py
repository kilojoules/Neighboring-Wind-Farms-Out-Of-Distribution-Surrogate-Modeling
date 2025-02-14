#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr
import pandas as pd
import pickle
import os
from pathlib import Path
import argparse

from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine
from py_wake.site import UniformSite  # Changed from GlobalWindAtlasSite
from py_wake.wind_turbines import WindTurbines
from py_wake import Nygaard_2022
from py_wake.wind_turbines.power_ct_functions import SimpleYawModel, PowerCtTabular
from py_wake.wind_turbines._wind_turbines import WindTurbine

from topfarm._topfarm import TopFarmProblem
from topfarm.plotting import XYPlotComp
from topfarm.cost_models.py_wake_wrapper import PyWakeAEPCostModelComponent

from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.constraint_components.boundary import XYBoundaryConstraint


# Farm boundary coordinates 
dk1d_tender_9 = np.array([
    695987.1049296035, 6201357.423000679,
    693424.6641628657, 6200908.39284379,
    684555.8480807217, 6205958.057693831,
    683198.5005206821, 6228795.090001539,
    696364.6957258906, 6223960.959626805,
    697335.3172284428, 6204550.027416158,]).reshape((-1,2)).T.tolist()

dk0z_tender_5 = np.array([
    696839.729308316, 6227384.493837256,
    683172.617280614, 6237928.363392656,
    681883.784179578, 6259395.212250201,
    695696.2991147212, 6254559.15746051,]).reshape((-1,2)).T.tolist()

dk0w_tender_3 = np.array([
    706694.3923283464, 6224158.532895836,
    703972.0844905999, 6226906.597455995,
    702624.6334635273, 6253853.5386425415,
    712771.6248419734, 6257704.934445341,
    715639.3355871611, 6260664.6846508905,
    721593.2420745814, 6257906.998015941,]).reshape((-1,2)).T.tolist()

dk0v_tender_1 = np.array([
    695387.9840492046, 6260724.982986244,
    690623.9453331482, 6265534.095966523,
    689790.3527486034, 6282204.661276843,
    706966.9276208277, 6287633.435873629,
    708324.2751808674, 6264796.40356592,
    696034.3037791394, 6260723.0585712865,]).reshape((-1,2)).T.tolist()

dk0Y_tender_4 = np.array([
    688971.224327626, 6289970.317104408,
    699859.6944068453, 6313455.877252994,
    706084.6136432136, 6313894.002391787,
    711981.4247481308, 6312278.1352986405,
    712492.2381035917, 6310678.304996811,
    705728.3384563944, 6295172.010736138,
    703484.1092881403, 6292667.063932351,
    695423.0025504732, 6290179.436863188,]).reshape((-1,2)).T.tolist()

dk0x_tender_2 = np.array([
    715522.0997350881, 6271624.869308893,
    714470.0221534981, 6296972.621665262,
    735902.8674733848, 6290515.568009201,
    726238.5223950314, 6268396.3424808625,]).reshape((-1,2)).T.tolist()

dk1a_tender_6 = np.array([
    741993.8028788125, 6285017.514473925,
    754479.4211245844, 6280870.400239231,
    755733.9969961186, 6260088.643106768,
    753546.8632103675, 6256441.8767611785,
    738552.854493294, 6267674.686871577,
    738130.3486627713, 6276124.151480917,]).reshape((-1,2)).T.tolist()

dk1b_tender7 = np.array([
    730392.0211541882, 6258565.789403263,
    741435.0294020491, 6261729.52759437,
    743007.816872067, 6238891.853815009,
    741806.5300242025, 6237068.79137804,
    729493.7204694732, 6233452.174200128,
    729032.3897788484, 6255601.548896144,]).reshape((-1,2)).T.tolist()

dk1c_tender_8 = np.array([
    719322.3683944925, 6234395.779001247,
    730063.1517509705, 6226372.251569298,
    720738.3338805687, 6206078.654364537,
    712391.7502303863, 6209300.125004388,
    709646.6042396385, 6212504.917381268,]).reshape((-1,2)).T.tolist()

dk1e_tender_10 = np.array([
    705363.6892801415, 6203384.473423205,
    716169.9420085563, 6202667.308115489,
    705315.7291588389, 6178496.656241821,
    693580.7248750407, 6176248.298099114,]).reshape((-1,2)).T.tolist()

def evaluate_sample(sample_no, output_dir, n_points=18, random_pct=50):
    """Evaluate a single sample configuration."""
    
    # Load sample data and time series
    samples = xr.load_dataset('samples.nc')
    ts = pd.read_csv('energy_island_10y_daily_av_wind.csv', sep=';', parse_dates=True)
    
    # Setup farm boundaries
    farm_list = [
        dk0w_tender_3,  # target farm
        dk1d_tender_9,
        dk0z_tender_5,
        dk0v_tender_1,
        dk0Y_tender_4,
        dk0x_tender_2,
        dk1a_tender_6,
        dk1b_tender7,
        dk1c_tender_8,
        dk1e_tender_10,
    ]
    
    # Setup simple uniform site with TI=0.1
    site = UniformSite(ti=0.1)

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
    u = wt_ref.powerCtFunction.ws_tab
    p, _ = wt_ref.powerCtFunction.power_ct_tab
    ct = np.zeros_like(p)
    powerCtFunction = PowerCtTabular(u, p, 'w', ct, ws_cutin=None, ws_cutout=None,
                                   power_idle=0, ct_idle=None, method='linear',
                                   additional_models=[SimpleYawModel()])
    wt_ref_no_wake = WindTurbine('WT_ref_no_wake', 240, 150, powerCtFunction)
    wt_list.append(wt_ref_no_wake)
    
    # Get target farm layout
    target_ss_file = f'ss_state_target_wf_{15}_{n_points}_{random_pct}.pkl'
    if os.path.exists(target_ss_file):
        with open(target_ss_file, 'rb') as f:
            state = pickle.load(f)
    else:
        nwt = 66  # Number of turbines in target farm
        bound = farm_list[0]
        x_init = (np.random.random(nwt)-0.5)*1000+np.mean(bound[0])
        y_init = (np.random.random(nwt)-0.5)*1000+np.mean(bound[1])
        
        wf_model = Nygaard_2022(site, wt_ref)
        problem = TopFarmProblem(
            design_vars={'x': x_init, 'y': y_init},
            cost_comp=PyWakeAEPCostModelComponent(wf_model, nwt, grad_method='autograd'),
            constraints=[
                XYBoundaryConstraint(np.asarray(bound).T, boundary_type='polygon'),
                SpacingConstraint(wt_ref.diameter())
            ],
            plot_comp=XYPlotComp()
        )
        
        xs = np.linspace(np.min(bound[0]), np.max(bound[0]), n_points)
        ys = np.linspace(np.min(bound[1]), np.max(bound[1]), n_points)
        YY, XX = np.meshgrid(ys, xs)
        problem.smart_start(XX, YY, problem.cost_comp.get_aep4smart_start(), 
                          random_pct=random_pct, seed=0)
        state = problem.state
        
        os.makedirs(os.path.dirname(target_ss_file), exist_ok=True)
        with open(target_ss_file, 'wb') as f:
            pickle.dump(state, f)
    
    x_target = list(state['x'])
    y_target = list(state['y'])
    
    # Get neighbor farm layouts
    x_neighbours = []
    y_neighbours = []
    nwt_list = []
    
    for n, (rp, seed) in enumerate(zip(rated_powers, ss_seeds)):
        nwt = int(1000/rp)  # Calculate number of turbines based on rated power
        nwt_list.append(nwt)
        
        ss_file = os.path.join(output_dir, f'ss_states/ss_state_{rp}_{seed}_{n}.pkl')
        os.makedirs(os.path.dirname(ss_file), exist_ok=True)
        
        if os.path.exists(ss_file):
            with open(ss_file, 'rb') as f:
                state = pickle.load(f)
        else:
            bound = farm_list[n+1]
            wt = wt_list[rp-10]  # Map RP to turbine index
            
            x_init = (np.random.random(nwt)-0.5)*1000+np.mean(bound[0])
            y_init = (np.random.random(nwt)-0.5)*1000+np.mean(bound[1])
            
            wf_model = Nygaard_2022(site, wt)
            problem = TopFarmProblem(
                design_vars={'x': x_init, 'y': y_init},
                cost_comp=PyWakeAEPCostModelComponent(wf_model, nwt, grad_method='autograd'),
                constraints=[
                    XYBoundaryConstraint(np.asarray(bound).T, boundary_type='polygon'),
                    SpacingConstraint(wt.diameter())
                ],
                plot_comp=XYPlotComp()
            )
            
            xs = np.linspace(np.min(bound[0]), np.max(bound[0]), n_points)
            ys = np.linspace(np.min(bound[1]), np.max(bound[1]), n_points)
            YY, XX = np.meshgrid(ys, xs)
            problem.smart_start(XX, YY, problem.cost_comp.get_aep4smart_start(), 
                              random_pct=random_pct, seed=seed)
            state = problem.state
            
            with open(ss_file, 'wb') as f:
                pickle.dump(state, f)
        
        x_neighbours.append(list(state['x']))
        y_neighbours.append(list(state['y']))
    
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
        try:
            evaluate_sample(sample_no, output_dir)
        except Exception as e:
            print(f"Error processing sample {sample_no}: {str(e)}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate wind farm samples")
    parser.add_argument("--start", type=int, required=True, help="Starting sample number")
    parser.add_argument("--end", type=int, required=True, help="Ending sample number")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()
    
    evaluate_samples(args.start, args.end, args.output)
