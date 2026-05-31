from pams.runners.sequential import SequentialRunner
from . import data_visualization as dv
from . import personalities as pr
from . import fcla as fc
from openai import OpenAI
from groq import Groq
import pandas as pd
import os
import json

def classic_parameters(c_params,file_name):
    if c_params == None:
        c_params = {"iter_steps":[0,2000],"tickSize":0.05,"marketPrice":300.0,"total_agents":400,"asset_vol":1000,"cash_am":300000.0,"mean_rev_time":[20,50],"time_window":[20,40],"order_margin":[0.01,0.05],"fundamental_price":300.0}

    classic_json = {
    "simulation": {
        "markets": ["Market"],
        "agents": ["FCNAgents"],
        "sessions": [
            { 
                "sessionName": 0,
                "iterationSteps": c_params["iter_steps"][0],  
                "withOrderPlacement": True,
                "withOrderExecution": False,
                "withPrint": True,
                "hiFrequencySubmitRate": 1.0
            },
            { 
                "sessionName": 1,
                "iterationSteps":c_params["iter_steps"][1],
                "withOrderPlacement": True,
                "withOrderExecution": True,
                "withPrint": False 
            }
        ]
    },
    "Market": {
        "class": "Market",
        "tickSize": c_params["tickSize"],  
        "marketPrice": c_params["marketPrice"],
        "fundamentalPrice": c_params.get("fundamental_price", c_params["marketPrice"])
    },
    "FCNAgents": {
        "class": "FCNAgent",
        "numAgents":c_params["total_agents"],
        "markets": ["Market"],
        "assetVolume": c_params["asset_vol"],
        "cashAmount": c_params["cash_am"],
        
        
        "fundamentalWeight": {"uniform": c_params.get("fundamental_weight", [0.8, 1.5])},
        
        
        "chartWeight": {"uniform": c_params.get("chart_weight", [0.0, 0.25])},
        
        
        "noiseWeight": {"uniform": c_params.get("noise_weight", [0.0, 0.5])},
        
        
        "meanReversionTime": {"uniform":  c_params["mean_rev_time"]},
        
        "noiseScale": c_params.get("noise_scale", 0.001),
        
        
        "timeWindowSize": c_params["time_window"],
        
        
        "orderMargin": c_params["order_margin"]
        }
    }
    json_string = json.dumps(classic_json, indent = 4)
    print(json_string)
    file_path = "./jsons/classic/"+file_name
    with open(file_path+".json","w") as file:
        json.dump(classic_json,file, indent=4)


def agentic_parameters(c_params, l_params, file_name):
    # Calculate regular agents so total matches (400 total - 5 LLMs = 395 FCNAgents)
    algo_agents = c_params["total_agents"] - l_params["total_agents"]
    
    agentic_json = {
        "simulation": {
            "markets": ["Market_1"],
            "agents": ["AlgorithmicGroup", "LLMGroup"],
            "sessions": [
                {
                    "sessionName": "Session_0",
                    "iterationSteps": c_params["iter_steps"][0],
                    "withOrderPlacement": True,
                    "withOrderExecution": False,
                    "withPrint": True,
                    "hiFrequencySubmitRate": 1.0
                },
                {
                    "sessionName": "Session_1",
                    "iterationSteps": c_params["iter_steps"][1],
                    "withOrderPlacement": True,
                    "withOrderExecution": True,
                    "withPrint": False
                }
            ]
        },
        "Market_1": {
            "class": "Market",
            "tickSize": c_params["tickSize"],
            "marketPrice": c_params["marketPrice"],
            "fundamentalPrice": c_params.get("fundamental_price", c_params["marketPrice"])
        },
        "AlgorithmicGroup": {
            "class": "FCNAgent",
            "numAgents": algo_agents,  # Maps to 395 dynamically based on your dicts
            "markets": ["Market_1"],   # FIXED: Removed the duplicate "Market" overwrite
            "assetVolume": c_params["asset_vol"],
            "cashAmount": c_params["cash_am"],
            "fundamentalWeight": {"uniform": c_params.get("fundamental_weight", [0.8, 1.5])},
            "chartWeight": {"uniform": c_params.get("chart_weight", [0.0, 0.25])},
            "noiseWeight": {"uniform": c_params.get("noise_weight", [0.0, 0.5])},
            "meanReversionTime": {"uniform": c_params["mean_rev_time"]},
            "noiseScale": c_params.get("noise_scale", 0.001),
            "timeWindowSize": c_params["time_window"],
            "orderMargin": c_params["order_margin"]
        },
        "LLMGroup": {
            "class": "FCLAgent",
            "numAgents": l_params["total_agents"], # Maps to 5
            "markets": ["Market_1"],
            "assetVolume": l_params["asset_vol"],
            "cashAmount": l_params["cash_am"],
            "fundamentalWeight": {"uniform": l_params.get("fundamental_weight", c_params.get("fundamental_weight", [0.8, 1.5]))},
            "chartWeight": {"uniform": l_params.get("chart_weight", c_params.get("chart_weight", [0.0, 0.25]))},
            "noiseWeight": {"uniform": l_params.get("noise_weight", c_params.get("noise_weight", [0.0, 0.5]))},
            "meanReversionTime": {"uniform": l_params["mean_rev_time"]},
            "noiseScale": l_params.get("noise_scale", c_params.get("noise_scale", 0.001)),
            "timeWindowSize": l_params["time_window"],
            "orderMargin": l_params["order_margin"],
            "referencePrice": l_params["ref_price"]
        }
    }
    
    # Safely save the JSON
    dir_path = "./jsons/llms/"
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, f"{file_name}.json")
    
    with open(file_path, "w") as file:
        json.dump(agentic_json, file, indent=4)
        
    print(f"Hybrid JSON configuration generated at: {file_path}")
    
def run_sim(json_path, sim_type, name, u_provider="ollama", u_model="qwen3:4b", hybrid=False,api_key = ""):
    if hybrid:
        fc.FCLAgent.configure_personalities(pr.diccionario_personalidades, distribution=pr.distribucion_definida)
        fc.FCLAgent.configure_api(provider=u_provider, model=u_model,api_key=api_key)
        print("Inicializando entorno híbrido...")
    else:
        print("Inicializando entorno clásico...")
        
    logger = fc.CustomOrderLogger()
    runner = SequentialRunner(settings=json_path, logger=logger)
    
    # Register the custom agent (Safe to do in both, but essential for hybrid)
    runner.class_register(fc.FCLAgent) 
    
    print(f"Starting {sim_type.capitalize()} Simulation...")
    runner.main()
    print("Simulation complete successfully.")
    
    # Extract and save data, ensuring the results directory actually exists
    res_dir = f"./results/{sim_type}"
    os.makedirs(res_dir, exist_ok=True)
    res_path = f"{res_dir}/{name}_base_results_fcn.csv"
    
    df_precios = pd.DataFrame(logger.market_step_logs)
    df_precios.to_csv(res_path, index=False)
    print(f"Data saved on '{res_path}'")
    
    return runner, logger
def statistical_info(df):
    print("\n" + "="*40)
    print("📊 RESUMEN ESTADÍSTICO DE LA SIMULACIÓN")
    print("="*40)
    # Filtramos solo la Sesión 1 (donde hubo trading real)
    df_real = df[df['market_time'] > 100]

    print(f"▶ Precio Inicial:        {df_real['market_price'].iloc[0]:.4f}")
    print(f"▶ Precio Final:          {df_real['market_price'].iloc[-1]:.4f}")
    print(f"▶ Precio Máximo (Pico):  {df_real['market_price'].max():.4f}")
    print(f"▶ Precio Mínimo (Caída): {df_real['market_price'].min():.4f}")
    print(f"▶ Volatilidad (Desv. E): {df_real['market_price'].std():.4f}")
    print("="*40)

def agents_order_separation(runner,logger):
    #Crear un mapa dinámico de ID de Agente -> Tipo de Agente
    agent_type_mapping = {}
    for agent in runner.simulator.agents:
        # Mapea, por ejemplo: {0: 'FCNAgent', 95: 'FCLAgent'}
        agent_type_mapping[agent.agent_id] = agent.__class__.__name__
    #Convertir las órdenes individuales atrapadas en un DataFrame de Pandas
    #Agregar una columna identificadora usando nuestro mapa dinámico
    df_orders = pd.DataFrame(logger.individual_orders)
    if df_orders.empty:
        columns = ["market_time", "agent_id", "is_buy", "price", "volume", "kind", "agent_type"]
        print("\n=== ESTADISTICAS DE ORDENES colocadas ===")
        print("No se registraron ordenes individuales en esta simulacion.")
        empty_df = pd.DataFrame(columns=columns)
        return empty_df.copy(), empty_df.copy()
    df_orders['agent_type'] = df_orders['agent_id'].map(agent_type_mapping)
    #Separar los DataFrames en agentes clásicos y agentes LLM
    df_classical_orders = df_orders[df_orders['agent_type'] == 'FCNAgent']
    df_llm_orders = df_orders[df_orders['agent_type'] == 'FCLAgent']
    print(f"\n=== ESTADÍSTICAS DE ÓRDENES colocadas ===")
    print(f"Órdenes totales de Agentes Clásicos (FCN): {len(df_classical_orders)}")
    print(f"Órdenes totales de Agentes con IA (LLM):  {len(df_llm_orders)}")
    print("\n--- MUESTRA DE ÓRDENES CLÁSICAS (FCNAgent) ---")
    print(df_classical_orders.head(10))
    print("\n--- MUESTRA DE ÓRDENES DE IA (FCLAgent) ---")
    print(df_llm_orders.head(10))
    return df_llm_orders,df_classical_orders

def sim_loop(file_name,c_params, l_params=None, hybrid=False,provider = "ollama",model="qwen3:4b" ,used_api_key=""):
    # Paths
    sim_type = "llms" if hybrid else "classic"
    j_dir = f"./jsons/{sim_type}"
    os.makedirs(j_dir, exist_ok=True)  # Ensures the directory exists
    full_path = os.path.join(j_dir, f"{file_name}.json")
    
    # Sim Parameters - CORRECCIÓN: Pasar full_path o j_dir a las funciones de setup
    # (Asegúrate de actualizar la firma de estas funciones en tu módulo 'su')
    if not hybrid:
        classic_parameters(c_params, file_name)
    else:
        agentic_parameters(c_params, l_params, file_name)

    # Correr simulación
    runner, logger = run_sim(full_path, sim_type, file_name,u_provider=provider,u_model=model, hybrid=hybrid,api_key=used_api_key)   
    
    # Plots
    llm_df, cls_df = agents_order_separation(runner, logger)
    
    # Determine the correct reference price
    ref_price = l_params["ref_price"] if hybrid and l_params else c_params["marketPrice"]
    dv.plot_agent_actions_vs_time(cls_df, llm_df, ref_price) 
    
    # Market behavior 
    r_path = f"./results/{sim_type}/{file_name}_base_results_fcn.csv"
    if not hybrid:
        dv.market_behaviour(r_path)
    else:  
        # CORRECCIÓN: Pasar el ref_price a la función de visualización
        dv.hybrid_market_behaviour(r_path, ref_price)

    return runner, logger


def market_scenario_catalog(iter_steps=(200, 1000), llm_agents=5):
    """Return ready-to-run market scenarios for classical and hybrid simulations."""
    warmup_steps, trading_steps = iter_steps

    sp500_price = 7580.06
    coffee_fnc_price = 2175000.0
    coffee_ice_price = 265.60

    return {
        "sp500_index_baseline": {
            "label": "S&P 500 index baseline",
            "source_note": "Anchored to the S&P 500 close on 2026-05-29.",
            "classical_file": "classic_sp500_index_baseline",
            "hybrid_file": "hybrid_sp500_index_baseline",
            "c_params": {
                "iter_steps": [warmup_steps, trading_steps],
                "tickSize": 0.25,
                "marketPrice": sp500_price * 0.995,
                "fundamental_price": sp500_price,
                "total_agents": 500,
                "asset_vol": 80,
                "cash_am": 900000.0,
                "mean_rev_time": [40, 120],
                "time_window": [20, 90],
                "order_margin": [0.0005, 0.004],
                "fundamental_weight": [0.9, 1.7],
                "chart_weight": [0.0, 0.35],
                "noise_weight": [0.0, 0.45],
                "noise_scale": 0.0008,
            },
            "l_params": {
                "total_agents": llm_agents,
                "asset_vol": 80,
                "cash_am": 900000.0,
                "mean_rev_time": [40, 120],
                "time_window": [20, 90],
                "order_margin": [0.0005, 0.004],
                "ref_price": sp500_price,
                "fundamental_weight": [0.9, 1.7],
                "chart_weight": [0.0, 0.35],
                "noise_weight": [0.0, 0.45],
                "noise_scale": 0.0008,
            },
        },
        "sp500_risk_off_shock": {
            "label": "S&P 500 risk-off shock",
            "source_note": "Same S&P 500 anchor, with a lower fundamental and wider order margins.",
            "classical_file": "classic_sp500_risk_off_shock",
            "hybrid_file": "hybrid_sp500_risk_off_shock",
            "c_params": {
                "iter_steps": [warmup_steps, trading_steps],
                "tickSize": 0.25,
                "marketPrice": sp500_price,
                "fundamental_price": sp500_price * 0.94,
                "total_agents": 500,
                "asset_vol": 80,
                "cash_am": 900000.0,
                "mean_rev_time": [15, 55],
                "time_window": [10, 45],
                "order_margin": [0.001, 0.012],
                "fundamental_weight": [1.0, 2.2],
                "chart_weight": [0.1, 0.65],
                "noise_weight": [0.1, 0.8],
                "noise_scale": 0.0018,
            },
            "l_params": {
                "total_agents": llm_agents,
                "asset_vol": 80,
                "cash_am": 900000.0,
                "mean_rev_time": [15, 55],
                "time_window": [10, 45],
                "order_margin": [0.001, 0.012],
                "ref_price": sp500_price,
                "fundamental_weight": [1.0, 2.2],
                "chart_weight": [0.1, 0.65],
                "noise_weight": [0.1, 0.8],
                "noise_scale": 0.0018,
            },
        },
        "colombia_coffee_fnc_spot": {
            "label": "Colombia coffee internal reference price",
            "source_note": "Anchored to the FNC internal reference price per 125 kg load on 2026-05-29.",
            "classical_file": "classic_colombia_coffee_fnc_spot",
            "hybrid_file": "hybrid_colombia_coffee_fnc_spot",
            "c_params": {
                "iter_steps": [warmup_steps, trading_steps],
                "tickSize": 500.0,
                "marketPrice": coffee_fnc_price * 0.99,
                "fundamental_price": coffee_fnc_price,
                "total_agents": 450,
                "asset_vol": 15,
                "cash_am": 45000000.0,
                "mean_rev_time": [25, 85],
                "time_window": [15, 60],
                "order_margin": [0.001, 0.010],
                "fundamental_weight": [1.0, 2.0],
                "chart_weight": [0.0, 0.45],
                "noise_weight": [0.05, 0.75],
                "noise_scale": 0.0015,
            },
            "l_params": {
                "total_agents": llm_agents,
                "asset_vol": 15,
                "cash_am": 45000000.0,
                "mean_rev_time": [25, 85],
                "time_window": [15, 60],
                "order_margin": [0.001, 0.010],
                "ref_price": coffee_fnc_price,
                "fundamental_weight": [1.0, 2.0],
                "chart_weight": [0.0, 0.45],
                "noise_weight": [0.05, 0.75],
                "noise_scale": 0.0015,
            },
        },
        "coffee_ice_export_proxy": {
            "label": "Arabica Coffee C export proxy",
            "source_note": "Anchored to the ICE Coffee C quote reported by FNC on 2026-05-29.",
            "classical_file": "classic_coffee_ice_export_proxy",
            "hybrid_file": "hybrid_coffee_ice_export_proxy",
            "c_params": {
                "iter_steps": [warmup_steps, trading_steps],
                "tickSize": 0.05,
                "marketPrice": coffee_ice_price * 0.99,
                "fundamental_price": coffee_ice_price,
                "total_agents": 450,
                "asset_vol": 1000,
                "cash_am": 300000.0,
                "mean_rev_time": [25, 85],
                "time_window": [15, 60],
                "order_margin": [0.001, 0.010],
                "fundamental_weight": [1.0, 2.0],
                "chart_weight": [0.0, 0.45],
                "noise_weight": [0.05, 0.75],
                "noise_scale": 0.0015,
            },
            "l_params": {
                "total_agents": llm_agents,
                "asset_vol": 1000,
                "cash_am": 300000.0,
                "mean_rev_time": [25, 85],
                "time_window": [15, 60],
                "order_margin": [0.001, 0.010],
                "ref_price": coffee_ice_price,
                "fundamental_weight": [1.0, 2.0],
                "chart_weight": [0.0, 0.45],
                "noise_weight": [0.05, 0.75],
                "noise_scale": 0.0015,
            },
        },
    }


def run_market_scenario(scenario_key, hybrid=False, provider="ollama", model="qwen3.5:4b", used_api_key="", iter_steps=(200, 1000), llm_agents=5):
    scenarios = market_scenario_catalog(iter_steps=iter_steps, llm_agents=llm_agents)
    if scenario_key not in scenarios:
        available = ", ".join(scenarios.keys())
        raise ValueError(f"Unknown scenario '{scenario_key}'. Available scenarios: {available}")

    scenario = scenarios[scenario_key]
    file_name = scenario["hybrid_file"] if hybrid else scenario["classical_file"]
    return sim_loop(
        file_name,
        scenario["c_params"],
        l_params=scenario["l_params"] if hybrid else None,
        hybrid=hybrid,
        provider=provider,
        model=model,
        used_api_key=used_api_key,
    )


def run_market_scenarios(scenario_keys, hybrid=False, provider="ollama", model="qwen3.5:4b", used_api_key="", iter_steps=(200, 1000), llm_agents=5):
    results = {}
    for scenario_key in scenario_keys:
        print(f"\nRunning scenario: {scenario_key} | hybrid={hybrid}")
        results[scenario_key] = run_market_scenario(
            scenario_key,
            hybrid=hybrid,
            provider=provider,
            model=model,
            used_api_key=used_api_key,
            iter_steps=iter_steps,
            llm_agents=llm_agents,
        )
    return results
