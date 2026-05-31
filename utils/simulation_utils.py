from pams.runners.sequential import SequentialRunner
from . import data_visualization as dv
from . import personalities as pr
from . import fcla as fc
from openai import OpenAI
from groq import Groq
import pandas as pd
import os
import  sys
import ollama
import json

def classic_parameters(c_params,file_name):
    if c_params == None:
        c_params = {"iter_steps":[0,2000],"tickSize":0.05,"marketPrice":300.0,"total_agents":400,"asset_vol":1000,"cash_am":300000.0,"mean_rev_time":[20,50],"time_window":[20,40],"order_margin":[0.01,0.05]}

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
        "marketPrice": c_params["marketPrice"]
    },
    "FCNAgents": {
        "class": "FCNAgent",
        "numAgents":c_params["total_agents"],
        "markets": ["Market"],
        "assetVolume": c_params["asset_vol"],
        "cashAmount": c_params["cash_am"],
        
        
        "fundamentalWeight": {"uniform": [0.8, 1.5]},
        
        
        "chartWeight": {"uniform": [0.0, 0.25]},
        
        
        "noiseWeight": {"uniform": [0.0, 0.5]},
        
        
        "meanReversionTime": {"uniform":  c_params["mean_rev_time"]},
        
        "noiseScale": 0.001,
        
        
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
            "fundamentalPrice": c_params["fundamental_price"]
        },
        "AlgorithmicGroup": {
            "class": "FCNAgent",
            "numAgents": algo_agents,  # Maps to 395 dynamically based on your dicts
            "markets": ["Market_1"],   # FIXED: Removed the duplicate "Market" overwrite
            "assetVolume": c_params["asset_vol"],
            "cashAmount": c_params["cash_am"],
            "fundamentalWeight": {"uniform": [0.8, 1.5]},
            "chartWeight": {"uniform": [0.0, 0.25]},
            "noiseWeight": {"uniform": [0.0, 0.5]},
            "meanReversionTime": {"uniform": c_params["mean_rev_time"]},
            "noiseScale": 0.001,
            "timeWindowSize": c_params["time_window"],
            "orderMargin": c_params["order_margin"]
        },
        "LLMGroup": {
            "class": "FCLAgent",
            "numAgents": l_params["total_agents"], # Maps to 5
            "markets": ["Market_1"],
            "assetVolume": l_params["asset_vol"],
            "cashAmount": l_params["cash_am"],
            "fundamentalWeight": {"uniform": [0.8, 1.5]},
            "chartWeight": {"uniform": [0.0, 0.25]},
            "noiseWeight": {"uniform": [0.0, 0.5]},
            "meanReversionTime": {"uniform": l_params["mean_rev_time"]},
            "noiseScale": 0.001,
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
    runner, logger = run_sim(full_path, sim_type, file_name,u_provider=provider,model=model, hybrid=hybrid,api_key=used_api_key)   
    
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
