from pams.runners.sequential import SequentialRunner
from . import data_visualization as dv
from . import personalities as pr
from . import fcla as fc
from . import apis as pi
from . import simulation_analysis as sa
import pandas as pd
import os
import json
from copy import deepcopy
from pathlib import Path

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
    
def run_sim(
    json_path,
    sim_type,
    name,
    u_provider="ollama",
    u_model="qwen3:4b",
    hybrid=False,
    api_key="",
    base_url=None,
    rag_context="",
):
    if hybrid:
        fc.FCLAgent.configure_personalities(pr.diccionario_personalidades, distribution=pr.distribucion_definida)
        fc.FCLAgent.configure_api(provider=u_provider, model=u_model, api_key=api_key, base_url=base_url)
        fc.FCLAgent.configure_rag_context(rag_context)
        print("Inicializando entorno híbrido...")
    else:
        fc.FCLAgent.configure_rag_context("")
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

    agent_type_mapping = {agent.agent_id: agent.__class__.__name__ for agent in runner.simulator.agents}

    df_orders = pd.DataFrame(logger.individual_orders)
    if not df_orders.empty:
        df_orders["agent_type"] = df_orders["agent_id"].map(agent_type_mapping)
        orders_path = f"{res_dir}/{name}_orders.csv"
        df_orders.to_csv(orders_path, index=False)
        print(f"Orders saved on '{orders_path}'")

    df_executions = pd.DataFrame(logger.individual_executions)
    if not df_executions.empty:
        df_executions["buy_agent_type"] = df_executions["buy_agent_id"].map(agent_type_mapping)
        df_executions["sell_agent_type"] = df_executions["sell_agent_id"].map(agent_type_mapping)
        executions_path = f"{res_dir}/{name}_executions.csv"
        df_executions.to_csv(executions_path, index=False)
        print(f"Executions saved on '{executions_path}'")

    df_portfolios = pd.DataFrame(logger.agent_portfolio_logs)
    if not df_portfolios.empty:
        portfolios_path = f"{res_dir}/{name}_agent_portfolios.csv"
        df_portfolios.to_csv(portfolios_path, index=False)
        print(f"Agent portfolios saved on '{portfolios_path}'")
    
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

def sim_loop(
    file_name,
    c_params,
    l_params=None,
    hybrid=False,
    provider="ollama",
    model="qwen3:4b",
    used_api_key="",
    base_url=None,
    rag_context="",
    cost_settings=None,
    api_key=None,
):
    if api_key is not None:
        used_api_key = api_key

    # Paths
    sim_type = "llms" if hybrid else "classic"
    j_dir = f"./jsons/{sim_type}"
    os.makedirs(j_dir, exist_ok=True)  # Ensures the directory exists
    full_path = os.path.join(j_dir, f"{file_name}.json")

    if hybrid:
        if l_params is None:
            raise ValueError("l_params is required when hybrid=True.")

        estimator_settings = {"expected_response_tokens": fc.FCLAgent._estimated_response_tokens}
        estimator_settings.update(cost_settings or {})
        cost_estimate = pi.estimate_llm_simulation_cost(
            c_params=c_params,
            l_params=l_params,
            provider=provider,
            model=model,
            rag_context=rag_context,
            **estimator_settings,
        )
        pi.print_llm_cost_estimate(cost_estimate)
        cost_path = os.path.join(j_dir, f"{file_name}_cost_estimate.json")
        pi.save_llm_cost_estimate(cost_estimate, cost_path)
        print(f"Hybrid LLM cost estimate saved at: {cost_path}")
    
    # Sim Parameters - CORRECCIÓN: Pasar full_path o j_dir a las funciones de setup
    # (Asegúrate de actualizar la firma de estas funciones en tu módulo 'su')
    if not hybrid:
        classic_parameters(c_params, file_name)
    else:
        agentic_parameters(c_params, l_params, file_name)

    # Correr simulación
    runner, logger = run_sim(
        full_path,
        sim_type,
        file_name,
        u_provider=provider,
        u_model=model,
        hybrid=hybrid,
        api_key=used_api_key,
        base_url=base_url,
        rag_context=rag_context,
    )   
    
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


def market_scenario_catalog(iter_steps=(200, 1000), llm_agents=5,classic_agents=400,):
    """Return ready-to-run market scenarios for classical and hybrid simulations."""
    warmup_steps, trading_steps = iter_steps

    sp500_price = 7580.06
    coffee_fnc_price = 2175000.0
    coffee_ice_price = 265.60
   
    return {
        "sp500_index_baseline": {
            "label": "S&P 500 index baseline",
            "source_note": "Anchored to the S&P 500 close on 2026-05-29.",
            "classical_file": f"classic_sp500_index_baseline",
            "hybrid_file": f"hybrid_sp500_index_baseline",
            "c_params": {
                "iter_steps": [warmup_steps, trading_steps],
                "tickSize": 0.25,
                "marketPrice": sp500_price * 0.995,
                "fundamental_price": sp500_price,
                "total_agents": classic_agents,
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
                "total_agents": classic_agents,
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
                "total_agents": classic_agents,
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
                "total_agents": classic_agents,
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


def real_market_data_config_catalog():
    """Default real-data sources used for backtest-style simulation setup.

    S&P 500 backtests use manually downloaded CSV files by default. The loader
    searches data/real_data, results/real_data, then data/real without web requests.
    For Colombian coffee/FNC data, provide a local CSV with the same fields in the returned config.
    """
    local_search_dirs = ["data/real_data", "results/real_data", "data/real"]
    return {
        "sp500_index_baseline": {
            "symbol": None,
            "source": "local",
            "file_name": "sp500_index.csv",
            "search_dirs": local_search_dirs,
            "api_key": None,
            "start": "2023-01-01",
            "end": None,
            "local_path": None,
            "force_download": False,
            "allow_download": False,
            "date_col": "Date",
            "price_col": "S&P500",
            "volume_col": None,
            "current_agent_knowledge": "2024-11-21",
            "holdout_steps": 20,
            "source_note": "Daily S&P 500 index data from a manually downloaded local CSV.",
        },
        "sp500_risk_off_shock": {
            "symbol": None,
            "source": "local",
            "file_name": "sp500_index.csv",
            "search_dirs": local_search_dirs,
            "api_key": None,
            "start": "2023-01-01",
            "end": None,
            "local_path": None,
            "force_download": False,
            "allow_download": False,
            "date_col": "Date",
            "price_col": "S&P500",
            "volume_col": None,
            "current_agent_knowledge": "2024-11-21",
            "holdout_steps": 20,
            "source_note": "Daily S&P 500 index data from a local CSV, then the scenario applies a lower fundamental value.",
        },
        "colombia_coffee_fnc_spot": {
            "symbol": None,
            "source": "local",
            "file_name": None,
            "search_dirs": local_search_dirs,
            "api_key": None,
            "start": None,
            "end": None,
            "local_path": None,
            "force_download": False,
            "allow_download": False,
            "date_col": "Date",
            "price_col": "Close",
            "volume_col": None,
            "current_agent_knowledge": None,
            "holdout_steps": 20,
            "source_note": "Set local_path to a FNC Colombian coffee CSV before running this backtest.",
        },
        "coffee_ice_export_proxy": {
            "symbol": None,
            "source": "local",
            "file_name": None,
            "search_dirs": local_search_dirs,
            "api_key": None,
            "start": None,
            "end": None,
            "local_path": None,
            "force_download": False,
            "allow_download": False,
            "date_col": "Date",
            "price_col": "Close",
            "volume_col": None,
            "current_agent_knowledge": None,
            "holdout_steps": 20,
            "source_note": "Set local_path to an ICE Coffee C or export proxy CSV before running this backtest.",
        },
    }


def market_data_cache_path(real_data_config, cache_dir="data/real"):
    """Return the expected local CSV path for a real-data config without downloading."""
    config = dict(real_data_config or {})
    local_path = config.get("local_path")
    if local_path:
        return Path(local_path)

    source = (config.get("source") or "").lower()
    if source == "local":
        file_name = config.get("file_name")
        search_dirs = config.get("search_dirs", ["data/real_data", "results/real_data", "data/real"])
        if not file_name:
            return None
        for directory in search_dirs:
            candidate = Path(directory) / file_name
            if candidate.exists():
                return candidate
        return Path(search_dirs[0]) / file_name if search_dirs else Path(file_name)

    symbol = config.get("symbol")
    if not symbol:
        return None

    cache_path = Path(cache_dir)
    if source == "stooq":
        safe_symbol = symbol.lower().replace("/", "_")
        return cache_path / f"stooq_{safe_symbol}.csv"

    if source == "yahoo":
        start = config.get("start")
        end = config.get("end")
        if start is None or end is None:
            return None
        start_ts = int(pd.Timestamp(start).timestamp())
        end_ts = int(pd.Timestamp(end).timestamp())
        safe_symbol = symbol.replace("/", "_")
        return cache_path / f"yahoo_{safe_symbol}_{start_ts}_{end_ts}.csv"

    return None


def real_market_cache_status(real_data_config, cache_dir="data/real"):
    """Check whether a backtest config can run without making a web request."""
    config = dict(real_data_config or {})
    expected_path = market_data_cache_path(config, cache_dir=cache_dir)
    status = {
        "source": config.get("source"),
        "symbol": config.get("symbol"),
        "local_path": config.get("local_path"),
        "expected_cache_path": str(expected_path) if expected_path else None,
        "path_exists": bool(expected_path and expected_path.exists()),
        "can_prepare_without_download": False,
        "message": "",
    }

    try:
        df = sa.load_or_fetch_real_data(
            local_path=config.get("local_path"),
            symbol=config.get("symbol"),
            source=config.get("source", "stooq"),
            start=config.get("start"),
            end=config.get("end"),
            file_name=config.get("file_name"),
            search_dirs=config.get("search_dirs", ["data/real_data", "results/real_data", "data/real"]),
            cache_dir=cache_dir,
            force_download=False,
            api_key=config.get("api_key"),
            allow_download=False,
        )
        status["rows"] = int(len(df))
        status["columns"] = list(df.columns)
        status["can_prepare_without_download"] = True
        status["message"] = "Valid local/cache CSV is available. No web request is needed."
    except Exception as exc:
        status["rows"] = 0
        status["columns"] = []
        status["message"] = str(exc)

    return pd.DataFrame([status])


def _frame_bound(frame):
    if frame is None or frame.empty:
        return None, None
    if "datetime" in frame.columns:
        return frame["datetime"].iloc[0], frame["datetime"].iloc[-1]
    return frame.index[0], frame.index[-1]


def load_and_split_real_market_data(real_data_config, cache_dir="data/real"):
    """Load real market data and split it into known history plus future holdout."""
    config = dict(real_data_config or {})
    local_path = config.get("local_path")
    symbol = config.get("symbol")
    source = (config.get("source") or "").lower()
    if not local_path and not symbol and not config.get("file_name"):
        raise ValueError("real_data_config needs local_path, symbol, or file_name.")

    raw_real = sa.load_or_fetch_real_data(
        local_path=local_path,
        symbol=symbol,
        source=config.get("source", "local"),
        start=config.get("start"),
        end=config.get("end"),
        file_name=config.get("file_name"),
        search_dirs=config.get("search_dirs", ["data/real_data", "results/real_data", "data/real"]),
        cache_dir=cache_dir,
        force_download=config.get("force_download", False),
        api_key=config.get("api_key"),
        allow_download=config.get("allow_download", False),
    )
    known_real, future_real = sa.split_real_data_for_backtest(
        raw_real,
        current_agent_knowledge=config.get("current_agent_knowledge"),
        holdout_steps=config.get("holdout_steps"),
        date_col=config.get("date_col"),
        price_col=config.get("price_col"),
    )
    if known_real.empty:
        raise ValueError("known_real is empty. Move current_agent_knowledge later or check date_col/price_col.")
    if future_real.empty:
        raise ValueError("future_real is empty. Move current_agent_knowledge earlier or increase the data end date.")

    known_start, known_end = _frame_bound(known_real)
    future_start, future_end = _frame_bound(future_real)
    metadata = {
        "source": config.get("source"),
        "symbol": symbol,
        "local_path": local_path,
        "file_name": config.get("file_name"),
        "search_dirs": config.get("search_dirs"),
        "date_col": config.get("date_col"),
        "price_col": config.get("price_col"),
        "volume_col": config.get("volume_col"),
        "current_agent_knowledge": config.get("current_agent_knowledge"),
        "holdout_steps": config.get("holdout_steps"),
        "n_raw": int(len(raw_real)),
        "n_known": int(len(known_real)),
        "n_future": int(len(future_real)),
        "known_start": known_start,
        "known_end": known_end,
        "future_start": future_start,
        "future_end": future_end,
    }
    return {
        "config": config,
        "raw_real": raw_real,
        "known_real": known_real,
        "future_real": future_real,
        "metadata": metadata,
    }


def estimate_tick_real_time_mapping(iter_steps, future_real):
    """Describe how simulation ticks map to the reserved real holdout window."""
    warmup_ticks, trading_ticks = [int(x) for x in iter_steps]
    future_n = max(int(len(future_real)), 0)
    comparable_gaps = max(future_n - 1, 1)
    ticks_per_real_observation = trading_ticks / comparable_gaps if future_n else None
    return {
        "warmup_ticks": warmup_ticks,
        "trading_ticks": trading_ticks,
        "total_ticks": warmup_ticks + trading_ticks,
        "holdout_observations": future_n,
        "ticks_per_real_observation": ticks_per_real_observation,
        "comparison_window": "Only trading ticks are mapped to future holdout observations; warmup ticks populate the LOB.",
    }


def calibrate_market_params_from_known_data(
    base_c_params,
    known_real,
    recent_window=30,
    ticks_per_real_observation=None,
    use_real_volatility=True,
    use_real_tick_size=False,
):
    """Calibrate simulation starting price and fundamental from known real history."""
    known = sa.prepare_price_frame(known_real, price_col="price")
    if known.empty:
        raise ValueError("known_real is empty; cannot calibrate c_params.")

    params = deepcopy(base_c_params)
    recent = known.tail(max(int(recent_window), 1))
    last_known_price = float(known["price"].iloc[-1])
    recent_mean_price = float(recent["price"].mean())

    params["marketPrice"] = last_known_price
    params["fundamental_price"] = recent_mean_price

    if use_real_tick_size:
        diffs = known["price"].diff().abs()
        diffs = diffs[(diffs > 0) & diffs.notna()]
        if not diffs.empty:
            params["tickSize"] = float(diffs.median())

    recent_vol = float(recent["log_return"].dropna().std()) if "log_return" in recent.columns else float("nan")
    if use_real_volatility and pd.notna(recent_vol) and recent_vol > 0:
        if ticks_per_real_observation and ticks_per_real_observation > 1:
            per_tick_vol = recent_vol / (ticks_per_real_observation ** 0.5)
        else:
            per_tick_vol = recent_vol
        params["noise_scale"] = float(min(max(per_tick_vol, 0.0001), 0.02))

    calibration = {
        "last_known_price": last_known_price,
        "recent_mean_price": recent_mean_price,
        "recent_window": int(len(recent)),
        "recent_log_return_volatility": recent_vol,
        "marketPrice_source": "last known real price",
        "fundamental_price_source": f"mean of last {len(recent)} known real prices",
        "noise_scale_source": "recent real volatility scaled by ticks_per_real_observation" if use_real_volatility else "scenario default",
    }
    return params, calibration


def prepare_realistic_backtest_scenario(
    scenario_key,
    iter_steps=(200, 1000),
    llm_agents=5,
    classic_agents=400,
    real_data_config=None,
    file_suffix="_backtest",
    calibrate_from_real_data=True,
    recent_window=30,
    use_real_volatility=True,
    use_real_tick_size=False,
):
    """Prepare a scenario using only pre-cut real data, reserving future rows for validation."""
    scenarios = market_scenario_catalog(iter_steps=iter_steps, llm_agents=llm_agents, classic_agents=classic_agents)
    if scenario_key not in scenarios:
        available = ", ".join(scenarios.keys())
        raise ValueError(f"Unknown scenario '{scenario_key}'. Available scenarios: {available}")

    scenario = deepcopy(scenarios[scenario_key])
    default_data_config = real_market_data_config_catalog().get(scenario_key, {})
    data_config = dict(default_data_config)
    data_config.update(real_data_config or {})
    split_bundle = load_and_split_real_market_data(data_config)
    tick_mapping = estimate_tick_real_time_mapping(iter_steps, split_bundle["future_real"])

    calibration = {}
    if calibrate_from_real_data:
        scenario["c_params"], calibration = calibrate_market_params_from_known_data(
            scenario["c_params"],
            split_bundle["known_real"],
            recent_window=recent_window,
            ticks_per_real_observation=tick_mapping["ticks_per_real_observation"],
            use_real_volatility=use_real_volatility,
            use_real_tick_size=use_real_tick_size,
        )
        scenario["l_params"]["ref_price"] = calibration["last_known_price"]

    if file_suffix:
        scenario["classical_file"] = f"{scenario['classical_file']}{file_suffix}"
        scenario["hybrid_file"] = f"{scenario['hybrid_file']}{file_suffix}"

    rag_context = pi.build_market_rag_context(
        split_bundle["raw_real"],
        current_agent_knowledge=data_config.get("current_agent_knowledge"),
        date_col=data_config.get("date_col"),
        price_col=data_config.get("price_col") or "Close",
        volume_col=data_config.get("volume_col"),
        extra_notes=data_config.get("source_note"),
    )
    summary = pd.DataFrame([{**split_bundle["metadata"], **tick_mapping, **calibration}])
    return {
        "scenario_key": scenario_key,
        "scenario": scenario,
        "c_params": scenario["c_params"],
        "l_params": scenario["l_params"],
        "classical_file": scenario["classical_file"],
        "hybrid_file": scenario["hybrid_file"],
        "raw_real": split_bundle["raw_real"],
        "known_real": split_bundle["known_real"],
        "future_real": split_bundle["future_real"],
        "real_data_config": data_config,
        "rag_context": rag_context,
        "tick_mapping": tick_mapping,
        "calibration": calibration,
        "summary": summary,
    }


def run_prepared_market_scenario(
    prepared_scenario,
    hybrid=False,
    provider="ollama",
    model="qwen3.5:4b",
    used_api_key="",
    base_url=None,
    rag_context=None,
    cost_settings=None,
    api_key=None,
):
    """Run a scenario returned by prepare_realistic_backtest_scenario."""
    scenario = prepared_scenario["scenario"]
    file_name = scenario["hybrid_file"] if hybrid else scenario["classical_file"]
    return sim_loop(
        file_name,
        scenario["c_params"],
        l_params=scenario["l_params"] if hybrid else None,
        hybrid=hybrid,
        provider=provider,
        model=model,
        used_api_key=used_api_key,
        base_url=base_url,
        rag_context=prepared_scenario.get("rag_context", "") if rag_context is None else rag_context,
        cost_settings=cost_settings,
        api_key=api_key,
    )


def simulated_trading_window(simulated_market_data, warmup_ticks=0):
    """Return only the post-warmup simulated rows used for real holdout comparison."""
    df = pd.DataFrame(simulated_market_data).copy()
    if df.empty or "market_time" not in df.columns:
        return df
    market_time = pd.to_numeric(df["market_time"], errors="coerce")
    return df[market_time >= int(warmup_ticks)].reset_index(drop=True)


def run_market_scenario(
    scenario_key,
    hybrid=False,
    provider="ollama",
    model="qwen3.5:4b",
    used_api_key="",
    iter_steps=(200, 1000),
    llm_agents=5,
    classic_agents=400,
    base_url=None,
    rag_context="",
    cost_settings=None,
    api_key=None,
):
    if api_key is not None:
        used_api_key = api_key

    scenarios = market_scenario_catalog(iter_steps=iter_steps, llm_agents=llm_agents, classic_agents=classic_agents)
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
        base_url=base_url,
        rag_context=rag_context,
        cost_settings=cost_settings,
    )


def run_market_scenarios(
    scenario_keys,
    hybrid=False,
    provider="ollama",
    model="qwen3.5:4b",
    used_api_key="",
    iter_steps=(200, 1000),
    llm_agents=5,
    classic_agents=400,
    base_url=None,
    rag_context="",
    cost_settings=None,
    api_key=None,
):
    if api_key is not None:
        used_api_key = api_key

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
            classic_agents=classic_agents,
            base_url=base_url,
            rag_context=rag_context,
            cost_settings=cost_settings,
        )
    return results
