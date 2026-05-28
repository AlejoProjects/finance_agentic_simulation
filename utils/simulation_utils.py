from pams.runners.sequential import SequentialRunner
from pams.logs.market_step_loggers import MarketStepSaver
from . import fcla as fc
from . import personalities as pr
import matplotlib.pyplot as plt
from openai import OpenAI

import ollama
import pandas as pd

def run_sim(json_path,name,u_provider="ollama",u_model="qwen3:4b",hybrid=False):
    if hybrid == True:
        fc.FCLAgent.configure_personalities(pr.diccionario_personalidades, distribution=pr.distribucion_definida)
        fc.FCLAgent.configure_api(provider=u_provider, model=u_model)
        print("Inicializando entorno híbrido...")

    else:
        print("Inicializando entorno clásico")
    logger = fc.CustomOrderLogger()

    # 3. Configuramos y corremos el Runner de PAMS pasándole nuestro logger
    
    #logger = MarketStepSaver()
    #runner = SequentialRunner(settings=json_path, logger=logger)
    runner = SequentialRunner(settings=json_path, logger=logger)
    runner.class_register(fc.FCLAgent)
    print("Starting Hybrid Simulation (FCNAgents + FCLAgents)...")
    runner.main()
    print("Simulation complete successfully.")
    # 4. Extracts and saves the data on a new Data Frame
    datos_log = logger.market_step_logs
    df_precios = pd.DataFrame(datos_log)
    df_precios.to_csv(f"./results/{name}_base_results_fcn.csv", index=False)
    print(f"Data saved on './results/{name}_base_results_fcn.csv'")
    return runner,logger
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
def test_llm(used_model: str, service: str, api_key=""):
    prompt = """Eres un inversor en un mercado continuo de doble subasta. Tu precio de compra de referencia fue 800.El precio actual en el mercado es 1000.Teniendo en cuenta tu sesgo de aversión a la pérdida, ¿deseas COMPRAR, VENDER o MANTENER?Responde estrictamente con una de estas palabras: BUY, SELL, HOLD.""".strip()
    if service == 'ollama':
        print(f"[Ollama] Procesando tu consulta con el modelo '{used_model}'...")
    if service == 'ollama':
        print(f"[Ollama] Verificando/Descargando el modelo '{used_model}'...")
        
        # Activamos stream=True para ver el progreso real en la consola y saber qué ocurre
        last_status = ""
        for chunk in ollama.pull(used_model, stream=True):
            status = chunk.get('status', '')
            if status != last_status:
                print(f" -> Status: {status}")
                last_status = status
        
        print("[Ollama] Procesando tu consulta...")
        response = ollama.chat(
            model=used_model, 
            messages=[{
                'role': 'user',
                'content': prompt,
            }],
            options={
                'num_ctx': 512
            }
        )
        
        # Sintaxis corregida para objetos ChatResponse: response.message.content
        answer = response.message.content
        print(f"[Ollama RESPUESTA]: {answer}")
        return answer
        
    elif service == 'nvidia':
       
        client = OpenAI(
             base_url = "https://integrate.api.nvidia.com/v1",
             api_key = api_key
            )

        completion = client.chat.completions.create(
        model=used_model,
        messages=[{"role":"user","content":prompt}],
        temperature=1,
        top_p=0.95,
        max_tokens=8192,
        stream=True
        )

        for chunk in completion:
            if not getattr(chunk, "choices", None):
                continue
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="")
        return answer
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