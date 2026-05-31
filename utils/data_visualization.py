
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
def market_behaviour(results_path):
    df = pd.read_csv(results_path)  
    # Prints the data to see the exported data
    print("Columnas exportadas por PAMS:", df.columns.tolist())
    # PAMS tends to saves the time 'market_time'.
    #We use the index of the table in case there's a name conflict
    tiempo = df['market_time'] if 'market_time' in df.columns else df.index
    plt.figure(figsize=(12, 6))
    plt.plot(df['market_time'], df['market_price'],
         label='Precio de Mercado Agentes FCN',
         color='#1f77b4', linewidth=1.5)
    plt.plot(df['market_time'], df['fundamental_price'],
         label='Precio Fundamental (P_F)',
         color='#d62728', linestyle='--', linewidth=1.5)
    plt.title("Simulación de Mercado Artificial (Agentes FCN)")
    plt.xlabel("Pasos de Tiempo (Ticks)")
    plt.ylabel("Precio")
    plt.legend()
    plt.grid(True)
    plt.show()

def hybrid_market_behaviour(llm_results_path, ref_price):
    df = pd.read_csv(llm_results_path)
    plt.figure(figsize=(14, 7))
    
    # 4. Graficar la evolución del precio de mercado y el precio fundamental
    plt.plot(df['market_time'], df['market_price'],
             label='Precio de Mercado (Híbrido: Algoritmos + LLM)',
             color='#1f77b4', linewidth=1.5)
             
    # Verificación de seguridad para el Precio Fundamental
    if 'fundamental_price' in df.columns:
        plt.plot(df['market_time'], df['fundamental_price'],
                 label='Precio Fundamental ($P_F$)',
                 color='#d62728', linestyle='--', linewidth=1.5)
                 
    # CORRECCIÓN: Uso dinámico del ref_price en lugar del 10.50 harcodeado
    plt.axhline(y=ref_price, color='#2ca02c', linestyle=':', linewidth=2,
                label=f'Punto de Referencia LLM ({ref_price})')
                
    plt.axvspan(0, 100, color='gray', alpha=0.2, label='Fase de Calentamiento (Sin Ejecución)')
    
    plt.title('Evolución del Precio en Mercado Artificial Híbrido (Agentes LLM con Aversión a la Pérdida)', fontsize=15)
    plt.xlabel('Tiempo de Mercado (Ticks)', fontsize=12)
    plt.ylabel('Precio del Activo', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def separate_orders(df,agent_sep):
    # 3. Separar a los agentes por su ID
    df_algo = df[df['agent_id'] <= agent_sep]
    df_llm = df[df['agent_id'] > agent_sep]
    plt.figure(figsize=(14, 7))
    # 5. Graficar las órdenes Algorítmicas (Fondo)
    plt.scatter(df_algo['market_time'], df['price'],
            color='#1f77b4', alpha=0.3, s=15,
            label='Intenciones Algorítmicas (FCN puros)')
    # 6. Graficar las órdenes del LLM (Frente)
    plt.scatter(df_llm['market_time'], df_llm['price'],
            color='#d62728', alpha=0.9, s=60, marker='X',
            label='Intenciones LLM (Sesgo Cognitivo)')
    # 7. Línea de Referencia
    plt.axhline(y=10.50, color='#2ca02c', linestyle=':', linewidth=2,
            label='Punto de Referencia LLM (10.50)')
    plt.title('Microestructura: Precios de Órdenes Colocadas (Algoritmos vs LLMs)', fontsize=15)
    plt.xlabel('Tiempo de Mercado (Ticks)', fontsize=12)
    plt.ylabel('Precio de la Orden Colocada', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

def plot_agent_actions_vs_time(cl_df,llm_df,ref_price):
    # Configurar un estilo limpio y profesional
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 7))
    # 1. Graficar las órdenes de los Agentes Clásicos (FCN) en el fondo
    plt.scatter(
    cl_df['market_time'], 
    cl_df['price'],
    color='#1f77b4', 
    alpha=0.3, 
    s=25, 
    label='Órdenes Algorítmicas (FCNAgent)',
    edgecolors='none'
    )

# 2. Graficar las órdenes de los Agentes con IA (LLM) al frente con un marcador resaltado
    plt.scatter(
    llm_df['market_time'], 
    llm_df['price'],
    color='#d62728', 
    alpha=0.9, 
    s=80, 
    marker='X', 
    label='Órdenes con IA (FCLAgent)',
    edgecolors='black',
    linewidths=0.5
)

# 3. Dibujar una línea horizontal con el Precio Fundamental o de Referencia original (ej: 10.50)
# Esto ayuda a ver si el mercado está sobrevalorado o subvalorado cuando los agentes actúan
    # Ajusta este valor según tu archivo agenticConfig.json
    plt.axhline(
    y=ref_price, 
    color="#431baf", 
    linestyle='--', 
    linewidth=2,
    label=f'Precio de Referencia ({ref_price})'
)

    # 4. Personalización de etiquetas y títulos
    plt.title('Microestructura del Mercado: Precios de Órdenes Colocadas (Clásicos vs LLMs)', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Tiempo de Mercado (Ticks / Pasos de Simulación)', fontsize=12)
    plt.ylabel('Precio de la Órden Enviada', fontsize=12)
    plt.legend(loc='lower right', fontsize=11, frameon=True, shadow=True)

    # 5. Optimizar la visualización y mostrar
    plt.tight_layout()
    plt.show()