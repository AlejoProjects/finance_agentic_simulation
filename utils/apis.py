from __future__ import annotations

from typing import Any

import pandas as pd

#Only for colab
#import requests
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    import ollama
except ImportError:
    ollama = None

import json
import os
import threading
import subprocess
import time

ALIBABA_DEFAULT_BASE_URL = os.getenv(
    "ALIBABA_BASE_URL",
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)
ALIBABA_CHINA_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


PROVIDER_ALIASES = {
    "ali": "alibaba",
    "aliyun": "alibaba",
    "alibaba_cloud": "alibaba",
    "dashscope": "alibaba",
    "qwen": "alibaba",
    "qwen_cloud": "alibaba",
    "ollama_local": "ollama",
    "local_model": "local",
}


DEFAULT_LLM_PRICING_USD_PER_1M = {
    "ollama": {
        "default": {"input": 0.0, "output": 0.0},
    },
    "local": {
        "default": {"input": 0.0, "output": 0.0},
    },
    "groq": {
        # Edit these values when Groq updates pricing for your selected model.
        "default": {"input": 0.59, "output": 0.79},
        "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
        "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
        "openai/gpt-oss-120b": {"input": 0.15, "output": 0.60},
    },
    "nvidia": {
        # NVIDIA endpoints vary by catalog/model/contract. Override when you know your rate.
        "default": {"input": 0.0, "output": 0.0},
        "nvidia/nemotron-3-super-120b-a12b": {"input": 0.20, "output": 0.80},
    },
    "alibaba": {
        # Alibaba Model Studio pricing is model and context-length dependent. Override if needed.
        "default": {"input": 0.40, "output": 1.20},
        "qwen-plus": {"input": 0.40, "output": 1.20},
        "qwen-plus-us": {"input": 0.40, "output": 1.20},
        "qwen3.5-plus": {"input": 0.40, "output": 2.40},
        "qwen3.5-flash": {"input": 0.10, "output": 0.40},
        "qwen-max": {"input": 0.40, "output": 1.20},
        "qwen3-coder-next": {"input": 0.30, "output": 1.50},
    },
}


def normalize_provider_name(provider: str) -> str:
    """This function resolves a provider alias to its canonical name.

    Params:
        provider: LLM provider name.
    """
    provider_key = (provider or "").strip().lower()
    return PROVIDER_ALIASES.get(provider_key, provider_key)


def rough_token_count(text: str, chars_per_token: float = 4.0) -> int:
    """This function estimates token usage from text length.

    Params:
        text: Input text.
        chars_per_token: Estimated characters per token.
    """
    if not text:
        return 0
    return max(1, int(round(len(str(text)) / chars_per_token)))


def fcl_prompt_cost_proxy(
    reference_price: float=300.0,
    current_price: float=300.0,
    current_time: int=0,
    all_time_high: float | None=None,
    all_time_low: float | None=None,
    rag_context: str | None="",
) -> str:
    """This function builds a representative FCL prompt for cost estimation.

    Params:
        reference_price: Agent reference price.
        current_price: Current market price.
        current_time: Current decision time.
        all_time_high: Observed maximum price.
        all_time_low: Observed minimum price.
        rag_context: Optional historical context.
    """
    all_time_high = current_price if all_time_high is None else all_time_high
    all_time_low = current_price if all_time_low is None else all_time_low
    high_nearness = current_price / all_time_high if all_time_high else 0.0
    low_nearness = current_price / all_time_low if all_time_low else 0.0
    rag_block = (
        "\nContexto historico recuperado antes de la fecha de conocimiento del agente:\n"
        f"{rag_context}\n"
        if rag_context else ""
    )
    return f"""
    Eres un inversor racional y equilibrado en el mercado.

    Contexto Macroeconomico del Mercado:
    - Tu precio de compra de referencia historico es: {reference_price}
    - El precio de cotizacion actual del mercado es: {current_price}
    - Tiempo actual de mercado en ticks: {current_time}
    - Precio maximo historico observado por el agente: {all_time_high}
    - Precio minimo historico observado por el agente: {all_time_low}
    - Cercania al maximo historico p_t / p^h_1:t: {high_nearness:.6f}
    - Relacion contra minimo historico p_t / p^l_1:t: {low_nearness:.6f}

    Estado del portafolio:
    - Cash: 300000.0000
    - Asset volume: 1000
    - Asset proportion PA_t: 0.5000
    - Recent trading history: No executions recorded yet
    {rag_block}

    Evaluando tus sesgos cognitivos, tu perfil de riesgo asignado y las condiciones actuales, determina tu accion.
    Responde estrictamente con una unica palabra: BUY, SELL o HOLD.
    """.strip()


def get_llm_token_price(provider: str, model: str | None = None, pricing_table: dict[str, Any] | None=None) -> dict[str, float]:
    """This function returns input and output token prices for a model.

    Params:
        provider: LLM provider name.
        model: Model name or loaded model.
        pricing_table: Optional model pricing table.
    """
    pricing_table = pricing_table or DEFAULT_LLM_PRICING_USD_PER_1M
    provider_key = normalize_provider_name(provider)
    provider_prices = pricing_table.get(provider_key, {})
    if model and model in provider_prices:
        return provider_prices[model]
    if model:
        lower_model = model.lower()
        for known_model, prices in provider_prices.items():
            if known_model != "default" and known_model.lower() in lower_model:
                return prices
    return provider_prices.get("default", {"input": 0.0, "output": 0.0})


def estimate_llm_call_count(c_params: dict[str, Any], l_params: dict[str, Any] | None, order_submission_probability: float=1.0) -> int:
    """This function estimates LLM calls for a hybrid simulation.

    Params:
        c_params: Classical simulation parameters.
        l_params: LLM-agent parameters.
        order_submission_probability: Expected order-submission probability.
    """
    if not c_params or not l_params:
        return 0
    iter_steps = c_params.get("iter_steps", [0, 0])
    total_steps = int(sum(iter_steps))
    total_agents = max(int(c_params.get("total_agents", 1)), 1)
    llm_agents = max(int(l_params.get("total_agents", 0)), 0)
    llm_selection_probability = min(llm_agents / total_agents, 1.0)
    return int(round(total_steps * llm_selection_probability * order_submission_probability))


def estimate_llm_simulation_cost(
    c_params: dict[str, Any],
    l_params: dict[str, Any],
    provider: str="ollama",
    model: str | None=None,
    rag_context: str="",
    expected_response_tokens: int=3,
    order_submission_probability: float=1.0,
    prompt_tokens: int | None=None,
    pricing_override: dict[str, float] | None=None,
    pricing_table: dict[str, Any] | None=None,
) -> dict[str, Any]:
    """This function estimates token usage and API cost before a simulation.

    Params:
        c_params: Classical simulation parameters.
        l_params: LLM-agent parameters.
        provider: LLM provider name.
        model: Model name or loaded model.
        rag_context: Optional historical context.
        expected_response_tokens: Expected tokens per response.
        order_submission_probability: Expected order-submission probability.
        prompt_tokens: Optional prompt-token estimate.
        pricing_override: Optional input and output token prices.
        pricing_table: Optional model pricing table.
    """
    if not l_params:
        raise ValueError("l_params is required for hybrid cost estimation.")

    reference_price = l_params.get("ref_price", c_params.get("fundamental_price", c_params.get("marketPrice", 300.0)))
    current_price = c_params.get("marketPrice", reference_price)
    prompt_proxy = fcl_prompt_cost_proxy(
        reference_price=reference_price,
        current_price=current_price,
        all_time_high=max(reference_price, current_price),
        all_time_low=min(reference_price, current_price),
        rag_context=rag_context,
    )
    estimated_prompt_tokens = prompt_tokens or rough_token_count(prompt_proxy)
    estimated_calls = estimate_llm_call_count(
        c_params=c_params,
        l_params=l_params,
        order_submission_probability=order_submission_probability,
    )
    prices = pricing_override or get_llm_token_price(provider, model=model, pricing_table=pricing_table)
    input_rate = float(prices.get("input", 0.0))
    output_rate = float(prices.get("output", 0.0))
    input_cost = estimated_calls * estimated_prompt_tokens * input_rate / 1_000_000
    output_cost = estimated_calls * expected_response_tokens * output_rate / 1_000_000
    total_cost = input_cost + output_cost

    return {
        "provider": normalize_provider_name(provider),
        "model": model or "default",
        "total_simulation_steps": int(sum(c_params.get("iter_steps", [0, 0]))),
        "total_agents": int(c_params.get("total_agents", 0)),
        "llm_agents": int(l_params.get("total_agents", 0)),
        "estimated_llm_calls": estimated_calls,
        "order_submission_probability": float(order_submission_probability),
        "estimated_prompt_tokens_per_call": int(estimated_prompt_tokens),
        "expected_response_tokens_per_call": int(expected_response_tokens),
        "estimated_input_tokens": int(estimated_calls * estimated_prompt_tokens),
        "estimated_output_tokens": int(estimated_calls * expected_response_tokens),
        "input_usd_per_1m_tokens": input_rate,
        "output_usd_per_1m_tokens": output_rate,
        "estimated_input_cost_usd": float(input_cost),
        "estimated_output_cost_usd": float(output_cost),
        "estimated_total_cost_usd": float(total_cost),
        "pricing_note": (
            "This is a pre-flight estimate. Exact billing depends on the provider tokenizer, "
            "model-specific pricing, retries, cache policy, and the actual number of FCLAgent LLM calls."
        ),
    }


def print_llm_cost_estimate(cost_estimate: dict[str, Any]) -> None:
    """This function prints a formatted LLM cost estimate.

    Params:
        cost_estimate: Calculated cost estimate.
    """
    print("\n=== ESTIMATED HYBRID LLM COST ===")
    print(f"Provider / model: {cost_estimate['provider']} / {cost_estimate['model']}")
    print(f"Simulation steps: {cost_estimate['total_simulation_steps']}")
    print(f"LLM agents: {cost_estimate['llm_agents']} / {cost_estimate['total_agents']}")
    print(f"Estimated LLM calls: {cost_estimate['estimated_llm_calls']}")
    print(f"Prompt tokens per call: {cost_estimate['estimated_prompt_tokens_per_call']}")
    print(f"Response tokens per call: {cost_estimate['expected_response_tokens_per_call']}")
    print(f"Input tokens: {cost_estimate['estimated_input_tokens']}")
    print(f"Output tokens: {cost_estimate['estimated_output_tokens']}")
    print(
        "Rates USD / 1M tokens: "
        f"input={cost_estimate['input_usd_per_1m_tokens']}, "
        f"output={cost_estimate['output_usd_per_1m_tokens']}"
    )
    print(f"Estimated total: ${cost_estimate['estimated_total_cost_usd']:.6f}")
    print("=================================\n")


def save_llm_cost_estimate(cost_estimate: dict[str, Any], file_path: str | os.PathLike[str]) -> str | os.PathLike[str]:
    """This function saves an LLM cost estimate as JSON.

    Params:
        cost_estimate: Calculated cost estimate.
        file_path: Target file path.
    """
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(file_path, "w") as file:
        json.dump(cost_estimate, file, indent=4)
    return file_path


def _coerce_market_history_df(market_history: Any) -> pd.DataFrame:
    """This function converts market history into a DataFrame.

    Params:
        market_history: Historical market observations.
    """
    import pandas as pd

    if market_history is None:
        return pd.DataFrame()
    if isinstance(market_history, pd.DataFrame):
        return market_history.copy()
    if isinstance(market_history, str):
        return pd.read_csv(market_history)
    return pd.DataFrame(market_history)


def split_known_future_market_data(
    market_history: Any,
    current_agent_knowledge: str | pd.Timestamp | None=None,
    holdout_steps: int | None=None,
    date_col: str | None="date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """This function separates known market history from its future holdout.

    Params:
        market_history: Historical market observations.
        current_agent_knowledge: Latest date known by the agent.
        holdout_steps: Reserved future observations.
        date_col: Date column name.
    """
    df = _coerce_market_history_df(market_history)
    if df.empty:
        return df.copy(), df.copy()

    if date_col in df.columns:
        df[date_col] = __import__("pandas").to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    if current_agent_knowledge is None:
        cut_idx = max(1, int(len(df) * 0.8))
    elif isinstance(current_agent_knowledge, int):
        cut_idx = max(1, min(len(df), current_agent_knowledge))
    elif date_col in df.columns:
        knowledge_date = __import__("pandas").to_datetime(current_agent_knowledge)
        cut_idx = int((df[date_col] <= knowledge_date).sum())
        cut_idx = max(1, min(len(df), cut_idx))
    else:
        raise ValueError("current_agent_knowledge must be an integer when no date_col exists.")

    known_df = df.iloc[:cut_idx].copy()
    future_df = df.iloc[cut_idx:].copy()
    if holdout_steps is not None:
        future_df = future_df.iloc[:holdout_steps].copy()
    return known_df, future_df


def build_market_rag_context(
    market_history: Any,
    current_agent_knowledge: str | pd.Timestamp | None=None,
    current_datetime: str | pd.Timestamp | None=None,
    date_col: str | None="date",
    price_col: str | None="close",
    volume_col: str | None=None,
    lookback_rows: int=30,
    max_context_rows: int=8,
    extra_notes: str | None=None,
) -> str:
    """This function summarizes market history available before a decision time.

    Params:
        market_history: Historical market observations.
        current_agent_knowledge: Latest date known by the agent.
        current_datetime: Current analysis timestamp.
        date_col: Date column name.
        price_col: Price column name.
        volume_col: Volume column name.
        lookback_rows: Number of recent rows considered.
        max_context_rows: Maximum rows included in context.
        extra_notes: Optional context notes.
    """
    import pandas as pd

    df = _coerce_market_history_df(market_history)
    if df.empty:
        return ""

    if price_col not in df.columns:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            raise ValueError(f"price_col '{price_col}' not found and no numeric fallback exists.")
        price_col = numeric_cols[0]

    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        boundary = current_agent_knowledge if current_agent_knowledge is not None else current_datetime
        if boundary is not None:
            df = df[df[date_col] <= pd.to_datetime(boundary)]
    elif isinstance(current_agent_knowledge, int):
        df = df.iloc[:current_agent_knowledge]

    window = df.tail(lookback_rows).copy()
    if window.empty:
        return ""

    first_price = float(window[price_col].iloc[0])
    last_price = float(window[price_col].iloc[-1])
    period_return = (last_price / first_price - 1.0) if first_price else 0.0
    all_time_high = float(df[price_col].max())
    all_time_low = float(df[price_col].min())
    high_nearness = last_price / all_time_high if all_time_high else 0.0
    low_nearness = last_price / all_time_low if all_time_low else 0.0
    recent_rows = window.tail(max_context_rows)

    lines = [
        "Retrieved market context available before the agent knowledge cut:",
        f"- Observations available: {len(df)}",
        f"- Lookback observations summarized: {len(window)}",
        f"- Last known price: {last_price:.6f}",
        f"- Lookback return: {period_return:.6f}",
        f"- All-time high in known data: {all_time_high:.6f}",
        f"- All-time low in known data: {all_time_low:.6f}",
        f"- Nearness to high p_t/p_h: {high_nearness:.6f}",
        f"- Ratio to low p_t/p_l: {low_nearness:.6f}",
    ]
    if volume_col and volume_col in window.columns:
        lines.append(f"- Average known volume in lookback: {float(window[volume_col].mean()):.6f}")
    if extra_notes:
        lines.append(f"- Notes: {extra_notes}")

    lines.append("Recent known rows:")
    display_cols = [c for c in [date_col, price_col, volume_col] if c and c in recent_rows.columns]
    if not display_cols:
        display_cols = [price_col]
    for _, row in recent_rows[display_cols].iterrows():
        values = ", ".join(f"{col}={row[col]}" for col in display_cols)
        lines.append(f"  - {values}")

    return "\n".join(lines)


def augment_prompt_with_rag_context(prompt: str, rag_context: str | None=None) -> str:
    """This function appends retrieved market context to a prompt.

    Params:
        prompt: Prompt sent to the model.
        rag_context: Optional historical context.
    """
    if not rag_context:
        return prompt
    return f"{prompt}\n\nRAG market context:\n{rag_context}"


def run_ollama_serve() -> None:
    """This function starts the Ollama server process.
    """
    subprocess.Popen(["ollama", "serve"])


def start_ollama_serve_background(wait_seconds: float=5) -> threading.Thread:
    """This function starts Ollama in a background thread.

    Params:
        wait_seconds: Startup wait duration in seconds.
    """
    thread = threading.Thread(target=run_ollama_serve, daemon=True)
    thread.start()
    time.sleep(wait_seconds)
    return thread


if os.getenv("AUTO_START_OLLAMA", "0") == "1":
    start_ollama_serve_background()


def ollama_test_api_request(used_model: str,prompt: str,default_ps: dict[str, float | int] = {"temperature":0.1,"max_tokens":512} ,enviroment: str='local') -> str | dict[str, Any]:
    """This function sends a test prompt to an Ollama model.

    Params:
        used_model: Selected model name.
        prompt: Prompt sent to the model.
        default_ps: Default generation parameters.
        enviroment: Target execution environment.
    """
    if ollama is None:
        raise ImportError("The 'ollama' package is required for Ollama calls. Install it before running local LLM simulations.")
    if enviroment != 'colab':
        print(f"[Ollama] Verificando/Descargando el modelo '{used_model}'...")
        last_status = ""
        for chunk in ollama.pull(used_model, stream=True):
            status = chunk.get('status', '')
            if status != last_status:
                print(f" -> Status: {status}")
                last_status = status
        
        print("[Ollama] Procesando tu consulta...")
        response = ollama.chat(
            model=used_model, 
            messages=[{'role': 'user', 'content': prompt}],
            options={'num_ctx':default_ps["max_tokens"],
                    'temperature':default_ps["temperature"]}
        )
        answer = response.message.content
        print(f"[Ollama RESPUESTA]: {answer}")
        return answer
    else:
        url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": used_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": default_ps["temperature"], 
            "num_predict": default_ps["max_tokens"]  
        }
    }
    
    try:
        response = None
        #response = requests.post(url, json=payload)
        
        response.raise_for_status()
        raw_text = response.json().get("response", "")
        
        # Parseo del JSON como lo diseñamos anteriormente
        start_idx = raw_text.find('{')
        end_idx = raw_text.rfind('}') + 1
        json_str = raw_text[start_idx:end_idx]
        
        return json.loads(json_str)
        
    except Exception as e:
        print(f"Error en la inferencia del agente: {e}")
        print("la variable de entorno {enviroment} debe ser colab (porfavor descomenta:)")
        print("linea 1: import requests")
        print("linea 53 #response=...")
        return {"justificacion": "Error de conexión/parseo", "accion": "MANTENER", "cantidad": 0}


def groq_test_api_request(api_key: str | None,groq_model: str,prompt: str,default_ps: dict[str, float | int] = {"temperature":0.1,"max_tokens":1000} ) -> str:
        """This function sends a test prompt to a Groq model.

        Params:
            api_key: Provider API key.
            groq_model: Groq model name.
            prompt: Prompt sent to the model.
            default_ps: Default generation parameters.
        """
        if Groq is None:
            raise ImportError("The 'groq' package is required for Groq calls. Install it before running Groq simulations.")
        client = Groq(api_key=api_key) 
        messages = [
            {
                "role": "system", 
                "content": prompt
            }
        ]

        print("Simulation started. Type 'exit' to quit.\n")
        completion = client.chat.completions.create(
            model=groq_model,  
            messages=messages,
            temperature=default_ps["temperature"],
            max_tokens=default_ps["max_tokens"],
            stream=True
            )
            
        print("Agent: ", end="")
            

        answer = ""
        for chunk in completion:
            chunk_text = chunk.choices[0].delta.content or ""
            answer += chunk_text
        return answer

def nvidia_test_api_request(nvidia_api_key: str,nvidia_model: str,prompt: str,default_ps: dict[str, float | int] = {"temperature":0.1,"max_tokens":1000} ) -> str:
        """This function sends a test prompt to an NVIDIA model.

        Params:
            nvidia_api_key: NVIDIA API key.
            nvidia_model: NVIDIA model name.
            prompt: Prompt sent to the model.
            default_ps: Default generation parameters.
        """
        if OpenAI is None:
            raise ImportError("The 'openai' package is required for NVIDIA-compatible calls.")
        print(f"[NVIDIA] Conectando a la API con el modelo '{nvidia_model}'...")
        client = OpenAI(
            base_url = "https://integrate.api.nvidia.com/v1",
            api_key = nvidia_api_key
        )       
        
        completion = client.chat.completions.create(
            model=nvidia_model,                   # <-- CORREGIDO: Usa el modelo que le pasas
            messages=[{"role": "user", "content": prompt}], # <-- CORREGIDO: Ya no está vacío
            temperature=default_ps["temperature"], 
            top_p=0.95,
            max_tokens=default_ps["max_tokens"],
            stream=True
        )
        
        # Vamos a guardar los pedazos de texto que van llegando para poder retornarlos
        texto_recibido = []
        print("[NVIDIA RESPUESTA]: ", end="", flush=True)
        
        for chunk in completion:
            if not getattr(chunk, "choices", None):
                continue
            if chunk.choices[0].delta.content is not None:
                chunk_text = chunk.choices[0].delta.content
                print(chunk_text, end="", flush=True) # Muestra en tiempo real
                texto_recibido.append(chunk_text)     # Guarda para el return
        
        print() # Al terminar el stream, hace un salto de línea
        
        # CORREGIDO: Creamos la variable answer para que no falle el return
        answer = "".join(texto_recibido)
        return answer

def alibaba_test_api_request(api_key: str | None, model: str, prompt: str, base_url: str | None=None, default_ps: dict[str, float | int]={"temperature":0.1,"max_tokens":1000}) -> str:
        """This function sends a test prompt to an Alibaba Cloud model.

        Params:
            api_key: Provider API key.
            model: Model name or loaded model.
            prompt: Prompt sent to the model.
            base_url: Optional provider endpoint.
            default_ps: Default generation parameters.
        """
        if OpenAI is None:
            raise ImportError("The 'openai' package is required for Alibaba OpenAI-compatible calls.")
        print(f"[Alibaba Cloud] Conectando a Model Studio con el modelo '{model}'...")
        client = OpenAI(
            base_url=base_url or ALIBABA_DEFAULT_BASE_URL,
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIBABA_API_KEY"),
        )
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=default_ps["temperature"],
            max_tokens=default_ps["max_tokens"],
        )
        answer = completion.choices[0].message.content
        print(f"[Alibaba Cloud RESPUESTA]: {answer}")
        return answer

def test_llm(used_model: str, service: str, api_key: str="", default_llm_settings: dict[str, float | int] = {"temperature":0.1,"max_tokens":1000}) -> str:
    """This function tests the selected LLM provider with one prompt.

    Params:
        used_model: Selected model name.
        service: Selected LLM service.
        api_key: Provider API key.
        default_llm_settings: Default LLM generation settings.
    """
    prompt = """
    Eres un inversor en un mercado continuo de doble subasta. 
    Tu precio de compra de referencia fue 800.
    El precio actual en el mercado es 1000.
    Teniendo en cuenta tu sesgo de aversión a la pérdida, ¿deseas COMPRAR, VENDER o MANTENER?
    Responde estrictamente con una de estas palabras: BUY, SELL, HOLD.
    """.strip()
   
    
    service = normalize_provider_name(service)

    if service == 'ollama':
       answer = ollama_test_api_request(used_model,prompt)
       print("ollama response: "+answer)
       return answer
    elif service == 'nvidia':
        answer = nvidia_test_api_request(api_key,used_model,prompt)
        print("Nvidia response: "+answer)
        return answer
    elif service == "groq":
        #used_model  
        answer = groq_test_api_request(api_key,used_model,prompt)
        print("Groq response: "+answer)
        return answer
    elif service == "alibaba":
        answer = alibaba_test_api_request(api_key, used_model, prompt)
        print("Alibaba Cloud response: "+answer)
        return answer

def _parse_intent(answer: str) -> str:
    """This function extracts a valid trading intention from an LLM response.

    Params:
        answer: Raw LLM response.
    """
    if "BUY" in answer: return "BUY"
    elif "SELL" in answer: return "SELL"
    else: return "HOLD"

def ollama_sim_api_request(prompt: str, model: str, agent_id: int, personality_name: str) -> str:
    """This function requests a trading intention from Ollama.

    Params:
        prompt: Prompt sent to the model.
        model: Model name or loaded model.
        agent_id: Agent identifier.
        personality_name: Agent personality name.
    """
    if ollama is None:
        print(f"  -> [ERROR API] Ollama package is not installed for Agente {agent_id}.")
        return "HOLD"
    try:
        response = ollama.chat(
            model=model, 
            messages=[{'role': 'user', 'content': prompt}],
        )            
        try:
            raw_answer = response.message.content
        except AttributeError:
            raw_answer = response['message']['content']
            
        answer = raw_answer.strip().upper()
        print(f"  -> Agente {agent_id} ({personality_name}) - API (Ollama): {answer}")
        return _parse_intent(answer)
        
    except Exception as e:
        print(f"  -> [ERROR API] Ollama falló para Agente {agent_id}: {e}")
        return "HOLD"

def nvidia_sim_api_request(url: str, api_key: str, prompt: str, model: str, agent_id: int, personality_name: str) -> str:
    """This function requests a trading intention from NVIDIA.

    Params:
        url: Provider endpoint URL.
        api_key: Provider API key.
        prompt: Prompt sent to the model.
        model: Model name or loaded model.
        agent_id: Agent identifier.
        personality_name: Agent personality name.
    """
    if OpenAI is None:
        print(f"  -> [ERROR API] OpenAI package is not installed for Nvidia-compatible calls. Agente {agent_id} fuerza HOLD.")
        return "HOLD"
    try:
        client = OpenAI(base_url=url, api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1  # Recomendado 0.0 para decisiones determinísticas en simulaciones
        )
        answer = response.choices[0].message.content.strip().upper()
        print(f"  -> Agente {agent_id} ({personality_name}) - API (Nvidia): {answer}")
        return _parse_intent(answer)
        
    except Exception as e:
        print(f"  -> [ERROR API] Nvidia falló para Agente {agent_id}: {e}")
        return "HOLD"

def groq_sim_api_request(api_key: str, groq_model: str, prompt: str, agent_id: int, personality_name: str) -> str:
    """This function requests a trading intention from Groq.

    Params:
        api_key: Provider API key.
        groq_model: Groq model name.
        prompt: Prompt sent to the model.
        agent_id: Agent identifier.
        personality_name: Agent personality name.
    """
    if Groq is None:
        raise ImportError("El módulo 'groq' no está instalado. Ejecuta: pip install groq")
        
    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        answer = response.choices[0].message.content.strip().upper()
        print(f"  -> Agente {agent_id} ({personality_name}) - API (Groq): {answer}")
        return _parse_intent(answer)
        
    except Exception as e:
        print(f"  -> [ERROR API] Groq falló para Agente {agent_id}: {e}")
        return "HOLD"


def alibaba_sim_api_request(
    api_key: str,
    model: str,
    prompt: str,
    agent_id: int,
    personality_name: str,
    base_url: str | None = None,
    rag_context: str | None = None,
) -> str:
    """This function requests a trading intention from Alibaba Cloud.

    Params:
        api_key: Provider API key.
        model: Model name or loaded model.
        prompt: Prompt sent to the model.
        agent_id: Agent identifier.
        personality_name: Agent personality name.
        base_url: Optional provider endpoint.
        rag_context: Optional historical context.
    """
    if OpenAI is None:
        print(f"  -> [ERROR API] OpenAI package is not installed for Alibaba-compatible calls. Agente {agent_id} fuerza HOLD.")
        return "HOLD"
    try:
        client = OpenAI(
            base_url=base_url or ALIBABA_DEFAULT_BASE_URL,
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIBABA_API_KEY"),
        )
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": augment_prompt_with_rag_context(prompt, rag_context)}],
            temperature=0.1,
        )
        answer = response.choices[0].message.content.strip().upper()
        print(f"  -> Agente {agent_id} ({personality_name}) - API (Alibaba Cloud): {answer}")
        return _parse_intent(answer)

    except Exception as e:
        print(f"  -> [ERROR API] Alibaba Cloud fallo para Agente {agent_id}: {e}")
        return "HOLD"
