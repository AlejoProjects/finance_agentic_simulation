
from openai import OpenAI
from groq import Groq
import ollama
def ollama_test_api_request(used_model,prompt,default_ps = {"temperature":0.1,"max_tokens":512} ):
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

def groq_test_api_request(api_key,groq_model,prompt,default_ps = {"temperature":0.1,"max_tokens":1000} ):
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

def nvidia_test_api_request(nvidia_api_key,nvidia_model,prompt,default_ps = {"temperature":0.1,"max_tokens":1000} ):
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

def test_llm(used_model: str, service: str, api_key="", default_llm_settings = {"temperature":0.1,"max_tokens":1000}):
    prompt = """
    Eres un inversor en un mercado continuo de doble subasta. 
    Tu precio de compra de referencia fue 800.
    El precio actual en el mercado es 1000.
    Teniendo en cuenta tu sesgo de aversión a la pérdida, ¿deseas COMPRAR, VENDER o MANTENER?
    Responde estrictamente con una de estas palabras: BUY, SELL, HOLD.
    """.strip()
   
    
    if service == 'ollama':
       answer = ollama_test_api_request(used_model,prompt)
       print("ollama response: "+answer)
       return answer
    elif service == 'nvidia':
        nvidia_test_api_request(api_key,used_model,prompt)
        print("Nvidia response: "+answer)
        return answer
    elif service == "groq":
        #used_model  
        answer = groq_test_api_request(api_key,used_model,prompt)
        print("Groq response: "+answer)
        return answer

def _parse_intent(answer: str) -> str:
    """Función auxiliar DRY para limpiar la respuesta del LLM y devolver la intención."""
    if "BUY" in answer: return "BUY"
    elif "SELL" in answer: return "SELL"
    else: return "HOLD"

def ollama_sim_api_request(prompt: str, model: str, agent_id: int, personality_name: str) -> str:
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