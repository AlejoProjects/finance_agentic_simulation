from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import snapshot_download
from transformers import GenerationConfig
import torch
import json

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

# Caché global para evitar cargar el modelo en VRAM repetidamente por cada orden
_loaded_pipelines = {}



def download_models(repo_id,model_name,hf_token= None):
    # Puedes cambiar a "Qwen/Qwen3.5-4B-Instruct" según disponibilidad
    local_dir = f"./models/{model_name}"
    print(f"Initializing download off {repo_id} in  the dir: {local_dir}...")
    # Descarga el repositorio ignorando archivos redundantes para ahorrar espacio
    if hf_token == None:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"], 
            local_dir_use_symlinks=False
        )
    else:
            snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"], 
            local_dir_use_symlinks=False,
            token = hf_token
        )
    print("¡Descarga completada con éxito!")




# Una vez que CUDA está despierto y enlazado, importamos las librerías pesadas
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_model(model_id: str, hardware_profile: str = "laptop_gtx"):
    """
    Carga un modelo LLM local optimizado según el hardware disponible.
    
    Args:
        model_id (str): Nombre de la carpeta del modelo en ./models/
        hardware_profile (str): 'laptop_gtx' (4GB VRAM) o 'lab_workstation' (Alta VRAM)
    """
    path_al_modelo = f"./models/{model_id}"
    
    # 1. Validación de Hardware rigurosa
    has_gpu = torch.cuda.is_available()
    if not has_gpu:
        print("CRÍTICO: CUDA no detectado. El modelo correrá en CPU, arruinando la latencia de simulación.")
        device_map = "cpu"
        quantization_config = None
    else:
        print(f"GPU Detectada: {torch.cuda.get_device_name(0)} | VRAM Aprox: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # 2. Perfilado de carga basado en el entorno
        if hardware_profile == "laptop_gtx":
            print("Cargando perfil estricto: Laptop GTX (4-bits, nf4)")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            device_map = {"": 0}  # Forzamos todo a la única GPU
            
        elif hardware_profile == "lab_workstation":
            print("Cargando perfil amplio: Workstation Lab (8-bits o bf16)")
            # Cuando tengas más VRAM, 8-bits o bf16 nativo reduce la pérdida de precisión
            # crucial para el razonamiento financiero de los agentes.
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                # Si tu GPU en el lab es serie 3000+, puedes usar bfloat16
                # llm_int8_threshold=6.0 
            )
            device_map = "auto" # Deja que Accelerate distribuya los pesos si hay múltiples GPUs
            
        else:
            raise ValueError(f"Perfil de hardware '{hardware_profile}' no soportado.")

    print(f"Cargando tokenizador desde {path_al_modelo}...")
    tokenizer = AutoTokenizer.from_pretrained(path_al_modelo)

    print("Cargando pesos del modelo...")
    model = AutoModelForCausalLM.from_pretrained(
        path_al_modelo,
        quantization_config=quantization_config,
        device_map=device_map,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16 if has_gpu else torch.float32
    )
    print(f"¡Modelo cargado exitosamente! Ubicación de tensores: {model.device}")
    return model, tokenizer
def test_local_llm(prompt,model,tokenizer,default_ps = None):
    """
    Función de prueba para evaluar la inferencia de Qwen dentro del Notebook.
    Mantiene la estructura estricta de hiperparámetros y el parseo robusto de JSON.
    """
    
    # 1. Definición del sistema base para forzar la estructura de salida JSON esperada
    prompt_sistema = (
        "Eres un agente financiero autónomo. Tu objetivo es procesar la información del mercado "
        "y tomar decisiones de inversión basándote en tu perfil psicológico.\n"
        "Debes responder EXCLUSIVAMENTE en un formato JSON válido que contenga estrictamente "
        "las llaves: 'precio', 'is_buy', 'volumen', 'razonamiento', 'emocion'."
    )
    
    mensajes = [
        {"role": "system", "content": prompt_sistema},
        {"role": "user", "content": prompt}
    ]
    
    # 2. Aplicar la plantilla de chat oficial del modelo Qwen
    texto_formateado = tokenizer.apply_chat_template(
        mensajes, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # 3. Tokenizar la entrada y enviarla al dispositivo correcto (CPU o GPU)
    inputs = tokenizer([texto_formateado], return_tensors="pt").to(model.device)
    
    # 4. Inferencia con los parámetros recomendados para el modo no-thinking de Qwen3
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            temperature=default_ps.get("temperature", 0.1),
            top_p=0.8,
            top_k=20,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=default_ps.get("max_tokens", 150) 
        )
    
    # 5. Aislamiento de la respuesta
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    respuesta_texto = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()
    
    print("=== [DEBUG] RESPUESTA CRUDA DEL MODELO ===")
    print(respuesta_texto)
    print("==========================================\n")
    
    # 6. Pipeline de parseo robusto (descomentado y corregido)
    try:
        decision_json = json.loads(respuesta_texto)
        return decision_json
    except json.JSONDecodeError:
        # Fallback de extracción
        start = respuesta_texto.find('{')
        end = respuesta_texto.rfind('}') + 1
        if start != -1 and end != 0:
            try:
                return json.loads(respuesta_texto[start:end])
            except json.JSONDecodeError:
                pass
        
        return {"error": "El modelo no devolvió un JSON válido", "raw": respuesta_texto}

def local_model_api_request(prompt: str, model: str, agent_id: int, personality_name: str, api_key: str = None) -> str:
    global _loaded_pipelines
    
    if pipeline is None:
        print(f"  -> [ERROR LOCAL] Librería 'transformers' no instalada. Agente {agent_id} hizo HOLD.")
        return "HOLD"
        
    try:
        # Lazy-loading del modelo
        if model not in _loaded_pipelines:
            print(f"[SISTEMA] Cargando modelo local '{model}' en memoria (Esto puede tardar)...")
            _loaded_pipelines[model] = pipeline("text-generation", model=model, device_map="auto")
            
        pipe = _loaded_pipelines[model]
        
        # Formateo básico de prompt instruct para modelos locales
        formatted_prompt = f"User: {prompt}\nAssistant:"
        
        # REVOLUCIÓN DE RIGOR: Configuramos un objeto explícito mapeando los hiperparámetros
        # Esto soluciona la depreciación y el conflicto quitando max_length implícitos
        gen_config = GenerationConfig.from_model_config(pipe.model.config)
        gen_config.max_new_tokens = 10
        gen_config.max_length = None # Forzamos a anular el max_length conflictivo
        gen_config.temperature = 0.0  # Rigor científico para decisiones deterministas
        gen_config.do_sample = False  # Requerido por transformers cuando temp es 0
        gen_config.pad_token_id = pipe.tokenizer.eos_token_id

        # Pasamos el objeto de configuración limpio
        output = pipe(
            formatted_prompt, 
            generation_config=gen_config, 
            return_full_text=False
        )
        
        answer = output[0]['generated_text'].strip().upper()
        
        print(f"  -> Agente {agent_id} ({personality_name}) - API (Local): {answer}")
        
        if "BUY" in answer: return "BUY"
        elif "SELL" in answer: return "SELL"
        else: return "HOLD"
        
    except Exception as e:
        print(f"  -> [ERROR API] Modelo Local falló para Agente {agent_id}: {e}")
        return "HOLD"