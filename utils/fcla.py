from typing import List, Union, Dict, Any, Optional
from pams.logs.market_step_loggers import MarketStepSaver
from pams.order import Order, Cancel
from pams.agents import FCNAgent
from pams.logs import OrderLog
from pams.market import Market
from . import local_models as lm
from . import apis as pi



# 1. Definimos un Logger propio que sí guarde las órdenes individuales
class CustomOrderLogger(MarketStepSaver):
    def __init__(self):
        super().__init__()
        self.individual_orders = []  # Aquí guardaremos cada orden individual

    def process_order_log(self, log: OrderLog) -> None:
        # Este método se dispara automáticamente cada vez que un agente envía una orden
        super().process_order_log(log)
        
        # Guardamos los atributos del OrderLog que mostraste en base.py
        self.individual_orders.append({
            "market_time": log.time,
            "agent_id": log.agent_id,
            "is_buy": log.is_buy,
            "price": log.price,
            "volume": log.volume,
            "kind": str(log.kind)
        })
class FCLAgent(FCNAgent):
    # Atributos de Clase globales compartidos para configuración dinámica
    _personalities: Dict[str, str] = {}
    _personality_distribution: Optional[Dict[str, int]] = None
    _agent_counter: int = 0
    
    _api_provider: str = "ollama"
    _api_model: str = "qwen3:4b"
    _api_key: Optional[str] = None
    _api_base_url: Optional[str] = None

    @classmethod
    def configure_personalities(cls, personalities: Dict[str, str], distribution: Optional[Dict[str, int]] = None):
        """
        Configura las descripciones de las personalidades y sus cuotas de distribución.
        Si distribution es None, se repartirán en partes iguales por defecto.
        """
        cls._personalities = personalities
        cls._personality_distribution = distribution
        cls._agent_counter = 0  # Reiniciar el contador de asignación para nuevas simulaciones

    @classmethod
    def configure_api(cls, provider: str, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Define el motor de inferencia de IA (ollama o nvidia), el modelo exacto y credenciales.
        """
        cls._api_provider = provider.lower()
        cls._api_model = model
        cls._api_key = api_key
        cls._api_base_url = base_url

    def setup(self, settings: Dict[str, Any], accessible_markets_ids: List[int], *args: Any, **kwargs: Any) -> None:
        # Inicialización matemática nativa de PAMS
        super().setup(settings, accessible_markets_ids, *args, **kwargs)
        self.my_market_id = accessible_markets_ids[0]
        self.reference_price = settings.get("referencePrice", 10.50)
        
        # --- Lógica de Asignación Automática de Personalidades ---
        personality_keys = list(FCLAgent._personalities.keys())
        
        if not personality_keys:
            self.personality_name = "Neutral"
            self.personality_description = "Eres un inversor racional y equilibrado en el mercado."
        else:
            if FCLAgent._personality_distribution:
                # Caso A: Distribución explícita definida por el usuario
                flat_pool = []
                for p_name, count in FCLAgent._personality_distribution.items():
                    flat_pool.extend([p_name] * count)
                
                # Asignación secuencial basada en el índice de creación del agente
                pool_index = FCLAgent._agent_counter % len(flat_pool)
                self.personality_name = flat_pool[pool_index]
            else:
                # Caso B: Partes iguales (Round-Robin sobre las llaves disponibles)
                pool_index = FCLAgent._agent_counter % len(personality_keys)
                self.personality_name = personality_keys[pool_index]
                
            self.personality_description = FCLAgent._personalities[self.personality_name]
        # Incrementamos el contador estático de la clase
        FCLAgent._agent_counter += 1
        print(f"[SETUP] Agente FCL {self.agent_id} creado con Personalidad: '{self.personality_name}'")

    def _call_llm_api(self, prompt: str) -> str:
        """
        Conector abstracto compatible con arquitecturas modulares (Ollama, NVIDIA, Groq, Local).
        La lógica de red, parsing de respuestas y manejo de excepciones está abstraída en 'apis' y 'local_models'.
        """
        provider = FCLAgent._api_provider
        model = FCLAgent._api_model
        api_key = FCLAgent._api_key
        base_url = FCLAgent._api_base_url
        
        # 1. Enrutar según proveedor
        if provider == "ollama":
            return pi.ollama_sim_api_request(
                prompt=prompt, 
                model=model, 
                agent_id=self.agent_id, 
                personality_name=self.personality_name
            )
            
        elif provider == "local":
            return lm.local_model_api_request(
                prompt=prompt, 
                model=model, 
                agent_id=self.agent_id, 
                personality_name=self.personality_name,
                api_key=api_key # Opcional, por si usas HuggingFace Tokens en local
            )
            
        elif provider == "nvidia":
            url = base_url or "https://integrate.api.nvidia.com/v1"
            return pi.nvidia_sim_api_request(
                url=url,
                api_key=api_key,
                prompt=prompt,
                model=model,
                agent_id=self.agent_id,
                personality_name=self.personality_name
            )
            
        elif provider == "groq":
            return pi.groq_sim_api_request(
                api_key=api_key,
                groq_model=model,
                prompt=prompt,
                agent_id=self.agent_id,
                personality_name=self.personality_name
            )
            
        else:
            print(f"  -> [ERROR ARQUITECTURA] Proveedor API desconocido: '{provider}'. Agente {self.agent_id} fuerza HOLD.")
            return "HOLD"
    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        import textwrap # Importación segura por si no está a nivel global
        
        algorithmic_orders = super().submit_orders(markets)
        if not algorithmic_orders:
            return []    
            
        market = markets[self.my_market_id]
        current_price = market.get_market_price()
        
        # Inyección dinámica de la Personalidad asignada en el Prompt estructurado
        raw_prompt = f"""
        {self.personality_description}
        Contexto Macroeconómico del Mercado:
        - Tu precio de compra de referencia histórico es: {self.reference_price}
        - El precio de cotización actual del mercado es: {current_price}

        Evaluando tus sesgos cognitivos, tu perfil de riesgo asignado y las condiciones actuales, determina tu acción.
        Responde estrictamente con una única palabra: BUY, SELL o HOLD.
        """
        
        # Magia pura: dedent elimina todos los espacios de indentación del código,
        # y strip quita los saltos de línea arriba y abajo.
        prompt = textwrap.dedent(raw_prompt).strip()
        
        # 3. Evaluar decisión por medio de IA
        llm_intent = self._call_llm_api(prompt)
        
        # 4. Filtrar órdenes matemáticas usando el validador del LLM
        final_orders = []
        for order in algorithmic_orders:
            if isinstance(order, Cancel):
                final_orders.append(order)
            elif isinstance(order, Order):
                if llm_intent == "BUY" and order.is_buy:
                    final_orders.append(order)
                elif llm_intent == "SELL" and not order.is_buy:
                    final_orders.append(order)
        
        return final_orders
        