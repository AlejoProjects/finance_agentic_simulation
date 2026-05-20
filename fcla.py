
from typing import List, Union
from pams.agents import FCNAgent
from pams.order import Order, Cancel
import random
import ollama
def test_llm(used_model:str):
        prompt = f"""
        Eres un inversor en un mercado continuo de doble subasta.
        Tu precio de compra de referencia fue 800.
        El precio actual en el mercado es 1000.
        Teniendo en cuenta tu sesgo de aversión a la pérdida, ¿deseas COMPRAR, VENDER o MANTENER?
        Responde estrictamente con una de estas palabras: BUY, SELL, HOLD.
        """
        ollama.pull(used_model)
        # Load and generate a response
        response = ollama.chat(model=used_model, messages=[
        {
            'role': 'user',
            'content': prompt,
        },
        ,
        options: {
        num_ctx: 100
            }
        ])
        print(response['message']['content'])

class FCLAgent(FCNAgent):

    def setup(self, settings, accessible_markets_ids, *args, **kwargs):
        super().setup(settings, accessible_markets_ids, *args, **kwargs)
        self.my_market_id = accessible_markets_ids[0]
        self.reference_price = settings.get("referencePrice", 10.50)
        print(f"Agente FCL {self.agent_id}")
    
    def _call_llm_api(self, prompt: str) -> str:
        try:
            used_model="qwen3:4b"
            print(f"  -> Consultando a Ollama para Agente {self.agent_id}...")
            response = ollama.chat(
                model=used_model, 
                messages=[{'role': 'user', 'content': prompt}]
            )
            print(response)
            answer = response['message']['content'].strip().upper()
            print(f"  -> Ollama Respondió: {answer}")
            
            if "BUY" in answer: return "BUY"
            elif "SELL" in answer: return "SELL"
            else: return "HOLD"
                
        except Exception as e:
            print(f"  -> Error local en Ollama: {e}")
            return "HOLD"

    def submit_orders(self, markets) -> List[Union[Order, Cancel]]:
        # 1. Calculamos la orden matemática base (FCN puro)
        algorithmic_orders = super().submit_orders(markets)
        
        # 2. Si el algoritmo no quiere hacer nada en este tick, nos saltamos a Ollama para no gastar tiempo
        if not algorithmic_orders:
            return []
            
        # ¡Si llegamos aquí, el agente quiere poner una orden!
        print(f"\n[TICK ACTIVO] Agente {self.agent_id} generó orden matemática: {algorithmic_orders}")
        
        market = markets[self.my_market_id]
        current_price = market.get_market_price()
        
        prompt = f"""
        Eres un inversor en un mercado continuo de doble subasta.
        Tu precio de compra de referencia fue {self.reference_price}.
        El precio actual en el mercado es {current_price}.
        Teniendo en cuenta tu sesgo de aversión a la pérdida, ¿deseas COMPRAR, VENDER o MANTENER?
        Responde estrictamente con una de estas palabras: BUY, SELL, HOLD.
        """
        
        llm_intent = self._call_llm_api(prompt)

        final_orders = []
        for order in algorithmic_orders:
            if isinstance(order, Cancel):
                final_orders.append(order)
                print(f"  -> Orden cancelada conservada.")
            elif isinstance(order, Order):
                if llm_intent == "BUY" and order.is_buy:
                    final_orders.append(order)
                    print(f"  -> LLM Aprobó COMPRA.")
                elif llm_intent == "SELL" and not order.is_buy:
                    final_orders.append(order)
                    print(f"  -> LLM Aprobó VENTA.")
                else:
                    print(f"  -> LLM Rechazó la orden (HOLD o contradicción).")

        return final_orders