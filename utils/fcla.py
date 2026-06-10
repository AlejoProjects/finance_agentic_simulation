from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import math
import textwrap

from pams.agents import FCNAgent
from pams.logs import ExecutionLog, MarketStepEndLog, OrderLog
from pams.logs.market_step_loggers import MarketStepSaver
from pams.market import Market
from pams.order import Cancel, Order

from . import apis as pi
from . import local_models as lm


class CustomOrderLogger(MarketStepSaver):
    """Market logger with order, execution, and FCL portfolio traces."""

    def __init__(self, portfolio_agent_classes: tuple[str, ...] | list[str]=("FCLAgent",)) -> None:
        """This function initializes order, execution, and portfolio log storage.

        Params:
            portfolio_agent_classes: Agent classes included in portfolio logs.
        """
        super().__init__()
        self.individual_orders = []
        self.individual_executions = []
        self.agent_portfolio_logs = []
        self.portfolio_agent_classes = set(portfolio_agent_classes or [])

    def process_order_log(self, log: OrderLog) -> None:
        """This function records one submitted market order.

        Params:
            log: PAMS log event.
        """
        super().process_order_log(log)
        self.individual_orders.append({
            "market_time": log.time,
            "market_id": log.market_id,
            "agent_id": log.agent_id,
            "is_buy": log.is_buy,
            "price": log.price,
            "volume": log.volume,
            "kind": str(log.kind),
            "order_id": log.order_id,
            "ttl": log.ttl,
        })

    def process_execution_log(self, log: ExecutionLog) -> None:
        """This function records one completed market execution.

        Params:
            log: PAMS log event.
        """
        self.individual_executions.append({
            "market_time": log.time,
            "market_id": log.market_id,
            "buy_agent_id": log.buy_agent_id,
            "sell_agent_id": log.sell_agent_id,
            "buy_order_id": log.buy_order_id,
            "sell_order_id": log.sell_order_id,
            "price": log.price,
            "volume": log.volume,
        })

    def process_market_step_end_log(self, log: MarketStepEndLog) -> None:
        """This function records eligible agent portfolios after a market step.

        Params:
            log: PAMS log event.
        """
        super().process_market_step_end_log(log)
        market = log.market
        current_price = market.get_market_price()

        for agent in log.simulator.agents:
            agent_type = agent.__class__.__name__
            if self.portfolio_agent_classes and agent_type not in self.portfolio_agent_classes:
                continue
            if not agent.is_market_accessible(market.market_id):
                continue

            asset_volume = agent.get_asset_volume(market.market_id)
            cash_amount = agent.get_cash_amount()
            asset_value = current_price * asset_volume
            portfolio_value = cash_amount + asset_value
            asset_proportion = asset_value / portfolio_value if portfolio_value > 0 else 0.0

            self.agent_portfolio_logs.append({
                "session_id": log.session.session_id,
                "market_time": market.get_time(),
                "market_id": market.market_id,
                "market_name": market.name,
                "agent_id": agent.agent_id,
                "agent_type": agent_type,
                "cash_amount": cash_amount,
                "asset_volume": asset_volume,
                "market_price": current_price,
                "asset_value": asset_value,
                "portfolio_value": portfolio_value,
                "asset_proportion": asset_proportion,
            })


class FCLAgent(FCNAgent):
    _personalities: Dict[str, Any] = {}
    _personality_distribution: Optional[Dict[str, int]] = None
    _agent_counter: int = 0

    _api_provider: str = "ollama"
    _api_model: str = "qwen3:4b"
    _api_key: Optional[str] = None
    _api_base_url: Optional[str] = None
    _static_rag_context: str = ""
    _estimated_response_tokens: int = 3

    @classmethod
    def configure_personalities(
        cls,
        personalities: Dict[str, Any],
        distribution: Optional[Dict[str, int]] = None,
    ) -> None:
        """This function configures personality prompts and their distribution.

        Params:
            personalities: Available personality prompts.
            distribution: Agent count per personality.
        """
        cls._personalities = personalities
        cls._personality_distribution = distribution
        cls._agent_counter = 0

    @classmethod
    def configure_api(
        cls,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        """This function configures the LLM provider used by every FCL agent.

        Params:
            provider: LLM provider name.
            model: Model name or loaded model.
            api_key: Provider API key.
            base_url: Optional provider endpoint.
        """
        cls._api_provider = pi.normalize_provider_name(provider)
        cls._api_model = model
        cls._api_key = api_key
        cls._api_base_url = base_url

    @classmethod
    def configure_rag_context(cls, rag_context: str = "") -> None:
        """This function assigns shared historical context to every FCL agent.

        Params:
            rag_context: Optional historical context.
        """
        cls._static_rag_context = rag_context or ""

    def setup(
        self,
        settings: Dict[str, Any],
        accessible_markets_ids: List[int],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """This function initializes an FCL agent from simulation settings.

        Params:
            settings: Agent configuration settings.
            accessible_markets_ids: Market identifiers available to the agent.
            args: Additional positional setup arguments.
            kwargs: Additional keyword setup arguments.
        """
        super().setup(settings, accessible_markets_ids, *args, **kwargs)
        self.my_market_id = accessible_markets_ids[0]
        self.reference_price = settings.get("referencePrice", 10.50)
        self.trading_history = []

        personality_keys = list(FCLAgent._personalities.keys())
        if not personality_keys:
            self.personality_name = "Neutral"
            self.personality_description = "Eres un inversor racional y equilibrado en el mercado."
        else:
            if FCLAgent._personality_distribution:
                flat_pool = []
                for p_name, count in FCLAgent._personality_distribution.items():
                    flat_pool.extend([p_name] * count)
                pool_index = FCLAgent._agent_counter % len(flat_pool)
                self.personality_name = flat_pool[pool_index]
            else:
                pool_index = FCLAgent._agent_counter % len(personality_keys)
                self.personality_name = personality_keys[pool_index]
            self.personality_description = FCLAgent._personalities[self.personality_name]

        FCLAgent._agent_counter += 1
        print(f"[SETUP] Agente FCL {self.agent_id} creado con Personalidad: '{self.personality_name}'")

    def _call_llm_api(self, prompt: str) -> str:
        """This function routes a prompt to the configured LLM provider.

        Params:
            prompt: Prompt sent to the model.
        """
        provider = FCLAgent._api_provider
        model = FCLAgent._api_model
        api_key = FCLAgent._api_key
        base_url = FCLAgent._api_base_url

        if provider == "ollama":
            return pi.ollama_sim_api_request(
                prompt=prompt,
                model=model,
                agent_id=self.agent_id,
                personality_name=self.personality_name,
            )

        if provider == "local":
            return lm.local_model_api_request(
                prompt=prompt,
                model=model,
                agent_id=self.agent_id,
                personality_name=self.personality_name,
                api_key=api_key,
            )

        if provider == "nvidia":
            return pi.nvidia_sim_api_request(
                url=base_url or "https://integrate.api.nvidia.com/v1",
                api_key=api_key,
                prompt=prompt,
                model=model,
                agent_id=self.agent_id,
                personality_name=self.personality_name,
            )

        if provider == "groq":
            return pi.groq_sim_api_request(
                api_key=api_key,
                groq_model=model,
                prompt=prompt,
                agent_id=self.agent_id,
                personality_name=self.personality_name,
            )

        if provider in {"alibaba", "dashscope", "qwen_cloud"}:
            return pi.alibaba_sim_api_request(
                api_key=api_key,
                model=model,
                prompt=prompt,
                agent_id=self.agent_id,
                personality_name=self.personality_name,
                base_url=base_url or pi.ALIBABA_DEFAULT_BASE_URL,
            )

        print(f"  -> [ERROR ARQUITECTURA] Proveedor API desconocido: '{provider}'. Agente {self.agent_id} fuerza HOLD.")
        return "HOLD"

    def executed_order(self, log: ExecutionLog) -> None:
        """This function updates agent history after an execution.

        Params:
            log: PAMS log event.
        """
        if log.buy_agent_id == self.agent_id:
            side = "BUY"
        elif log.sell_agent_id == self.agent_id:
            side = "SELL"
        else:
            return

        self.trading_history.append({
            "market_time": log.time,
            "side": side,
            "price": log.price,
            "volume": log.volume,
        })
        self.trading_history = self.trading_history[-8:]

    def _safe_market_price_window(self, market: Market, current_time: int) -> tuple[float, float, float]:
        """This function returns an available recent market-price window.

        Params:
            market: Current market instance.
            current_time: Current decision time.
        """
        prices = []
        for t in range(current_time + 1):
            try:
                price = market.get_market_price(t)
                if price is not None and math.isfinite(price):
                    prices.append(price)
            except Exception:
                continue
        current_price = market.get_market_price()
        if not prices:
            prices = [current_price]
        return current_price, max(prices), min(prices)

    def _portfolio_context(self, market: Market, current_price: float) -> str:
        """This function formats the agent portfolio state for its prompt.

        Params:
            market: Current market instance.
            current_price: Current market price.
        """
        asset_volume = self.get_asset_volume(market.market_id)
        cash_amount = self.get_cash_amount()
        asset_value = current_price * asset_volume
        portfolio_value = cash_amount + asset_value
        asset_proportion = asset_value / portfolio_value if portfolio_value > 0 else 0.0
        recent_history = self.trading_history[-3:]
        history_text = "; ".join(
            f"{trade['side']} {trade['volume']} @ {trade['price']:.4f} (t={trade['market_time']})"
            for trade in recent_history
        ) or "No executions recorded yet"
        return (
            f"- Cash: {cash_amount:.4f}\n"
            f"- Asset volume: {asset_volume}\n"
            f"- Asset proportion PA_t: {asset_proportion:.4f}\n"
            f"- Recent trading history: {history_text}"
        )

    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        """This function converts an LLM intention into market orders.

        Params:
            markets: Accessible market instances.
        """
        algorithmic_orders = super().submit_orders(markets)
        if not algorithmic_orders:
            return []

        market = markets[self.my_market_id]
        current_time = market.get_time()
        current_price, all_time_high, all_time_low = self._safe_market_price_window(market, current_time)
        high_nearness = current_price / all_time_high if all_time_high else 0.0
        low_nearness = current_price / all_time_low if all_time_low else 0.0
        rag_context = FCLAgent._static_rag_context.strip()
        rag_block = (
            "\nContexto historico recuperado antes de la fecha de conocimiento del agente:\n"
            f"{rag_context}\n"
            if rag_context else ""
        )

        raw_prompt = f"""
        {self.personality_description}

        Contexto Macroeconomico del Mercado:
        - Tu precio de compra de referencia historico es: {self.reference_price}
        - El precio de cotizacion actual del mercado es: {current_price}
        - Tiempo actual de mercado en ticks: {current_time}
        - Precio maximo historico observado por el agente: {all_time_high}
        - Precio minimo historico observado por el agente: {all_time_low}
        - Cercania al maximo historico p_t / p^h_1:t: {high_nearness:.6f}
        - Relacion contra minimo historico p_t / p^l_1:t: {low_nearness:.6f}

        Estado del portafolio:
        {self._portfolio_context(market, current_price)}
        {rag_block}

        Evaluando tus sesgos cognitivos, tu perfil de riesgo asignado y las condiciones actuales, determina tu accion.
        Responde estrictamente con una unica palabra: BUY, SELL o HOLD.
        """
        prompt = textwrap.dedent(raw_prompt).strip()
        llm_intent = self._call_llm_api(prompt)

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
