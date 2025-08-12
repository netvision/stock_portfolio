from dataclasses import dataclass
from typing import Optional, Dict, Any
import requests


@dataclass
class MStockConfig:
    base_url: str
    api_key: str
    jwt_token: str
    private_key: Optional[str] = None
    version: int = 1
    client_id: Optional[str] = None
    timeout: int = 10
    dry_run: bool = True


class MStockClient:
    """
    Minimal mStock API client wrapper.
    NOTE: Endpoints and payloads vary by broker/version. Configure base_url and token accordingly.
    In dry_run mode, requests are not sent; calls return mock-like responses.
    """

    def __init__(self, config: MStockConfig):
        self.config = config
        self.session = requests.Session()
        headers = {
            "X-Mirae-Version": str(self.config.version),
            "Authorization": f"token {self.config.api_key}:{self.config.jwt_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.config.private_key:
            headers["X-PrivateKey"] = self.config.private_key
        self.session.headers.update(headers)

    def _url(self, path: str) -> str:
        return f"{self.config.base_url.rstrip('/')}/{path.lstrip('/')}"

    def _maybe_send(self, method: str, path: str, json: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None):
        if self.config.dry_run:
            return {"dry_run": True, "method": method, "path": path, "json": json, "params": params}
        resp = self.session.request(method, self._url(path), json=json, params=params, timeout=self.config.timeout)
        resp.raise_for_status()
        try:
            return resp.json()
        except Exception:
            return {"status_code": resp.status_code, "text": resp.text}

    # Public methods (adjust paths to match your mStock API)
    def get_profile(self):
        # Endpoint name may vary per API grouping
        return self._maybe_send("GET", "/user/profile")

    def get_holdings(self):
        return self._maybe_send("GET", "/portfolio/holdings")

    def get_positions(self):
        return self._maybe_send("GET", "/position")

    def get_quote(self, symbol: str):
        return self._maybe_send("GET", "/market/quote", params={"symbol": symbol})

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = "MARKET",
        product: str = "CNC",
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        validity: str = "DAY",
        remarks: Optional[str] = None,
    ):
        payload = {
            "symbol": symbol,
            "side": side.upper(),
            "quantity": int(quantity),
            "order_type": order_type,
            "product": product,
            "validity": validity,
            "price": price,
            "stop_loss": stop_loss,
            "remarks": remarks,
        }
    return self._maybe_send("POST", "/orders", json=payload)

    def cancel_order(self, order_id: str):
        return self._maybe_send("DELETE", f"/orders/{order_id}")

    def list_orders(self):
        return self._maybe_send("GET", "/orders")
