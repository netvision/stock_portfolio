from dataclasses import dataclass
from typing import Optional, Dict, Any
import requests


@dataclass
class MStockConfig:
    base_url: str
    api_key: str
    jwt_token: Optional[str] = None
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
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.config.jwt_token:
            headers["Authorization"] = f"token {self.config.api_key}:{self.config.jwt_token}"
        if self.config.private_key:
            headers["X-PrivateKey"] = self.config.private_key
        self.session.headers.update(headers)

    def set_jwt(self, jwt_token: str):
        self.config.jwt_token = jwt_token
        # update header
        self.session.headers["Authorization"] = f"token {self.config.api_key}:{self.config.jwt_token}"

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
        # Use documented fund summary endpoint as a connectivity check
        return self._maybe_send("GET", "/openapi/typea/user/fundsummary")

    def get_holdings(self):
        return self._maybe_send("GET", "/openapi/typea/portfolio/holdings")

    def get_positions(self):
        return self._maybe_send("GET", "/openapi/typea/position")

    def get_quote(self, symbol: str):
        return self._maybe_send("GET", "/openapi/typea/market/quote", params={"symbol": symbol})

    # --- Authentication Flow (Type A) ---
    def request_login_otp(self, username: str, password: str):
        """Call connect/login to trigger OTP to registered number.
        Headers require only X-Mirae-Version and form-encoded body.
        """
        if self.config.dry_run:
            return {"dry_run": True, "endpoint": "connect/login", "username": username}
        headers = {"X-Mirae-Version": str(self.config.version), "Content-Type": "application/x-www-form-urlencoded"}
        data = {"username": username, "password": password}
        resp = self.session.post(self._url("/openapi/typea/connect/login"), headers=headers, data=data, timeout=self.config.timeout)
        resp.raise_for_status()
        return resp.json()

    def generate_session_token(self, otp: str, checksum: str = "L"):
        """Exchange API key + OTP + checksum for access token (JWT)."""
        if self.config.dry_run:
            return {"dry_run": True, "endpoint": "session/token", "otp": otp, "checksum": checksum, "mock_access_token": "dryrun.jwt"}
        headers = {"X-Mirae-Version": str(self.config.version), "Content-Type": "application/x-www-form-urlencoded"}
        data = {"api_key": self.config.api_key, "request_token": otp, "checksum": checksum}
        resp = self.session.post(self._url("/openapi/typea/session/token"), headers=headers, data=data, timeout=self.config.timeout)
        resp.raise_for_status()
        j = resp.json()
        # Try to pick token from standard field names
        token = None
        try:
            token = j.get("data", {}).get("access_token")
        except Exception:
            token = None
        if token:
            self.set_jwt(token)
        return j

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
        # NOTE: Real API expects form-encoded; adjust when wiring live orders
        return self._maybe_send("POST", "/openapi/typea/orders", json=payload)

    def cancel_order(self, order_id: str):
        return self._maybe_send("DELETE", f"/openapi/typea/orders/{order_id}")

    def list_orders(self):
        return self._maybe_send("GET", "/openapi/typea/orders")
