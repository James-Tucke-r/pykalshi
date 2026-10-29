import os
import pytest
from unittest import mock
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from pykalshi._base import _BaseKalshiClient


@pytest.fixture
def mock_load_private_key():
    with mock.patch("pykalshi._base._BaseKalshiClient._load_private_key") as m:
        yield m


@pytest.fixture
def mock_load_private_key_string():
    with mock.patch("pykalshi._base._BaseKalshiClient._load_private_key_string") as m:
        yield m


@mock.patch.dict(os.environ, {"KALSHI_API_KEY_ID": "env_api_key"})
def test_credential_precedence_private_key(mock_load_private_key, mock_load_private_key_string):
    """Test that private_key arg takes precedence over env vars and private_key_path."""
    mock_load_private_key_string.return_value = mock.Mock(spec=RSAPrivateKey)
    
    with mock.patch.dict(os.environ, {"KALSHI_PRIVATE_KEY": "env_private_key_str", "KALSHI_PRIVATE_KEY_PATH": "env_path"}):
        client = _BaseKalshiClient(
            private_key="arg_private_key_str",
            private_key_path="arg_path"
        )
        
        # Should use the argument string, not the env var or path
        mock_load_private_key_string.assert_called_once_with("arg_private_key_str")
        mock_load_private_key.assert_not_called()
        assert isinstance(client.private_key, RSAPrivateKey)


@mock.patch.dict(os.environ, {"KALSHI_API_KEY_ID": "env_api_key"})
def test_credential_precedence_private_key_path(mock_load_private_key, mock_load_private_key_string):
    """Test that private_key_path arg takes precedence over env vars if private_key is omitted."""
    mock_load_private_key.return_value = mock.Mock(spec=RSAPrivateKey)
    
    with mock.patch.dict(os.environ, {"KALSHI_PRIVATE_KEY": "env_private_key_str", "KALSHI_PRIVATE_KEY_PATH": "env_path"}):
        client = _BaseKalshiClient(
            private_key_path="arg_path"
        )
        
        # Should use the argument path, not the env var
        mock_load_private_key.assert_called_once_with("arg_path")
        mock_load_private_key_string.assert_not_called()


@mock.patch.dict(os.environ, {"KALSHI_API_KEY_ID": "env_api_key"})
def test_credential_precedence_env_vars(mock_load_private_key, mock_load_private_key_string):
    """Test that KALSHI_PRIVATE_KEY takes precedence over KALSHI_PRIVATE_KEY_PATH when both are set."""
    mock_load_private_key_string.return_value = mock.Mock(spec=RSAPrivateKey)
    
    with mock.patch.dict(os.environ, {"KALSHI_PRIVATE_KEY": "env_private_key_str", "KALSHI_PRIVATE_KEY_PATH": "env_path"}):
        client = _BaseKalshiClient()
        
        # Should use the env var string
        mock_load_private_key_string.assert_called_once_with("env_private_key_str")
        mock_load_private_key.assert_not_called()


@mock.patch.dict(os.environ, {"KALSHI_API_KEY_ID": "env_api_key"})
def test_invalid_private_key_type():
    """Test that passing an invalid type to private_key raises TypeError."""
    with pytest.raises(TypeError, match="Expected private_key to be str or RSAPrivateKey, got int"):
        _BaseKalshiClient(private_key=12345)


# ---------------------------------------------------------------------------
# Cancel routing (exchange shards)
# ---------------------------------------------------------------------------
# Orders on a non-zero exchange shard (e.g. the USDC crypto shard) cannot be
# cancelled without exchange_index or market_ticker: the API defaults to shard 0
# and returns 404. See "Cancel Order (V2)" / "Batch Cancel Orders (V2)".

from unittest.mock import MagicMock

from pykalshi._sync.portfolio import Portfolio
from pykalshi.enums import OrderStatus


def _portfolio(response):
    client = MagicMock()
    client.delete.return_value = response
    return Portfolio(client), client


def test_cancel_order_defaults_send_no_routing_params():
    p, client = _portfolio({"order_id": "o1", "ticker": "T"})
    p.cancel_order("o1")
    assert client.delete.call_args[0][0] == "/portfolio/events/orders/o1"


def test_cancel_order_sends_market_ticker_and_exchange_index():
    p, client = _portfolio({"order_id": "o1", "ticker": "T"})
    p.cancel_order("o1", exchange_index=-1, market_ticker="KXBTC15M-26SEP010000-00")
    endpoint = client.delete.call_args[0][0]
    assert "exchange_index=-1" in endpoint
    assert "market_ticker=KXBTC15M-26SEP010000-00" in endpoint


def test_cancel_order_sends_explicit_shard():
    p, client = _portfolio({"order_id": "o1", "ticker": "T"})
    p.cancel_order("o1", exchange_index=2)
    assert "exchange_index=2" in client.delete.call_args[0][0]


def test_batch_cancel_applies_routing_to_every_item():
    p, client = _portfolio({"orders": [{"order_id": "a"}, {"order_id": "b"}]})
    p.batch_cancel_orders(["a", "b"], exchange_index=-1, market_ticker="T")
    body = client.delete.call_args[0][1]
    assert len(body["orders"]) == 2
    for item in body["orders"]:
        assert item["exchange_index"] == -1
        assert item["market_ticker"] == "T"


def test_batch_cancel_without_routing_is_unchanged():
    p, client = _portfolio({"orders": [{"order_id": "a"}]})
    p.batch_cancel_orders(["a"])
    assert client.delete.call_args[0][1]["orders"] == [{"order_id": "a"}]
