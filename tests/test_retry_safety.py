"""A request that may have reached the server is never silently re-sent.

The request loop used to retry EVERY method on a read timeout and on
500/502/503/504, up to three times. A POST that times out waiting for its
response may already have placed the order or moved the money, so each retry
could repeat it: one `move --amount 100` could transfer $400, and one batch
order could buy four contracts. The intra-account transfer and create-subaccount
endpoints have no idempotency key at all.

Reads, and POSTs that provably never arrived (connection refused, connect or
pool timeout) or were explicitly refused (429), are still retried.
"""
import httpx
import pytest

TRANSFER = {"transfer_id": "t-1", "source": "event_contract",
            "destination": "event_contract", "source_exchange_shard": 0,
            "destination_exchange_shard": 2, "amount": "1.0000",
            "status": "complete", "created_ts": 1789204023}


@pytest.fixture(autouse=True)
def no_sleep(mocker):
    mocker.patch("pykalshi._sync.client.time.sleep")


def _resp(mock_response, data, status=200):
    r = mock_response(data, status_code=status)
    r.headers = {}
    return r


def _posts(client):
    return [c for c in client._session.request.call_args_list if c.args[0] == "POST"]


# ---------------------------------------------------------------- POST: never replayed once sent

def test_a_transfer_that_times_out_waiting_for_its_reply_is_sent_once(client):
    client._session.request.side_effect = httpx.ReadTimeout("no reply")
    with pytest.raises(httpx.ReadTimeout):
        client.portfolio.intra_account_transfer(
            amount_centicents=1000000, destination_exchange_shard=2)
    assert len(_posts(client)) == 1, "a $100 move would have been sent again"


def test_a_transfer_answered_with_a_5xx_is_sent_once(client, mock_response):
    for status in (500, 502, 503, 504):
        client._session.request.reset_mock()
        client._session.request.side_effect = None
        client._session.request.return_value = _resp(mock_response, {}, status)
        with pytest.raises(Exception):
            client.portfolio.intra_account_transfer(amount_centicents=100)
        assert len(_posts(client)) == 1, f"re-sent after {status}"


def test_a_subaccount_create_that_times_out_is_sent_once(client):
    """Subaccounts cannot be deleted; a replay leaves a permanent extra one."""
    client._session.request.side_effect = httpx.ReadTimeout("no reply")
    with pytest.raises(httpx.ReadTimeout):
        client.portfolio.create_subaccount(exchange_index=2)
    assert len(_posts(client)) == 1


def test_a_batch_order_that_times_out_is_sent_once(client):
    """Without this, one decision could buy up to four contracts."""
    client._session.request.side_effect = httpx.ReadTimeout("no reply")
    with pytest.raises(httpx.ReadTimeout):
        client.portfolio.batch_place_orders([{
            "ticker": "KXBTC15M-26SEP130600-00", "action": "buy", "side": "yes",
            "count": "1", "yes_price_dollars": "0.0200"}])
    assert len(_posts(client)) == 1


def test_a_write_timeout_counts_as_possibly_sent(client):
    client._session.request.side_effect = httpx.WriteTimeout("partial")
    with pytest.raises(httpx.WriteTimeout):
        client.portfolio.intra_account_transfer(amount_centicents=100)
    assert len(_posts(client)) == 1


# ---------------------------------------------------------------- POST: retried when it never arrived

def test_a_refused_connection_is_retried_because_nothing_was_sent(client, mock_response):
    client._session.request.side_effect = [
        httpx.ConnectError("refused"), _resp(mock_response, {"transfer_id": "t-2"})]
    assert client.portfolio.intra_account_transfer(amount_centicents=100) == "t-2"
    assert len(_posts(client)) == 2


@pytest.mark.parametrize("exc", [httpx.ConnectTimeout, httpx.PoolTimeout])
def test_a_timeout_before_sending_is_retried(client, mock_response, exc):
    client._session.request.side_effect = [
        exc("not sent"), _resp(mock_response, {"transfer_id": "t-3"})]
    assert client.portfolio.intra_account_transfer(amount_centicents=100) == "t-3"
    assert len(_posts(client)) == 2


def test_a_429_is_retried_because_the_server_refused_it(client, mock_response):
    client._session.request.side_effect = [
        _resp(mock_response, {}, 429), _resp(mock_response, {"transfer_id": "t-4"})]
    assert client.portfolio.intra_account_transfer(amount_centicents=100) == "t-4"
    assert len(_posts(client)) == 2


# ---------------------------------------------------------------- reads keep their retries

def test_a_read_still_retries_after_a_timeout(client, mock_response):
    client._session.request.side_effect = [
        httpx.ReadTimeout("slow"), _resp(mock_response, {"transfer": TRANSFER})]
    t = client.portfolio.get_intra_account_transfer("t-1")
    assert t.status == "complete"
    assert client._session.request.call_count == 2


def test_a_read_still_retries_after_a_503(client, mock_response):
    client._session.request.side_effect = [
        _resp(mock_response, {}, 503), _resp(mock_response, {"transfer": TRANSFER})]
    assert client.portfolio.get_intra_account_transfer("t-1").transfer_id == "t-1"
    assert client._session.request.call_count == 2
