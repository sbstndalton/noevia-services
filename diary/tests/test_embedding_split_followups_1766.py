"""#1766: follow-ups to the long-chunk splitting (#1759). Synthetic text and stub senders only.

The cap on pieces, bisecting a refused batch, and the limits of the splitting (depth cap, halving when
the server reports no token count, the minimum piece size) and the wording of a context-size refusal."""
from __future__ import annotations

import httpx
import pytest

from agent import embedding as emb
from agent.llm import InputTooLarge, too_large_from_response


class Sender:
    """A ``send(texts, kinds, titles)`` that refuses any batch holding a text over ``limit`` characters."""

    def __init__(self, limit, tokens=None, token_limit=None):
        self.limit, self.tokens, self.token_limit = limit, tokens, token_limit
        self.calls = []  # list of the texts of each request

    def __call__(self, texts, kinds, titles):
        self.calls.append(list(texts))
        if any(len(t) > self.limit for t in texts):
            raise InputTooLarge("too large", self.tokens, self.token_limit)
        return [[float(len(t)), 1.0] for t in texts]


def test_pieces_from_one_cut_are_capped_whatever_token_count_a_server_reports():
    # A server that claims 10 million tokens against a limit of 100 must not fan out into ~100k requests.
    text = ("coffee river " * 4000).strip()
    send = Sender(limit=len(text) // 2, tokens=10_000_000, token_limit=100)
    out = emb.embed_fitting(send, [text], "document")
    assert len(out) == 1
    first_cut = send.calls[1:1 + emb.SPLIT_MAX_PARTS + 1]
    assert emb.SPLIT_MAX_PARTS == 32
    pieces_of_first_cut = [c for c in send.calls[1:] if len(c) == 1 and len(c[0]) < len(text)]
    # every piece of the (single) cut was sent once and accepted: no more than the cap
    assert 2 <= len(pieces_of_first_cut) <= emb.SPLIT_MAX_PARTS, len(pieces_of_first_cut)
    assert first_cut  # at least one request after the refused whole


def test_a_refused_batch_resends_only_the_refused_text_not_every_text_one_by_one():
    texts = ["alpha " * 10, "bravo " * 10, "charlie " * 10, "delta " * 400]  # the last is over the limit
    send = Sender(limit=1000)
    out = emb.embed_fitting(send, texts, "document")
    assert len(out) == 4
    assert [round(v[0]) for v in out[:3]] == [len(t) for t in texts[:3]], "accepted texts keep their own vectors, in order"
    # Bisected: whole batch, then [a, b] (accepted), then [c, d] refused again, then c alone, then d split.
    assert send.calls[0] == texts and send.calls[1] == texts[:2] and send.calls[2] == texts[2:]
    accepted = [c for c in send.calls if all(len(t) <= 1000 for t in c)]
    for t in texts[:3]:
        assert sum(c.count(t) for c in accepted) == 1, "each accepted text is embedded by exactly one successful request"
    assert [texts[0]] not in send.calls and [texts[1]] not in send.calls, "a and b were never sent alone"


def test_the_depth_cap_ends_the_splitting_with_the_refusal_raised():
    text = "x" * 100_000
    send = Sender(limit=0)  # refuses everything
    with pytest.raises(InputTooLarge):
        emb.embed_fitting(send, [text], "document")
    # The whole text, then one cut per level down the first branch, SPLIT_MAX_DEPTH levels: no deeper.
    assert len(send.calls) == 7
    assert len(send.calls[-1][0]) == len(text) // 2 ** 6


def test_with_no_token_count_a_refused_text_is_halved_until_it_fits():
    text = "x" * 4000
    send = Sender(limit=1000)  # InputTooLarge carries no tokens / limit
    out = emb.embed_fitting(send, [text], "document")
    lengths = [len(c[0]) for c in send.calls]
    assert lengths[:2] == [4000, 2000], "first cut is in halves"
    assert sorted(set(lengths)) == [1000, 2000, 4000]
    assert len(out) == 1 and len(out[0]) == 2, "one vector for the chunk"


def test_a_piece_shorter_than_the_minimum_is_a_real_refusal_and_is_not_cut_further():
    send = Sender(limit=0)
    with pytest.raises(InputTooLarge):
        emb.embed_fitting(send, ["x" * (emb.SPLIT_MIN_CHARS - 1)], "document")
    assert len(send.calls) == 1, "too short to cut: raised at once"
    send = Sender(limit=0)
    with pytest.raises(InputTooLarge):
        emb.embed_fitting(send, ["x" * 200], "document")
    assert [len(c[0]) for c in send.calls] == [200, 100, 50], "the 50-character piece is under the minimum: raised, not cut"


def test_a_refused_query_is_never_split():
    send = Sender(limit=0)
    with pytest.raises(InputTooLarge):
        emb.embed_fitting(send, ["x" * 500], "query")
    assert len(send.calls) == 1


@pytest.mark.parametrize("body,tokens,limit", [
    ('{"error":{"message":"input (2162 tokens) is too large to process. increase the physical batch size (current batch size: 2048)"}}', 2162, 2048),
    ('{"error":{"message":"the request exceeds the available context size, try increasing it","n_prompt_tokens":9000,"n_ctx":8192}}', 9000, 8192),
    ('{"error":{"type":"exceed_context_size_error","n_prompt_tokens":9000,"n_ctx":8192}}', 9000, 8192),
    ('{"error":{"message":"input is larger than the max context size. skipping"}}', None, None),
])
def test_the_context_size_and_batch_wordings_are_a_typed_too_large_with_the_counts(body, tokens, limit):
    exc = too_large_from_response(httpx.Response(500, text=body))
    assert isinstance(exc, InputTooLarge)
    assert (exc.tokens, exc.limit) == (tokens, limit)


def test_other_errors_and_successes_are_not_too_large():
    assert too_large_from_response(httpx.Response(500, text='{"error":{"message":"out of memory"}}')) is None
    assert too_large_from_response(httpx.Response(200, text="too large to process")) is None
