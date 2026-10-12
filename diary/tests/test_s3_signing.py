"""SigV4 signing correctness against the official AWS test vector.

The backend hand-rolls SigV4 (no boto3 at runtime), so it deserves the
gold-standard check: AWS's published example for S3 GET Object, GET with a
Range header, from the AWS "SigV4 Test Suite" examples. Byte-exact expected
signature; any canonicalization drift fails here.
"""
from agent.s3_storage import S3CorpusBackend


def test_sigv4_get_object_with_range_matches_aws_official_vector():
    backend = S3CorpusBackend(
        endpoint_url="https://examplebucket.s3.amazonaws.com",
        bucket="examplebucket",
        access_key="AKIAIOSFODNN7EXAMPLE",
        secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        region="us-east-1",
    )
    # Force the canonical path for the virtual-host-style official example:
    # our backend is path-style, so the object lives at /examplebucket/test.txt.
    backend.host = "examplebucket.s3.amazonaws.com"

    # Official vector: GET /test.txt with Range, frozen timestamp.
    headers = backend._signed_headers(
        "GET",
        "/test.txt",
        query={},
        payload=b"",
        extra={"Range": "bytes=0-9"},
        amzdate="20130524T000000Z",
    )
    assert headers["Authorization"] == (
        "AWS4-HMAC-SHA256 "
        "Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request, "
        "SignedHeaders=host;range;x-amz-content-sha256;x-amz-date, "
        "Signature=f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41"
    )


# ---- #1094: the signer must agree with core (server/s3-sign.cjs -> Rust) and with what httpx sends ----

import pytest

from agent.s3_storage import canonical_query, js_trim

ENDPOINTS = [
    "https://S3.Example.com",
    "https://s3.example.com:443",
    "http://minio.example:80",
    "https://minio.example:9000",
    "https://s3.example.com:8443",
    "https://bücher.example",
]


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_signed_host_is_the_host_header_the_client_sends(endpoint):
    """The canonical request covers the Host the client actually sends (lower case, IDNA, default
    port dropped), so the server computes the same signature (#1094)."""
    backend = S3CorpusBackend(endpoint_url=endpoint, bucket="b", access_key="AK", secret_key="SK")
    sent = backend._client.build_request("GET", "/b/k").headers["host"]
    assert backend.host == sent


def test_signed_host_matches_core_for_the_table_in_the_issue():
    """The rows core signs as WHATWG URL.host: lower case, default port dropped, IDNA."""
    expect = {
        "https://S3.Example.com": "s3.example.com",
        "https://s3.example.com:443": "s3.example.com",
        "http://minio.example:80": "minio.example",
        "https://bücher.example": "xn--bcher-kva.example",
    }
    for endpoint, host in expect.items():
        assert S3CorpusBackend(endpoint_url=endpoint, bucket="b", access_key="AK", secret_key="SK").host == host


@pytest.mark.parametrize("endpoint", [
    "https://user:not-a-real-secret@s3.example.com",
    "https://user@s3.example.com",
    "https://@s3.example.com",
])
def test_userinfo_in_endpoint_is_refused_without_echoing_it(endpoint):
    with pytest.raises(ValueError) as err:
        S3CorpusBackend(endpoint_url=endpoint, bucket="b", access_key="AK", secret_key="SK")
    assert "not-a-real-secret" not in str(err.value) and "user" not in str(err.value)


def test_js_trim_strips_what_ecmascript_trim_strips_and_nothing_else():
    assert js_trim("﻿v﻿") == "v"            # JS strips U+FEFF; Python str.strip() does not
    assert js_trim("\u0085v\u0085") == "\u0085v\u0085"  # JS keeps U+0085; Python str.strip() removes it
    assert js_trim("\x1cv\x1f") == "\x1cv\x1f"        # C0 separators: Python strips them, JS does not
    assert js_trim("\t   v  　") == "v"
    assert js_trim("  padded  ") == "padded"


def test_query_sorts_by_utf16_code_units_as_core_does():
    """core sorts keys in JS string order (UTF-16 code units). U+1F600 (surrogates D83D) sorts before
    U+FF01 there, but after it by code point, so the two orders differ on exactly this pair."""
    assert canonical_query({"！": "1", "\U0001f600": "2"}).startswith("%F0%9F%98%80=2&")
    assert canonical_query({"b": "1", "a": "2", "a-b": "3"}) == "a=2&a-b=3&b=1"
    assert canonical_query(None) == ""
