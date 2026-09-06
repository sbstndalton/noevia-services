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
