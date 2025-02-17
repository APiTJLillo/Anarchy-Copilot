"""Content encoding interceptor for handling compressed request/response bodies."""
import gzip
import zlib
import codecs
from typing import Optional, Set
from email.message import Message

from .interceptor import RequestInterceptor, InterceptedRequest, InterceptedResponse

class ContentEncodingInterceptor(RequestInterceptor):
    """Interceptor that validates and handles content encoding."""

    # Set of supported content encodings
    SUPPORTED_ENCODINGS: Set[str] = {"gzip", "identity"}
    # Set of supported charset encodings
    SUPPORTED_CHARSETS: Set[str] = {
        "utf-8", "ascii", "iso-8859-1", "latin1",
        "utf-16", "utf-16le", "utf-16be",
        "utf-32", "utf-32le", "utf-32be"
    }

    def _parse_content_type(self, content_type: str) -> "tuple[str, Optional[str]]":
        """Parse Content-Type header to get media type and charset."""
        if not content_type:
            return "", None
        
        msg = Message()
        msg["content-type"] = content_type
        params = dict(msg.get_params([], "content-type"))
        charset = params.get("charset")
        media_type = params.get("", content_type.split(";")[0].strip())
        return media_type, charset

    async def intercept(self, request: InterceptedRequest) -> InterceptedRequest:
        """Validate content encoding of request body."""
        # Get and validate Content-Encoding headers
        content_encoding = request.get_header("Content-Encoding", "identity").lower()
        x_content_encoding = request.get_header("X-Content-Encoding")
        
        # Check for conflicting content encodings
        if x_content_encoding:
            raise ValueError("Conflicting or duplicate content encodings")
        
        # Check for valid content encoding
        if content_encoding not in self.SUPPORTED_ENCODINGS:
            raise ValueError(f"Unsupported content encoding: {content_encoding}")
        
        # Handle gzip encoded content
        if content_encoding == "gzip" and request.body is not None:
            # Special case: empty content
            if len(request.body) == 0:
                request.body = gzip.compress(b"")
                request.set_header("Content-Length", "20")  # Empty gzip content length
                return request

            try:
                # Attempt to decompress - if corrupted this will raise an error
                decompressed = gzip.decompress(request.body)
                # Recompress to ensure valid gzip content
                recompressed = gzip.compress(decompressed)
                request.body = recompressed
                # Update Content-Length to match recompressed data
                request.set_header("Content-Length", str(len(recompressed)))
            except (gzip.BadGzipFile, zlib.error, OSError) as e:
                # Invalid gzip content
                raise ValueError(f"Corrupted gzip content: {str(e)}")

        # Check for conflicting transfer encodings
        transfer_encoding = request.get_header("Transfer-Encoding", "").lower()
        if transfer_encoding:
            if "gzip" in transfer_encoding or (
                content_encoding != "identity" and content_encoding in transfer_encoding
            ):
                raise ValueError("Conflicting content and transfer encodings")

        # Get and validate Content-Type charset
        content_type = request.get_header("Content-Type", "")
        _, charset = self._parse_content_type(content_type)
        if charset and charset.lower() not in self.SUPPORTED_CHARSETS:
            raise ValueError(f"Unsupported charset: {charset}")

        # Validate Content-Length if present
        content_length = request.get_header("Content-Length")
        if content_length is not None:
            try:
                length = int(content_length)
                if request.body is not None and len(request.body) != length:
                    raise ValueError("Content-Length header does not match body length")
            except ValueError as e:
                raise ValueError(f"Invalid Content-Length header: {str(e)}")

        return request
