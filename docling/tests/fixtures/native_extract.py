"""CI-only fixed fake-engine adapter; actual HTTP and worker modules are unchanged."""
import socket
from fake_converter import extract as fake

SUPPORTED = {".bmp", ".csv", ".docx", ".epub", ".htm", ".html", ".jpeg", ".jpg",
             ".md", ".odp", ".ods", ".odt", ".pdf", ".png", ".pptx", ".tif", ".tiff", ".xlsx"}


def extract(path, name):
    # Fixed synthetic instrumentation proves conversion started before a disconnect.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as notice:
        notice.sendto(b"started", ("127.0.0.1", 18931))
    return fake(path, name)
