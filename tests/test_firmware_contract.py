"""Cheap static checks on the firmware source (no compiler needed)."""
import re
from pathlib import Path

INO = Path(__file__).resolve().parent.parent / "firmware" / "spine_node" / "spine_node.ino"


def test_ino_present_and_pins_on_adc1():
    src = INO.read_text()
    assert "PIN_MIC0   34" in src and "PIN_MIC1   35" in src       # ESP32 ADC1 (GPIO32-39)
    s3 = re.search(r"PIN_MIC0\s+(\d+)\s+#define PIN_MIC1\s+(\d+)", src.split("#if BOARD_IS_S3")[1])
    assert s3 and 1 <= int(s3.group(1)) <= 10 and 1 <= int(s3.group(2)) <= 10   # S3 ADC1 = GPIO1-10


def test_advert_length_matches_python():
    from spine import protocol as P
    n = len(P.enc_advert(2, 1, 10000, 0x100, 12, 0, 1, 1, True, 1, 1, 1))
    assert f"w.begin(spine::SERVICE_ADVERTISEMENT, NODE_ID, ++tx_seq, {n})" in INO.read_text()
