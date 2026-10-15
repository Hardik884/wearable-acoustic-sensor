from spine import protocol as P
from spine.transports.serial_transport import SerialTransport


def test_serial_transport_loopback():
    """loop:// echoes our own bytes: proves open/write/read + framing through pyserial."""
    t = SerialTransport("loop://")
    t.open()
    try:
        frame = P.encode_frame(P.MsgType.SERVICE_ADVERTISEMENT,
                               P.enc_advert(2, 1, 10000, 0x100, 12, 0, 120, 300, True, 300, 4000, 50), 1, 1)
        t.write(frame)
        parser, got = P.FrameParser(), []
        for _ in range(10):
            got += parser.feed(t.read(0.05))
            if got:
                break
        assert got and P.decode(got[0])["rate_hz"] == 10000
    finally:
        t.close()
