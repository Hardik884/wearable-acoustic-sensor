// SPINE-style binary protocol, C++ side (header-only, no Arduino dependency).
// Spec: protocol/spine_protocol.md.  Python twin: spine/protocol.py.
//
// Frame (little-endian):
//   0xA5 0x5A | ver u8 | type u8 | node u8 | seq u16 | len u16 | payload | crc16 u16
// CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF) over ver..payload.
#pragma once
#include <stdint.h>
#include <stddef.h>

namespace spine {

static const uint8_t SYNC0 = 0xA5, SYNC1 = 0x5A, VERSION = 1;
static const size_t HEADER_SIZE = 9;

enum MsgType : uint8_t {
  SERVICE_DISCOVERY = 0x01, SETUP_SENSOR = 0x02, SETUP_PROBE = 0x03, CALIBRATION_UPDATE = 0x04,
  START = 0x05, STOP = 0x06, RESET = 0x07,
  SERVICE_ADVERTISEMENT = 0x81, DATA = 0x82, LIVE = 0x83, SERVICE_MESSAGE = 0x84
};
enum MsgCode : uint8_t { CODE_ACK = 0, CODE_ERROR = 1, CODE_STATUS = 2 };
static const uint8_t FLAG_PROBE_USED = 0x01, FLAG_SIMULATED = 0x02;

inline uint16_t crc16_update(uint16_t crc, uint8_t b) {
  crc ^= (uint16_t)b << 8;
  for (int i = 0; i < 8; i++) crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
  return crc;
}

// Streams one frame into any Sink that has `void write(const uint8_t*, size_t)`.
// The CRC is accumulated on the fly, so large DATA frames need no big buffer.
template <class Sink>
class FrameWriter {
 public:
  explicit FrameWriter(Sink& s) : s_(s) {}
  void begin(uint8_t type, uint8_t node, uint16_t seq, uint16_t len) {
    const uint8_t sync[2] = {SYNC0, SYNC1};
    s_.write(sync, 2);
    crc_ = 0xFFFF;
    u8(VERSION); u8(type); u8(node); u16(seq); u16(len);
  }
  void u8(uint8_t v) { crc_ = crc16_update(crc_, v); s_.write(&v, 1); }
  void u16(uint16_t v) { u8(v & 0xFF); u8(v >> 8); }
  void i16(int16_t v) { u16((uint16_t)v); }
  void u32(uint32_t v) { u16(v & 0xFFFF); u16(v >> 16); }
  void bytes(const uint8_t* p, size_t n) { for (size_t i = 0; i < n; i++) u8(p[i]); }
  void end() { const uint8_t c[2] = {(uint8_t)(crc_ & 0xFF), (uint8_t)(crc_ >> 8)}; s_.write(c, 2); }
 private:
  Sink& s_;
  uint16_t crc_ = 0xFFFF;
};

struct Frame {
  uint8_t type = 0, node = 0;
  uint16_t seq = 0, len = 0;
  uint8_t payload[32];  // coordinator->node commands are tiny
};

// Byte-at-a-time receive parser. push() returns true when `out` holds a valid frame.
class FrameParser {
 public:
  bool push(uint8_t b, Frame& out) {
    switch (st_) {
      case 0: if (b == SYNC0) st_ = 1; return false;
      case 1: st_ = (b == SYNC1) ? 2 : (b == SYNC0 ? 1 : 0); crc_ = 0xFFFF; n_ = 0; return false;
      case 2:  // header body: ver type node seq(2) len(2)
        hdr_[n_++] = b; crc_ = crc16_update(crc_, b);
        if (n_ == 7) {
          out.type = hdr_[1]; out.node = hdr_[2];
          out.seq = hdr_[3] | (hdr_[4] << 8);
          out.len = hdr_[5] | (hdr_[6] << 8);
          if (out.len > sizeof(out.payload)) { st_ = 0; return false; }
          n_ = 0; st_ = out.len ? 3 : 4; rx_crc_n_ = 0;
        }
        return false;
      case 3:
        out.payload[n_++] = b; crc_ = crc16_update(crc_, b);
        if (n_ == out.len) { st_ = 4; rx_crc_n_ = 0; }
        return false;
      default:  // 4: crc low, then high
        if (rx_crc_n_ == 0) { rx_crc_lo_ = b; rx_crc_n_ = 1; return false; }
        st_ = 0;
        return ((uint16_t)(rx_crc_lo_ | (b << 8)) == crc_);
    }
  }
 private:
  uint8_t st_ = 0, hdr_[7], rx_crc_n_ = 0, rx_crc_lo_ = 0;
  uint16_t crc_ = 0xFFFF, n_ = 0;
};

inline uint16_t rd16(const uint8_t* p) { return p[0] | (p[1] << 8); }

}  // namespace spine
