// Host-side check that the C++ encoder produces the same bytes as Python.
//   g++ -std=c++11 firmware/test/test_protocol.cpp -o test_protocol && ./test_protocol protocol/golden_vectors.json
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "../spine_node/spine_protocol.h"

struct VecSink {
  std::vector<uint8_t> v;
  void write(const uint8_t* p, size_t n) { v.insert(v.end(), p, p + n); }
};

static std::string hex(const std::vector<uint8_t>& v) {
  static const char* d = "0123456789abcdef";
  std::string s;
  for (uint8_t b : v) { s += d[b >> 4]; s += d[b & 15]; }
  return s;
}

static std::string golden(const std::string& json, const std::string& key) {
  size_t k = json.find("\"" + key + "\"");
  if (k == std::string::npos) return "";
  size_t a = json.find('"', json.find(':', k)) + 1;
  return json.substr(a, json.find('"', a) - a);
}

static int fails = 0;
static void check(const std::string& json, const char* name, const std::vector<uint8_t>& got) {
  bool ok = hex(got) == golden(json, name);
  printf("%-14s %s\n", name, ok ? "OK" : "MISMATCH");
  if (!ok) { fails++; printf("  got    %s\n  expect %s\n", hex(got).c_str(), golden(json, name).c_str()); }
}

int main(int argc, char** argv) {
  std::ifstream f(argc > 1 ? argv[1] : "protocol/golden_vectors.json");
  std::stringstream ss; ss << f.rdbuf();
  std::string json = ss.str();
  using namespace spine;
  { VecSink s; FrameWriter<VecSink> w(s); w.begin(SERVICE_DISCOVERY, 0, 1, 0); w.end(); check(json, "discovery", s.v); }
  { VecSink s; FrameWriter<VecSink> w(s); w.begin(SETUP_SENSOR, 1, 2, 6); w.u16(10000); w.u16(150); w.u16(300); w.end(); check(json, "setup_sensor", s.v); }
  { VecSink s; FrameWriter<VecSink> w(s); w.begin(SETUP_PROBE, 1, 3, 7); w.u8(1); w.u16(300); w.u16(4000); w.u16(50); w.end(); check(json, "setup_probe", s.v); }
  { VecSink s; FrameWriter<VecSink> w(s); w.begin(CALIBRATION_UPDATE, 1, 4, 4); w.u16(200); w.u16(350); w.end(); check(json, "calibration", s.v); }
  { VecSink s; FrameWriter<VecSink> w(s); w.begin(START, 1, 5, 1); w.u8(1); w.end(); check(json, "start", s.v); }
  { VecSink s; FrameWriter<VecSink> w(s); w.begin(STOP, 1, 6, 0); w.end(); check(json, "stop", s.v); }
  { VecSink s; FrameWriter<VecSink> w(s); w.begin(RESET, 1, 7, 0); w.end(); check(json, "reset", s.v); }
  { VecSink s; FrameWriter<VecSink> w(s); w.begin(SERVICE_ADVERTISEMENT, 1, 8, 19);
    w.u8(2); w.u8(1); w.u16(10000); w.u16(0x0100); w.u8(12); w.u8(0); w.u16(150); w.u16(300);
    w.u8(1); w.u16(300); w.u16(4000); w.u16(50); w.end(); check(json, "advert", s.v); }
  { VecSink s; FrameWriter<VecSink> w(s); const int n_pre = 4, n_pt = 10, n_probe = 3;
    w.begin(DATA, 1, 9, 16 + 2 * 2 * n_pt + 2 * 2 * n_probe);
    w.u32(123456); w.u16(10000); w.u8(2); w.u8(FLAG_PROBE_USED); w.u16(n_pre); w.u16(n_pt - n_pre);
    w.u16(n_probe); w.u16(150);
    for (int c = 0; c < 2; c++) for (int i = 0; i < n_pt; i++) w.i16((c + 1) * (i * 37 - 100));
    for (int c = 0; c < 2; c++) for (int i = 0; i < n_probe; i++) w.i16(-(c + 1) * (i * 11 + 5));
    w.end(); check(json, "data", s.v); }
  { VecSink s; FrameWriter<VecSink> w(s); w.begin(LIVE, 1, 10, 8 + 2 * 2 * 5);
    w.u32(5000); w.u16(1000); w.u8(2); w.u8(5);
    for (int c = 0; c < 2; c++) for (int i = 0; i < 5; i++) w.i16((c * 7 + 1) * (i - 2) * 13);
    w.end(); check(json, "live", s.v); }
  { VecSink s; FrameWriter<VecSink> w(s); w.begin(SERVICE_MESSAGE, 1, 11, 4);
    w.u8(CODE_ACK); w.u8(START); w.u8('o'); w.u8('k'); w.end(); check(json, "message", s.v); }
  printf(fails ? "FAILED\n" : "ALL OK\n");
  return fails;
}
