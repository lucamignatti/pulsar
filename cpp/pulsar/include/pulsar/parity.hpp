// Parity harness loader for the Pulsar training system.
//
// Loads the runtime-neutral reference dumps written by parity/dump_reference.py
// (one .npy per record + manifest.json) and compares them against C++ tensors.
// Every rewrite gate (Phases 1-6) uses this to assert numerical parity vs the
// working Python prototype.
#pragma once

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace pulsar::parity {

// A loaded .npy array, flattened to double in C (row-major) order.
struct NpyArray {
  std::vector<std::int64_t> shape;
  std::vector<double> data;  // values upcast to double for dtype-agnostic compare

  std::int64_t numel() const {
    std::int64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
  }
};

namespace detail {

inline std::string read_all(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("parity: cannot open " + path);
  return std::string((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
}

// Parse a numpy .npy v1.0 header. Fills dtype descr (e.g. "<f4"), fortran flag,
// shape, and returns the byte offset where raw data begins.
inline std::size_t parse_header(const std::string& buf, std::string& descr,
                                bool& fortran, std::vector<std::int64_t>& shape) {
  if (buf.size() < 10 || buf.compare(0, 6, "\x93NUMPY") != 0)
    throw std::runtime_error("parity: bad npy magic");
  const unsigned hlen =
      static_cast<unsigned char>(buf[8]) | (static_cast<unsigned char>(buf[9]) << 8);
  const std::size_t header_start = 10;
  const std::string header = buf.substr(header_start, hlen);
  const std::size_t data_start = header_start + hlen;

  auto find_val = [&](const std::string& key) -> std::string {
    auto p = header.find("'" + key + "'");
    if (p == std::string::npos) throw std::runtime_error("parity: npy key " + key);
    p = header.find(':', p) + 1;
    while (p < header.size() && header[p] == ' ') ++p;
    return header.substr(p);
  };

  // descr
  {
    std::string v = find_val("descr");
    auto a = v.find('\''), b = v.find('\'', a + 1);
    descr = v.substr(a + 1, b - a - 1);
  }
  // fortran_order
  fortran = find_val("fortran_order").compare(0, 4, "True") == 0;
  // shape tuple
  {
    std::string v = find_val("shape");
    auto a = v.find('('), b = v.find(')', a);
    std::string inner = v.substr(a + 1, b - a - 1);
    shape.clear();
    std::string num;
    for (char c : inner) {
      if (c == ',' || c == ' ') {
        if (!num.empty()) { shape.push_back(std::stoll(num)); num.clear(); }
      } else {
        num += c;
      }
    }
    if (!num.empty()) shape.push_back(std::stoll(num));
  }
  return data_start;
}

template <typename T>
inline double cast_at(const char* p) {
  T v;
  std::memcpy(&v, p, sizeof(T));
  return static_cast<double>(v);
}

}  // namespace detail

inline NpyArray load_npy(const std::string& path) {
  const std::string buf = detail::read_all(path);
  std::string descr;
  bool fortran = false;
  std::vector<std::int64_t> shape;
  const std::size_t off = detail::parse_header(buf, descr, fortran, shape);
  if (fortran) throw std::runtime_error("parity: fortran-order npy unsupported: " + path);

  NpyArray out;
  out.shape = shape;
  const std::int64_t n = out.numel();
  out.data.resize(static_cast<std::size_t>(n));
  const char* base = buf.data() + off;

  // dtype dispatch: descr like "<f4", "<f8", "<i4", "<i8", "|u1"
  const char kind = descr.size() >= 2 ? descr[1] : '?';
  const int width = descr.size() >= 3 ? descr[2] - '0' : 0;
  for (std::int64_t i = 0; i < n; ++i) {
    const char* p = base + static_cast<std::size_t>(i) * width;
    if (kind == 'f' && width == 4) out.data[i] = detail::cast_at<float>(p);
    else if (kind == 'f' && width == 8) out.data[i] = detail::cast_at<double>(p);
    else if (kind == 'i' && width == 4) out.data[i] = detail::cast_at<std::int32_t>(p);
    else if (kind == 'i' && width == 8) out.data[i] = detail::cast_at<std::int64_t>(p);
    else if (kind == 'u' && width == 1) out.data[i] = detail::cast_at<std::uint8_t>(p);
    else throw std::runtime_error("parity: unsupported dtype " + descr + " in " + path);
  }
  return out;
}

// Max absolute difference between a reference array and a flat C++ buffer.
inline double max_abs_diff(const NpyArray& ref, const double* got, std::int64_t n) {
  if (n != ref.numel())
    throw std::runtime_error("parity: numel mismatch ref=" +
                             std::to_string(ref.numel()) + " got=" + std::to_string(n));
  double m = 0.0;
  for (std::int64_t i = 0; i < n; ++i) {
    const double d = std::abs(ref.data[i] - got[i]);
    if (d > m) m = d;
  }
  return m;
}

}  // namespace pulsar::parity
