// Copyright © 2023-2024 Apple Inc.
#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

// Used by pread implementation.
#ifdef _WIN32
#ifdef _MSC_VER
#define NOMINMAX
#endif
#include <windows.h>
#endif // _WIN32

#include "mlx/io/load.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

// Adapted from
// https://github.com/angeloskath/supervised-lda/blob/master/include/ldaplusplus/NumpyFormat.hpp

namespace mlx::core {

namespace {

constexpr uint8_t MAGIC[] = {
    0x93,
    0x4e,
    0x55,
    0x4d,
    0x50,
    0x59,
};

inline bool is_big_endian() {
  union ByteOrder {
    int32_t i;
    uint8_t c[4];
  };
  ByteOrder b = {0x01234567};

  return b.c[0] == 0x01;
}

// Array protocol typestring for Dtype
std::string dtype_to_array_protocol(const Dtype& t) {
  std::ostringstream r;
  if (size_of(t) > 1) {
    r << (is_big_endian() ? ">" : "<");
  } else {
    r << "|";
  }
  r << kindof(t) << (int)size_of(t);
  return r.str();
}

// Dtype from array protocol type string
Dtype dtype_from_array_protocol(std::string_view t) {
  if (t.length() == 2 || t.length() == 3) {
    std::string_view r = t.length() == 3 ? t.substr(1, 2) : t;

    if (r == "V2") {
      return bfloat16;
    }

    uint8_t size = r[1] - '0';

    switch (r[0]) {
      case 'b': {
        if (size == 1)
          return bool_;
      }
      case 'i': {
        if (size == 1)
          return int8;
        else if (size == 2)
          return int16;
        else if (size == 4)
          return int32;
        else if (size == 8)
          return int64;
      }
      case 'u': {
        if (size == 1)
          return uint8;
        else if (size == 2)
          return uint16;
        else if (size == 4)
          return uint32;
        else if (size == 8)
          return uint64;
      }
      case 'f': {
        if (size == 2)
          return float16;
        else if (size == 4)
          return float32;
      }
      case 'c': {
        return complex64;
      }
    }
  }

  throw std::invalid_argument(
      "[from_str] Invalid array protocol type-string: " + std::string(t));
}

#ifdef _WIN32
// There is no pread on Windows, emulate it with ReadFile.
int64_t pread(int fd, void* buf, uint64_t size, uint64_t offset) {
  HANDLE file = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
  if (file == INVALID_HANDLE_VALUE) {
    return -1;
  }

  OVERLAPPED overlapped = {0};
  overlapped.Offset = offset & 0xFFFFFFFF;
  overlapped.OffsetHigh = (offset >> 32) & 0xFFFFFFFF;

  DWORD bytes_read;
  if (!ReadFile(file, buf, size, &bytes_read, &overlapped)) {
    if (GetLastError() != ERROR_HANDLE_EOF) {
      return -1;
    }
  }

  return bytes_read;
}
#endif

} // namespace

/** Save array to out stream in .npy format */
void save(std::shared_ptr<io::Writer> out_stream, array a) {
  ////////////////////////////////////////////////////////
  // Check array

  a = contiguous(a, true);
  a.eval();

  if (a.nbytes() == 0) {
    throw std::invalid_argument("[save] cannot serialize an empty array");
  }

  ////////////////////////////////////////////////////////
  // Check file
  if (!out_stream->good() || !out_stream->is_open()) {
    throw std::runtime_error("[save] Failed to open " + out_stream->label());
  }

  ////////////////////////////////////////////////////////
  // Prepare header
  std::ostringstream magic_ver_len;
  magic_ver_len.write(reinterpret_cast<const char*>(MAGIC), 6);

  std::string fortran_order = a.flags().col_contiguous ? "True" : "False";
  std::ostringstream header;
  header << "{'descr': '" << dtype_to_array_protocol(a.dtype()) << "',"
         << " 'fortran_order': " << fortran_order << "," << " 'shape': (";
  for (auto i : a.shape()) {
    header << i << ", ";
  }
  header << ")}";

  size_t header_len = static_cast<size_t>(header.tellp());
  bool is_v1 = header_len + 15 < std::numeric_limits<uint16_t>::max();

  // Pad out magic + version + header_len + header + \n to be divisible by 16
  size_t padding = (6 + 2 + (2 + 2 * is_v1) + header_len + 1) % 16;

  header << std::string(padding, ' ') << '\n';

  if (is_v1) {
    magic_ver_len << (char)0x01 << (char)0x00;

    uint16_t v1_header_len = header.tellp();
    const char* len_bytes = reinterpret_cast<const char*>(&v1_header_len);

    if (!is_big_endian()) {
      magic_ver_len.write(len_bytes, 2);
    } else {
      magic_ver_len.write(len_bytes + 1, 1);
      magic_ver_len.write(len_bytes, 1);
    }
  } else {
    magic_ver_len << (char)0x02 << (char)0x00;

    uint32_t v2_header_len = header.tellp();
    const char* len_bytes = reinterpret_cast<const char*>(&v2_header_len);

    if (!is_big_endian()) {
      magic_ver_len.write(len_bytes, 4);
    } else {
      magic_ver_len.write(len_bytes + 3, 1);
      magic_ver_len.write(len_bytes + 2, 1);
      magic_ver_len.write(len_bytes + 1, 1);
      magic_ver_len.write(len_bytes, 1);
    }
  }
  ////////////////////////////////////////////////////////
  // Serialize array

  out_stream->write(magic_ver_len.str().c_str(), magic_ver_len.str().length());
  out_stream->write(header.str().c_str(), header.str().length());
  out_stream->write(a.data<char>(), a.nbytes());
}

/** Save array to file in .npy format */
void save(std::string file, array a) {
  // Add .npy to file name if it is not there
  if (file.length() < 4 || file.substr(file.length() - 4, 4) != ".npy")
    file += ".npy";

  // Serialize array
  save(std::make_shared<io::FileWriter>(std::move(file)), a);
}

/** Load array from reader in .npy format */
array load(std::shared_ptr<io::Reader> in_stream, StreamOrDevice s) {
  ////////////////////////////////////////////////////////
  // Open and check file
  if (!in_stream->good() || !in_stream->is_open()) {
    throw std::runtime_error("[load] Failed to open " + in_stream->label());
  }

  ////////////////////////////////////////////////////////
  // Read header and prepare array details

  // Read and check magic
  char read_magic_and_ver[8];
  in_stream->read(read_magic_and_ver, 8);
  if (std::memcmp(read_magic_and_ver, MAGIC, 6) != 0) {
    throw std::runtime_error("[load] Invalid header in " + in_stream->label());
  }

  // Read and check version
  if (read_magic_and_ver[6] != 1 && read_magic_and_ver[6] != 2) {
    throw std::runtime_error(
        "[load] Unsupported npy format version in " + in_stream->label());
  }

  // Read header len and header
  int header_len_size = read_magic_and_ver[6] == 1 ? 2 : 4;
  size_t header_len;

  if (header_len_size == 2) {
    uint16_t v1_header_len;
    in_stream->read(reinterpret_cast<char*>(&v1_header_len), header_len_size);
    header_len = v1_header_len;
  } else {
    uint32_t v2_header_len;
    in_stream->read(reinterpret_cast<char*>(&v2_header_len), header_len_size);
    header_len = v2_header_len;
  }

  // Read the header
  std::vector<char> buffer(header_len + 1);
  in_stream->read(&buffer[0], header_len);
  buffer[header_len] = 0;
  std::string header(&buffer[0]);

  // Read data type from header
  std::string dtype_str = header.substr(11, 3);
  bool read_is_big_endian = dtype_str[0] == '>';
  Dtype dtype = dtype_from_array_protocol(dtype_str);

  // Read contiguity order
  bool col_contiguous = header[34] == 'T';

  // Read array shape from header
  Shape shape;

  size_t st = header.find_last_of('(') + 1;
  size_t ed = header.find_last_of(')');
  std::string shape_str = header.substr(st, ed - st);

  while (!shape_str.empty()) {
    // Read current number and get position of comma
    size_t pos;
    int dim = std::stoi(shape_str, &pos);
    shape.push_back(dim);

    // Skip the comma and space and read the next number
    if (pos + 2 <= shape_str.length())
      shape_str = shape_str.substr(pos + 2);
    else {
      shape_str = shape_str.substr(pos);
      if (!shape_str.empty() && shape_str != " " && shape_str != ",") {
        throw std::runtime_error(
            "[load] Unknown error while parsing header in " +
            in_stream->label());
      }
      shape_str = "";
    }
  }

  ////////////////////////////////////////////////////////
  // Build primitive

  size_t offset = 8 + header_len_size + header.length();
  bool swap_endianness = read_is_big_endian != is_big_endian();

  if (col_contiguous) {
    std::reverse(shape.begin(), shape.end());
  }
  auto loaded_array = array(
      shape,
      dtype,
      std::make_shared<Load>(to_stream(s), in_stream, offset, swap_endianness),
      std::vector<array>{});
  if (col_contiguous) {
    loaded_array = transpose(loaded_array, s);
  }

  return loaded_array;
}

/** Load array from file in .npy format */
array load(std::string file, StreamOrDevice s) {
  return load(std::make_shared<io::ParallelFileReader>(std::move(file)), s);
}

namespace io {

ThreadPool& thread_pool() {
  static ThreadPool pool_{4};
  return pool_;
}

ThreadPool ParallelFileReader::thread_pool_{4};

void ParallelFileReader::read(char* data, size_t n) {
  while (n != 0) {
    auto m = ::read(fd_, data, std::min(n, static_cast<size_t>(INT32_MAX)));
    if (m <= 0) {
      std::ostringstream msg;
      msg << "[read] Unable to read " << n << " bytes from file.";
      throw std::runtime_error(msg.str());
    }
    data += m;
    n -= m;
  }
}

void ParallelFileReader::read(char* data, size_t n, size_t offset) {
  auto readfn = [fd = fd_](size_t offset, size_t size, char* buffer) -> bool {
    while (size != 0) {
      auto m = pread(fd, buffer, size, offset);
      if (m <= 0) {
        return false;
      }
      buffer += m;
      size -= m;
    }
    return true;
  };
  std::vector<std::future<bool>> futs;
  while (n != 0) {
    if (n < batch_size_) {
      if (!readfn(offset, n, data)) {
        throw std::runtime_error("[read] Unable to read from file.");
      }
      break;
    } else {
      size_t m = batch_size_;
      futs.emplace_back(thread_pool_.enqueue(readfn, offset, m, data));
      data += m;
      n -= m;
      offset += m;
    }
  }
  for (auto& f : futs) {
    if (!f.get()) {
      throw std::runtime_error("[read] Unable to read from file.");
    }
  }
}

std::string read_digits(std::istream& s) {
  std::string num = "";
  char ch = s.get();
  while (std::isdigit(ch) || ch == '-' || ch == '.' || ch == 'e' || ch == 'E') {
    num += ch;
    ch = s.get();
  }
  s.seekg(-1, std::ios::cur);
  return num;
}

json parse_json_number(std::istream& s) {
  auto num = read_digits(s);
  if (num.find_first_of(".eE") != std::string::npos) {
    return json(std::stod(num));
  } else {
    return json(std::stol(num));
  }
}

std::string parse_json_string(std::istream& s) {
  bool in_escape = false;
  std::string str = "";
  char ch = s.get();
  while (ch != '"' || in_escape) {
    if (in_escape) {
      if (ch == '"' || ch == '\\' || ch == '/') {
        str += ch;
      } else if (ch == 'b') {
        str += '\b';
      } else if (ch == 'f') {
        str += '\f';
      } else if (ch == 'n') {
        str += '\n';
      } else if (ch == 'r') {
        str += '\r';
      } else if (ch == 't') {
        str += '\t';
      } else if (ch == 'u') {
        throw std::invalid_argument(
            "[json] Unicode decoding not yet supported.");
      } else {
        throw std::invalid_argument("[json] Invalid escape sequence.");
      }
      in_escape = false;
    } else if (ch == '\\') {
      in_escape = true;
    } else {
      str += ch;
    }

    ch = s.get();
    if (s.eof()) {
      throw std::invalid_argument("[json] Unfinished string value.");
    }
  }
  return str;
}

json parse_json_helper(std::istream& s) {
  char ch;
  s >> std::ws >> ch;
  // object
  if (ch == '{') {
    json_object object;
    while (true) {
      s >> std::ws >> ch;
      if (ch == '}') {
        break;
      } else if (ch != '"') {
        throw std::invalid_argument("[json] Invalid json: expected '\"'.");
      }
      std::string key = parse_json_string(s);
      s >> std::ws >> ch;
      if (ch != ':') {
        throw std::invalid_argument("[json] Invalid json: expected '\"'.");
      }
      json value = parse_json_helper(s);
      object[key] = value;

      s >> std::ws >> ch;
      if (ch == '}') {
        break;
      } else if (ch != ',') {
        throw std::invalid_argument("[json] Invalid json: expected ','.");
      }
    }
    return object;
    // array
  } else if (ch == '[') {
    json_array array;
    s >> std::ws;
    while (true) {
      if (s.peek() == ']') {
        s.get();
        break;
      }
      json value = parse_json_helper(s);
      array.push_back(value);
      s >> std::ws >> ch;
      if (ch == ']') {
        break;
      } else if (ch != ',') {
        throw std::invalid_argument("[json] Invalid json: expected ','.");
      }
    }
    return array;
    // null
  } else if (ch == 'n') {
    s.seekg(3, std::ios::cur);
    return json(nullptr);
    // true
  } else if (ch == 't') {
    s.seekg(3, std::ios::cur);
    return json(true);
    // false
  } else if (ch == 'f') {
    s.seekg(4, std::ios::cur);
    return json(false);
    // string
  } else if (ch == '"') {
    return json(parse_json_string(s));
    // number
  } else if (ch == '-' || std::isdigit(ch)) {
    s.seekg(-1, std::ios::cur);
    return parse_json_number(s);
  } else {
    throw std::invalid_argument("[json] Invalid json: Unrecognized value.");
  }
}

json parse_json(std::istream& s, bool allow_extra) {
  json result = parse_json_helper(s);
  s.get();
  if (!s.eof() && !allow_extra) {
    throw std::invalid_argument(
        "[json] json finished before the end of the stream."
        " Pass `allow_extra` to allow this.");
  }
  return result;
}

json parse_json(const std::string& s, bool allow_extra) {
  // TODO: remove this copy
  std::istringstream stream(s);
  return parse_json(stream, allow_extra);
}

void apply_indent(std::ostream& os, int indent) {
  for (int i = 0; i < indent; i++) {
    os << " ";
  }
}

void print_json(std::ostream& os, const json& obj, int indent) {
  os << std::boolalpha;
  if (obj.is<json_array>()) {
    os << "[" << std::endl;
    bool first = true;
    for (const json& val : obj) {
      if (!first) {
        os << ",";
        os << std::endl;
      }
      first = false;
      apply_indent(os, indent + 2);
      print_json(os, val, indent + 2);
    }
    os << std::endl;
    apply_indent(os, indent);
    os << "]";
  } else if (obj.is<json_object>()) {
    os << "{" << std::endl;
    bool first = true;
    for (const auto& [key, val] : obj.items()) {
      if (!first) {
        os << ",";
        os << std::endl;
      }
      first = false;
      apply_indent(os, indent + 2);
      os << '"' << key << '"' << ": ";
      print_json(os, val, indent + 2);
    }
    os << std::endl;
    apply_indent(os, indent);
    os << "}";
  } else if (obj.is<double>()) {
    double val = obj;
    os << val;
  } else if (obj.is<long>()) {
    long val = obj;
    os << val;
  } else if (obj.is<bool>()) {
    bool val = obj;
    os << val;
  } else if (obj.is<std::string>()) {
    std::string val = obj;
    // Escape special string characters
    const std::vector<std::pair<char, std::string>> special_chars = {
        {'\\', "\\\\"},
        {'"', "\\\""},
        {'/', "\\/"},
        {'\b', "\\b"},
        {'\f', "\\f"},
        {'\n', "\\n"},
        {'\r', "\\r"},
        {'\t', "\\t"},
    };
    for (const auto& [ch, new_str] : special_chars) {
      int pos = -1;
      while ((pos = val.find(ch, pos + new_str.length())) !=
             std::string::npos) {
        val.replace(pos, 1, new_str);
      }
    }
    os << '"' << val << '"';
  } else if (obj.is<std::nullptr_t>()) {
    os << "null";
  }
}

std::ostream& operator<<(std::ostream& os, const json& obj) {
  print_json(os, obj, 0);
  return os;
}

} // namespace io

} // namespace mlx::core
