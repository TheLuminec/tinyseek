#include "tinyseek_core/kv.hpp"

namespace tinyseek {

std::string trim(const std::string &s) {
  const char *ws = " \t\r";
  const size_t a = s.find_first_not_of(ws);
  if (a == std::string::npos) return std::string();
  const size_t b = s.find_last_not_of(ws);
  return s.substr(a, b - a + 1);
}

std::vector<std::string> text_lines(const std::string &text) {
  std::vector<std::string> lines;
  size_t pos = 0;
  // A byte-order mark (Windows Notepad's "UTF-8 with BOM") would otherwise become part
  // of the first key.
  if (text.size() >= 3 && static_cast<unsigned char>(text[0]) == 0xEF &&
      static_cast<unsigned char>(text[1]) == 0xBB && static_cast<unsigned char>(text[2]) == 0xBF) {
    pos = 3;
  }
  while (pos < text.size()) {
    size_t end = text.find('\n', pos);
    if (end == std::string::npos) end = text.size();
    const std::string line = trim(text.substr(pos, end - pos));
    pos = end + 1;
    if (!line.empty() && line[0] != '#') lines.push_back(line);
  }
  return lines;
}

std::map<std::string, std::string> parse_kv(const std::string &text) {
  std::map<std::string, std::string> kv;
  for (const std::string &line : text_lines(text)) {
    const size_t eq = line.find('=');
    if (eq == std::string::npos) continue;
    const std::string key = trim(line.substr(0, eq));
    if (!key.empty()) kv[key] = trim(line.substr(eq + 1));  // a later line wins
  }
  return kv;
}

}  // namespace tinyseek
