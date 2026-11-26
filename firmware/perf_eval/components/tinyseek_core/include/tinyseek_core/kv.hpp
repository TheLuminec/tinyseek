// key=value text files (bench.txt, config.txt, model sidecars), parsed the same way on the
// board and in the host tests (portable C++, no ESP headers).
//
// Lines: a UTF-8 byte-order mark at the start of the file is dropped; each line is trimmed
// of spaces, tabs and CR at both ends; empty lines and lines whose first character is '#'
// are skipped. key=value: split at the first '=', key and value trimmed again. A key that
// appears twice keeps its last value, so a line added at the end of the file wins.
// tinyseek.deploy.layout.read_kv on the PC follows the same rules.
#pragma once

#include <map>
#include <string>
#include <vector>

namespace tinyseek {

// " \t\r" stripped at both ends.
std::string trim(const std::string &s);

// The meaningful lines of a text file (see above).
std::vector<std::string> text_lines(const std::string &text);

// key=value pairs; lines without '=' or with an empty key are ignored.
std::map<std::string, std::string> parse_kv(const std::string &text);

}  // namespace tinyseek
