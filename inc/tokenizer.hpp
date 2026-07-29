#pragma once
#include <string>
#include <vector>

#define PCRE2_CODE_UNIT_WIDTH 8
#define PCRE2_STATIC
#include <pcre2/pcre2.h>

std::vector<std::string> regex_split(const std::string& text);