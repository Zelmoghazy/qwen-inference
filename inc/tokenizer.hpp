#pragma once
#include <string>
#include <vector>

#define PCRE2_CODE_UNIT_WIDTH 8
#define PCRE2_STATIC
#include <pcre2/pcre2.h>

#define UTF8PROC_STATIC
#include <utf8proc/utf8proc.h>


std::string decode_id(int id);
std::vector<int> encode(const std::string& text);
void init_tokenizer(void);

std::string normalize_nfc(const std::string& input);
std::vector<std::string> regex_split(const std::string& text) ;
std::string decode_id(int id); 
