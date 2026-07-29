#include <vector>
#include <stdexcept>
#include "tokenizer.hpp"

constexpr const char* PATTERN =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}"
    "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

std::vector<std::string> regex_split(const std::string& text) 
{
    int errorcode;
    PCRE2_SIZE erroffset;
    pcre2_code* re = pcre2_compile(reinterpret_cast<PCRE2_SPTR>(PATTERN), 
                                  PCRE2_ZERO_TERMINATED,
                                  PCRE2_UTF | PCRE2_UCP,
                                  &errorcode, &erroffset, nullptr);
    if (!re) 
    {
        PCRE2_UCHAR buf[256];
        pcre2_get_error_message(errorcode, buf, sizeof(buf));
        throw std::runtime_error("regex compile failed: " + std::string(reinterpret_cast<char*>(buf)));
    }

    pcre2_match_data* match_data = pcre2_match_data_create_from_pattern(re, nullptr);

    std::vector<std::string> pieces;
    PCRE2_SIZE offset = 0;
    PCRE2_SIZE len = text.size();

    while (offset <= len) 
    {
        int rc = pcre2_match(re, reinterpret_cast<PCRE2_SPTR>(text.data()), len,
                             offset, 0, match_data, nullptr);

        if (rc < 0) break;

        PCRE2_SIZE* ov = pcre2_get_ovector_pointer(match_data);
        PCRE2_SIZE m_start = ov[0];
        PCRE2_SIZE m_end = ov[1];

        if (m_end == m_start) {
            offset = m_start + 1;
            continue;
        }

        pieces.push_back(text.substr(m_start, m_end - m_start));
        offset = m_end;
    }

    pcre2_match_data_free(match_data);
    return pieces;
}
