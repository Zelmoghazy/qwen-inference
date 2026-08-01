#include <unordered_map>
#include <vector>
#include <stdexcept>
#include "tokenizer.hpp"

extern const size_t VOCAB_SIZE;
extern const size_t NUM_MERGES;
extern const int BYTE_TO_ID[256];
extern const int MERGE_LEFT[], MERGE_RIGHT[], MERGE_RESULT[];
extern const unsigned char ID_TO_BYTES_BLOB[];
extern const uint32_t ID_TO_BYTES_OFFSET[];

const char* regex_pattern = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}" "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

std::vector<std::string> special_tokens = {
    "<|endoftext|>", "<|im_start|>","<|im_end|>","<|object_ref_start|>","<|object_ref_end|>","<|box_start|>","<|box_end|>","<|quad_start|>","<|quad_end|>","<|vision_start|>","<|vision_end|>","<|vision_pad|>","<|image_pad|>","<|video_pad|>", "<tool_call>","</tool_call>","<|fim_prefix|>","<|fim_middle|>","<|fim_suffix|>","<|fim_pad|>","<|repo_name|>","<|file_sep|>"
};

int special_token_to_id[] = {
    151643, 151644, 151645, 151646, 151647, 151648, 151649, 151650, 
    151651, 151652, 151653, 151654, 151655, 151656, 151657, 151658, 
    151659, 151660, 151661, 151662, 151663, 151664
};


struct MergeRule { int rank; int result_id; };
enum class SegmentType
{
    Normal,
    Special
};

struct Segment
{
    SegmentType type;
    int    id;
    size_t start;
    size_t end;
};

std::unordered_map<uint64_t, MergeRule> g_merge_lookup;

std::string normalize_nfc(const std::string& input)
{
    utf8proc_uint8_t* normalized = utf8proc_NFC(
        reinterpret_cast<const utf8proc_uint8_t*>(input.c_str())
    );

    if (!normalized)
        throw std::runtime_error("UTF-8 normalization failed");

    std::string result(
        reinterpret_cast<const char*>(normalized)
    );

    free(normalized);

    return result;
}

std::vector<std::string> regex_split(const std::string& text) 
{
    int errorcode;
    PCRE2_SIZE erroffset;
    pcre2_code* re = pcre2_compile(reinterpret_cast<PCRE2_SPTR>(regex_pattern), 
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

std::vector<Segment> special_split(const std::string& text)
{
    const size_t n = text.size();

    std::vector<Segment> segments;

    size_t pos = 0;

    while(pos < n)
    {
        size_t best_pos = std::string::npos;
        size_t best_len = 0;
        int    best_idx = -1;
        /*
            Find the special token at the earliest position 
         */
        for (size_t k = 0; k < special_tokens.size(); ++k)
        {
            const auto& token = special_tokens[k];
            size_t found = text.find(token, pos);

            if (found == std::string::npos){
                continue;
            } 

            if (found < best_pos)
            {
                best_pos = found;
                best_len = token.size();
                best_idx = (int)special_token_to_id[k];
            }
        }

        // Last special token, rest of the text is normal
        if (best_pos == std::string::npos){
            segments.push_back({SegmentType::Normal, -1, pos, text.size()});
            break;
        }

        // normal segment before the special token
        if (best_pos > pos){
            segments.push_back({SegmentType::Normal, -1, pos, best_pos});
        }

        segments.push_back({SegmentType::Special, best_idx, best_pos, best_pos + best_len});

        pos = best_pos + best_len;
    }

    return segments;
}

inline uint64_t pack(int a, int b) 
{
    return (uint64_t(uint32_t(a)) << 32) | uint32_t(b);
}

void init_tokenizer(void) 
{
    g_merge_lookup.reserve(NUM_MERGES * 2);
    for (size_t i = 0; i < NUM_MERGES; ++i) {
        g_merge_lookup[pack(MERGE_LEFT[i], MERGE_RIGHT[i])] = { (int)i, MERGE_RESULT[i] };
    }
}

/*
    We look at every adjacent pair of current tokens
    for every pair look for the one that has the lowes rank
    replace them with the merged result
    repeat
 */
std::vector<int> bpe_merge_ids(std::vector<int> ids) 
{
    if (ids.size() < 2) return ids;

    while (true) 
    {
        int best_rank = INT_MAX;
        size_t best_i = SIZE_MAX;
        int best_result = -1;

        for (size_t i = 0; i + 1 < ids.size(); ++i) 
        {
            auto it = g_merge_lookup.find(pack(ids[i], ids[i+1]));
            if (it != g_merge_lookup.end() && it->second.rank < best_rank) 
            {
                best_rank = it->second.rank;
                best_i = i;
                best_result = it->second.result_id;
            }
        }
        if (best_i == SIZE_MAX) break;

        ids[best_i] = best_result;
        ids.erase(ids.begin() + best_i + 1);
    }
    return ids;
}

std::vector<int> encode_piece(const std::string& raw_utf8_piece) 
{
    std::vector<int> ids;
    ids.reserve(raw_utf8_piece.size());
    for (unsigned char b : raw_utf8_piece) {
        ids.push_back(BYTE_TO_ID[b]);   
    }
    return bpe_merge_ids(ids);
}

std::string decode_id(int id) 
{
    uint32_t start = ID_TO_BYTES_OFFSET[id];
    uint32_t end   = ID_TO_BYTES_OFFSET[id+1];
    return std::string(reinterpret_cast<const char*>(ID_TO_BYTES_BLOB + start), end - start);
}

std::vector<int> encode(const std::string& text)
{
    std::vector<int> out;
    std::vector<Segment> segments = special_split(text);
    for (auto& seg : segments)
    {
        if (seg.type == SegmentType::Special)
        {
            out.push_back(seg.id);
        }
        else
        {
            std::string normal = normalize_nfc(text.substr(seg.start, seg.end - seg.start));
            for (auto& piece : regex_split(normal)) 
            {
                auto ids = encode_piece(piece);
                out.insert(out.end(), ids.begin(), ids.end());
            }
        }
    }
    return out;
}