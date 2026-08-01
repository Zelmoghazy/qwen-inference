#include <catch2/catch_test_macros.hpp>
#include <unordered_map>
#include <print>
#include "tokenizer.hpp"
#include "simdjson/simdjson.h"

extern const int BYTE_TO_ID[256];
extern const int MERGE_LEFT[], MERGE_RIGHT[], MERGE_RESULT[];
extern const size_t NUM_MERGES;
extern const unsigned char ID_TO_BYTES_BLOB[];
extern const uint32_t ID_TO_BYTES_OFFSET[];
extern const size_t VOCAB_SIZE;

namespace {

struct MergeRule {
    std::string left;
    std::string right;
};

std::string encode_cp(int32_t cp) 
{
    std::string out;
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return out;
}
 
std::unordered_map<uint8_t, std::string> reference_byte_to_unicode() 
{
    std::vector<int> bs;
    for (int i = '!'; i <= '~'; ++i) bs.push_back(i);
    for (int i = 0xA1; i <= 0xAC; ++i) bs.push_back(i);
    for (int i = 0xAE; i <= 0xFF; ++i) bs.push_back(i);
 
    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }
 
    std::unordered_map<uint8_t, std::string> table;
    for (size_t i = 0; i < bs.size(); ++i) {
        table[static_cast<uint8_t>(bs[i])] = encode_cp(cs[i]);
    }
    return table;
}
 
const std::unordered_map<std::string, int64_t>& load_vocab() 
{
    static std::unordered_map<std::string, int64_t> vocab = [] 
    {
        std::unordered_map<std::string, int64_t> v;
 
        simdjson::dom::parser parser;
        simdjson::dom::element doc;
        auto error = parser.load("../test/tokenizer.json").get(doc);
        REQUIRE_FALSE(error);
 
        simdjson::dom::object vocab_obj;
        error = doc["model"]["vocab"].get(vocab_obj);
        REQUIRE_FALSE(error);
 
        for (auto [key, value] : vocab_obj) {
            int64_t id;
            error = value.get(id);
            REQUIRE_FALSE(error);
            v[std::string(key)] = id;
        }
        return v;
    }();
    return vocab;
}
 
const std::vector<MergeRule>& load_merges() 
{
    static std::vector<MergeRule> merges = [] {
        std::vector<MergeRule> m;
 
        simdjson::dom::parser parser;
        simdjson::dom::element doc;
        auto error = parser.load("../test/tokenizer.json").get(doc);
        REQUIRE_FALSE(error);
 
        simdjson::dom::array merges_arr;
        error = doc["model"]["merges"].get(merges_arr);
        REQUIRE_FALSE(error);
 
        for (auto entry : merges_arr) {
            MergeRule rule;
            if (entry.is_array()) {
                simdjson::dom::array pair;
                error = entry.get(pair);
                REQUIRE_FALSE(error);
                auto it = pair.begin();
                std::string_view lv;
                REQUIRE_FALSE((*it).get(lv));
                rule.left = std::string(lv);
                ++it;
                std::string_view rv;
                REQUIRE_FALSE((*it).get(rv));
                rule.right = std::string(rv);
            } else {
                std::string_view s;
                error = entry.get(s);
                REQUIRE_FALSE(error);
                std::string line(s);
                size_t sp = line.find(' ');
                REQUIRE(sp != std::string::npos);
                rule.left = line.substr(0, sp);
                rule.right = line.substr(sp + 1);
            }
            m.push_back(std::move(rule));
        }
        return m;
    }();
    return merges;
}
} 

TEST_CASE("pretokenize output", "[regex][pretokenize]") 
{
    static const std::vector<std::pair<std::string, std::vector<std::string>>> regex_golden = 
    {
        { "Hello, world!", { "Hello", ",", " world", "!" } },
        { "I'm can't won't they've", { "I", "'m", " can", "'t", " won", "'t", " they", "'ve" } },
        { "  leading and trailing spaces  ", { " ", " leading", " and", " trailing", " spaces", "  " } },
        { "line1\nline2\n\nline3", { "line", "1", "\n", "line", "2", "\n\n", "line", "3" } },
        { "中文测试 with English", { "中文测试", " with", " English" } },
        { "日本語のテスト", { "日本語のテスト" } },
        { "한국어 테스트입니다", { "한국어", " 테스트입니다" } },
        { "中英文混合test123混合", { "中英文混合test", "1", "2", "3", "混合" } },
        { "「これは日本語のテストです。」", { "「これは日本語のテストです", "。」" } },
        { "繁體中文測試，包含標點符號。", { "繁體中文測試", "，包含標點符號", "。" } },
        { "مرحبا بالعالم", { "مرحبا", " بالعالم" } },
        { "שלום עולם", { "שלום", " עולם" } },
        { "مرحبا hello مزيج", { "مرحبا", " hello", " مزيج" } },
        { "नमस्ते दुनिया", { "नमस", "्त", "े", " द", "ुन", "िय", "ा" } },
        { "வணக்கம் உலகம்", { "வணக", "்கம", "்", " உலகம", "்" } },
        { "café", { "café" } },
        { "café", { "cafe", "́" } },
        { "naïve résumé", { "naïve", " résumé" } },
        { "Zürich München", { "Zürich", " München" } },
        { "emoji test 😀🎉 mixed", { "emoji", " test", " 😀🎉", " mixed" } },
        { "family: 👨‍👩‍👧‍👦", { "family", ":", " 👨‍👩‍👧‍👦" } },
        { "wave 👋🏽 with skin tone", { "wave", " 👋🏽", " with", " skin", " tone" } },
        { "flags: 🇺🇸🇯🇵🇩🇪", { "flags", ":", " 🇺🇸🇯🇵🇩🇪" } },
        { "keycap: 1️⃣ 2️⃣", { "keycap", ":", " ", "1", "️⃣", " ", "2", "️⃣" } },
        { "don't worry", { "don", "'t", " worry" } },
        { "full　width　space", { "full", "　width", "　space" } },
        { "zero​width​space", { "zero", "​width", "​space" } },
        { "tabs\tand\tmore", { "tabs", "\tand", "\tmore" } },
        { "\n\n\nonly newlines", { "\n\n\n", "only", " newlines" } },
        { "mixed\r\nline\rendings", { "mixed", "\r\n", "line", "\r", "endings" } },
        { "numbers 123 456.789", { "numbers", " ", "1", "2", "3", " ", "4", "5", "6", ".", "7", "8", "9" } },
        { "Arabic-Indic digits ١٢٣٤٥", { "Arabic", "-Indic", " digits", " ", "١", "٢", "٣", "٤", "٥" } },
        { "Devanagari digits १२३४५", { "Devanagari", " digits", " ", "१", "२", "३", "४", "५" } },
        { "full-width digits1234ABC", { "full", "-width", " digits", "1", "2", "3", "4", "ABC" } },
        { "code: def foo():\n    return 1", { "code", ":", " def", " foo", "():\n", "   ", " return", " ", "1" } },
        { "http://example.com/path?query=1&x=2", { "http", "://", "example", ".com", "/path", "?query", "=", "1", "&x", "=", "2" } },
        { "email@example.com", { "email", "@example", ".com" } },
        { "math: 2+2=4, x≠y, α×β", { "math", ":", " ", "2", "+", "2", "=", "4", ",", " x", "≠y", ",", " α", "×β" } },
        { "currency: $100 €50 £30 ¥1000", { "currency", ":", " $", "1", "0", "0", " €", "5", "0", " £", "3", "0", " ¥", "1", "0", "0", "0" } },
        { "punctuation!!!???...", { "punctuation", "!!!???..." } },
        { "CamelCaseWord and snake_case_word", { "CamelCaseWord", " and", " snake", "_case", "_word" } },
        { "multiple    spaces     here", { "multiple", "   ", " spaces", "    ", " here" } },
        { "", {  } },
        { " ", { " " } },
        { "a", { "a" } },
        { "'", { "'" } },
        { "trailing newline at end\n", { "trailing", " newline", " at", " end", "\n" } },
        { "‍", { "‍" } },
    };
    for (const auto& [input, expected] : regex_golden) {
        DYNAMIC_SECTION("input: \"" << input << "\"") {
            auto actual = regex_split(input);
            REQUIRE(actual == expected);
        }
    }
}

TEST_CASE("normalize_nfc", "[unicode][nfc]")
{
    struct Case { std::string input; std::string expected; };

    static const std::vector<Case> cases = {
        { "\u30CF\u309A",         "\u30D1" },
        { "e\u0301\u0323",        "\u1EB9\u0301" },
        { "A\u030A",              "\u00C5" },
        { "\u1112\u1161\u11AB",   "\uD55C" },
        { "caf\u00E9",            "caf\u00E9" },
        { "e\u0301\u0323",        "\u1E19" },
        { "Hello, World!",        "Hello, World!" },
        { "",                     "" },
    };

    for (const auto& [input, expected] : cases) {
        DYNAMIC_SECTION("input: \"" << input << "\"") {
            auto actual = normalize_nfc(input);
            REQUIRE(actual == expected);
        }
    }
}

#if 0
TEST_CASE("Testing the generated lookup table against tokenizer.json", "[bpe][merges]")
{
    const auto& vocab = load_vocab();
    const auto& merges = load_merges();
    
    const auto ref_table = reference_byte_to_unicode();
    REQUIRE(ref_table.size() == 256);

    std::vector<int> ids(BYTE_TO_ID, BYTE_TO_ID + 256);
    std::vector<int> sorted_ids = ids;
    std::sort(sorted_ids.begin(), sorted_ids.end());
    auto last = std::unique(sorted_ids.begin(), sorted_ids.end());
    REQUIRE(std::distance(sorted_ids.begin(), last) == 256);
 
    for (int b = 0; b < 256; ++b) 
    {
        DYNAMIC_SECTION("byte 0x" << std::hex << b << std::dec) 
        {
            const std::string& mapped_char = ref_table.at(static_cast<uint8_t>(b));
 
            auto it = vocab.find(mapped_char);
            INFO("byte " << b << " maps to a stand-in unicode char; "
                 "expected this string to exist in tokenizer.json vocab: \"" << mapped_char << "\"");
            REQUIRE(it != vocab.end());
 
            const int expected_id = static_cast<int>(it->second);
            REQUIRE(BYTE_TO_ID[b] == expected_id);
        }
    }

    REQUIRE(NUM_MERGES == merges.size());

    size_t failures = 0;
    for (size_t rank = 0; rank < merges.size(); ++rank) 
    {
        const std::string& left  = merges[rank].left;
        const std::string& right = merges[rank].right;
        const std::string merged = left + right;
 
        auto lit = vocab.find(left);
        auto rit = vocab.find(right);
        auto mit = vocab.find(merged);
 
        if (lit == vocab.end() || rit == vocab.end() || mit == vocab.end()) {
            INFO("rank " << rank << ": missing vocab entry for \"" << left << "\" + \"" << right << "\" -> \"" << merged << "\"");
            CHECK(false);
            ++failures;
            continue;
        }
 
        const int expected_left   = static_cast<int>(lit->second);
        const int expected_right  = static_cast<int>(rit->second);
        const int expected_result = static_cast<int>(mit->second);
 
        if (MERGE_LEFT[rank] != expected_left || MERGE_RIGHT[rank] != expected_right || MERGE_RESULT[rank] != expected_result) 
        {
            INFO("rank " << rank << " (\"" << left << "\"+\"" << right << "\"): expected ("
                 << expected_left << "," << expected_right << "," << expected_result << ") but baked ("
                 << MERGE_LEFT[rank] << "," << MERGE_RIGHT[rank] << "," << MERGE_RESULT[rank] << ")");
            CHECK(false);
            ++failures;
        }
    }
    INFO("total mismatches: " << failures << " / " << merges.size());
    CHECK(failures == 0);

 
    failures = 0;
    for (size_t rank = 0; rank < merges.size(); ++rank) {
        auto lit = vocab.find(merges[rank].left);
        auto rit = vocab.find(merges[rank].right);
        auto mit = vocab.find(merges[rank].left + merges[rank].right);
 
        if (lit == vocab.end() || rit == vocab.end() || mit == vocab.end()) {
            INFO("rank " << rank << ": one of left/right/merged missing from vocab");
            CHECK(false);
            ++failures;
            continue;
        }
 
        if (!(mit->second > lit->second) || !(mit->second > rit->second)) {
            INFO("rank " << rank << " (\"" << merges[rank].left << "\"+\"" << merges[rank].right
                 << "\"): merged id " << mit->second << " not greater than both operand ids ("
                 << lit->second << ", " << rit->second << ")");
            CHECK(false);
            ++failures;
        }
    }
    INFO("total ordering violations: " << failures << " / " << merges.size());
    CHECK(failures == 0);
} 
#endif

TEST_CASE("qwen2.5 tokenizer matches tiktoken reference", "[tokenizer]")
{
    init_tokenizer();

    SECTION("empty string")
    {
        std::vector<int> expected = {};
        REQUIRE(encode("") == expected);
    }
 
    SECTION("simple ascii")
    {
        std::vector<int> expected = {9707, 11, 1879, 0};
        REQUIRE(encode("Hello, world!") == expected);
    }
 
    SECTION("whitespace runs")
    {
        std::vector<int> expected = {64, 256, 293, 197, 1444, 271, 67};
        REQUIRE(encode("a   b\t\tc\n\nd") == expected);
    }
 
    SECTION("contractions")
    {
        std::vector<int> expected = {40, 2776, 2704, 432, 594, 6915, 11, 807, 3003, 2814, 432, 11, 582, 3278, 1490, 13};
        REQUIRE(encode("I'm sure it's fine, they've done it, we'll see.") == expected);
    }
 
    SECTION("numbers")
    {
        std::vector<int> expected = {785, 1042, 220, 17, 15, 17, 20, 702, 220, 18, 21, 20, 2849, 11, 476, 220, 18, 21, 21, 304, 264, 31471, 1042, 13};
        REQUIRE(encode("The year 2025 has 365 days, or 366 in a leap year.") == expected);
    }
 
    SECTION("punctuation heavy")
    {
        std::vector<int> expected = {11489, 1112, 12555, 25984, 2167, 1177, 902, 1616, 25956};
        REQUIRE(encode("wait...what?! really -- no way :)") == expected);
    }
 
    SECTION("unicode / emoji")
    {
        std::vector<int> expected = {71, 18503, 385, 289, 9416, 507, 220, 108386, 99489, 90316, 145836};
        REQUIRE(encode("héllo wörld 你好世界 😀🚀") == expected);
    }
 
    SECTION("code snippet")
    {
        std::vector<int> expected = {750, 912, 2877, 11, 293, 982, 262, 470, 264, 488, 293, 198};
        REQUIRE(encode("def add(a, b):\n    return a + b\n") == expected);
    }
 
    SECTION("chat template")
    {
        std::vector<int> expected = {151644, 872, 198, 9707, 0, 151645, 198, 151644, 77091, 198};
        REQUIRE(encode("<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n") == expected);
    }
 
    SECTION("tool call")
    {
        std::vector<int> expected = {151657, 4913, 606, 788, 330, 455, 69364, 9207, 151658};
        REQUIRE(encode("<tool_call>{\"name\": \"get_weather\"}</tool_call>") == expected);
    }
 
    SECTION("endoftext")
    {
        std::vector<int> expected = {14801, 151643, 10694};
        REQUIRE(encode("before<|endoftext|>after") == expected);
    }
 
    SECTION("fim tokens")
    {
        std::vector<int> expected = {151659, 750, 15229, 7, 151661, 982, 262, 1494, 151660, 87, 11, 379};
        REQUIRE(encode("<|fim_prefix|>def foo(<|fim_suffix|>):\n    pass<|fim_middle|>x, y") == expected);
    }
 
    SECTION("repeated chars")
    {
        std::vector<int> expected = {69440, 5305};
        REQUIRE(encode("aaaaaaaaaa") == expected);
    }
 
    SECTION("mixed special + normal")
    {
        std::vector<int> expected = {2320, 9934, 13, 151644, 8948, 198, 2610, 525, 10950, 13, 151645};
        REQUIRE(encode("System prompt.<|im_start|>system\nYou are helpful.<|im_end|>") == expected);
    }

    SECTION("arabic simple")
    {
        std::vector<int> expected = {124122, 29825, 124671, 124476, 129634};
        REQUIRE(encode("مرحبا بالعالم") == expected);
    }

    SECTION("arabic sentence")
    {
        std::vector<int> expected = {16157, 124169, 136837, 124422, 56794, 138230, 137057, 129071, 13};
        REQUIRE(encode("هذا اختبار لنموذج اللغة العربية.") == expected);
    }

    SECTION("arabic with numbers")
    {
        std::vector<int> expected = {31382, 137426, 220, 17, 15, 17, 20, 125998, 124006, 220, 18, 21, 20, 128587, 5703, 13};
        REQUIRE(encode( "السنة 2025 لديها 365 يوما.") == expected);
    }

    SECTION("arabic mixed english")
    {
        std::vector<int> expected = {9707, 23364, 126860, 124671, 1879, 0};
        REQUIRE(encode("Hello مرحبا world!") == expected);
    }
}