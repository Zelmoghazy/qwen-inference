#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>


#include "tokenizer.hpp"
#include "simdjson/simdjson.h"

#include "utils.hpp"
#include "gguf.hpp"
#include "engine.hpp"

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

using Catch::Matchers::WithinAbs;
constexpr float EPS = 1e-5f;

static void check_slice(const float *x, int start, const std::vector<float> &expected)
{
    for (size_t i = 0; i < expected.size(); ++i)
    {
        INFO("dim " << (start + i));
        REQUIRE_THAT(x[start + i], WithinAbs(expected[i], EPS));
    }
}

struct EmbedCase
{
    int token_id;
    const char *piece;              // for readability / failure messages only
    std::vector<float> first16;     // [0:16]
    std::vector<float> mid_a;       // [16:24]
    std::vector<float> mid_b;       // [1020:1028]
    std::vector<float> mid_c;       // [2024:2032]
    std::vector<float> last16;      // [2032:2048]
};

static const std::vector<EmbedCase> kCases = {
    {71, "h",
     {-0.003296f, -0.000942f, 0.004709f, 0.000000f, 0.002354f, -0.017893f, -0.002354f, 0.009418f,
      -0.008947f, 0.018364f, 0.004238f, -0.041908f, -0.043792f, -0.008005f, 0.009418f, -0.008947f},
     {-0.027311f, 0.031078f, 0.009888f, 0.021660f, -0.044733f, -0.019777f, 0.059801f, 0.027782f},
     {0.007947f, 0.034233f, -0.044625f, 0.000611f, -0.003614f, -0.014006f, -0.039307f, 0.015361f},
     {-0.008492f, -0.019106f, -0.014860f, 0.016983f, 0.012737f, 0.014153f, 0.035381f, 0.002123f},
     {0.026890f, -0.007784f, 0.058733f, -0.001415f, 0.007784f, -0.029013f, -0.024059f, 0.002123f,
      0.045288f, -0.004953f, 0.038919f, 0.005661f, -0.031136f, 0.000708f, 0.049534f, 0.089869f}},

    {18503, "\xC3\xA9l", // "él"
     {0.007401f, -0.018839f, -0.006055f, -0.022203f, 0.040369f, -0.002018f, -0.036332f, 0.014802f,
      0.008074f, 0.010092f, -0.024221f, 0.029604f, 0.004037f, 0.000000f, -0.053153f, -0.015475f},
     {-0.002691f, -0.085448f, -0.018839f, -0.007401f, -0.040369f, 0.020185f, 0.016820f, -0.003364f},
     {0.019415f, -0.038830f, -0.054649f, 0.000000f, 0.003276f, -0.030575f, -0.001092f, 0.037127f},
     {-0.028770f, -0.065909f, -0.029293f, -0.019877f, 0.033478f, 0.019877f, -0.027201f, 0.021447f},
     {-0.046555f, -0.040801f, -0.032432f, 0.012554f, -0.016216f, 0.066432f, -0.027724f, -0.006277f,
      -0.038709f, -0.003139f, -0.038709f, -0.051786f, -0.026155f, 0.019354f, -0.014123f, -0.031909f}},

    {385, "lo",
     {-0.007377f, -0.005796f, 0.015280f, 0.026345f, -0.010011f, -0.010538f, 0.033195f, -0.006323f,
      0.010538f, -0.056906f, 0.032141f, 0.011592f, 0.001581f, 0.036883f, -0.008430f, 0.015807f},
     {-0.030560f, 0.000000f, 0.006323f, -0.040045f, 0.019495f, -0.066917f, -0.061121f, 0.001581f},
     {0.011470f, -0.051312f, 0.028373f, 0.002415f, -0.016861f, -0.030746f, 0.030746f, -0.013390f},
     {0.023246f, -0.012159f, 0.012517f, 0.034332f, 0.040412f, 0.005007f, -0.031114f, 0.010729f},
     {-0.015378f, -0.001073f, -0.004649f, 0.010371f, -0.033617f, 0.023246f, 0.002503f, -0.006080f,
      -0.037551f, 0.019670f, -0.011444f, -0.013947f, 0.009656f, 0.021815f, 0.029683f, 0.002146f}},

    {289, " w",
     {-0.005224f, 0.027071f, -0.018047f, -0.005224f, 0.032295f, -0.020897f, 0.016623f, -0.011873f,
      0.033245f, 0.008549f, 0.016623f, 0.056042f, -0.019472f, -0.056042f, -0.037519f, -0.021847f},
     {-0.025171f, -0.011398f, -0.002375f, 0.005224f, -0.005224f, 0.019472f, -0.060316f, 0.026121f},
     {0.030197f, 0.002745f, -0.037975f, -0.035687f, 0.000000f, 0.016705f, 0.001576f, -0.010401f},
     {0.010692f, 0.043165f, 0.020593f, 0.023365f, -0.017425f, 0.006336f, -0.005148f, -0.024553f},
     {-0.006732f, 0.034849f, -0.003168f, -0.005544f, 0.012276f, -0.050294f, -0.000396f, 0.018613f,
      -0.012672f, -0.030493f, 0.029701f, -0.002772f, 0.003168f, -0.013860f, 0.014256f, 0.002376f}},

    {9416, "\xC3\xB6r", // "ör"
     {0.004429f, -0.000492f, 0.010826f, 0.003445f, 0.019192f, 0.015747f, -0.007381f, -0.011318f,
      -0.014271f, 0.020668f, 0.008858f, 0.012302f, -0.046749f, -0.027065f, 0.028049f, -0.046257f},
     {-0.015747f, -0.003445f, -0.012794f, 0.027065f, 0.010826f, -0.016731f, -0.001968f, -0.017715f},
     {0.008303f, 0.001384f, -0.035978f, 0.004843f, 0.011542f, 0.033577f, -0.005246f, -0.044420f},
     {0.016472f, 0.008236f, -0.021317f, 0.059105f, 0.061527f, -0.027130f, -0.007751f, -0.006298f},
     {0.021317f, -0.019863f, 0.022285f, 0.001938f, -0.005329f, 0.016956f, 0.032459f, -0.051838f,
      -0.013081f, 0.003391f, -0.020832f, -0.002422f, 0.024223f, 0.019863f, -0.014050f, 0.013081f}},

    {507, "ld",
     {0.004382f, -0.016652f, 0.037248f, 0.005259f, -0.055653f, 0.011394f, 0.051709f, 0.024102f,
      0.014899f, -0.050833f, 0.012270f, 0.023225f, -0.034619f, -0.039877f, -0.030237f, 0.031551f},
     {-0.010955f, 0.019720f, -0.024540f, 0.018405f, 0.010955f, -0.016652f, -0.023664f, 0.017090f},
     {0.037369f, 0.040137f, -0.017992f, -0.024451f, 0.001996f, -0.029933f, -0.002661f, 0.035255f},
     {-0.001199f, 0.044781f, -0.013194f, 0.026389f, 0.014794f, 0.030387f, 0.010795f, 0.019991f},
     {-0.011595f, -0.050778f, -0.025189f, 0.004798f, -0.017992f, 0.013994f, -0.007197f, 0.004798f,
      0.019592f, 0.023990f, -0.007197f, -0.007997f, 0.025589f, -0.019592f, 0.027588f, -0.015993f}},

    {220, " ",
     {0.040754f, 0.024540f, -0.008764f, 0.018843f, 0.016652f, -0.052586f, 0.029360f, -0.006135f,
      0.018843f, -0.055653f, 0.000438f, -0.015337f, -0.017529f, -0.019720f, 0.043383f, -0.003067f},
     {0.015337f, 0.007888f, 0.014461f, 0.030237f, 0.041192f, -0.020596f, -0.031551f, -0.005259f},
     {-0.007147f, 0.013745f, -0.006598f, -0.024191f, -0.005692f, 0.001423f, -0.008537f, 0.025612f},
     {-0.017226f, -0.006380f, -0.008932f, -0.005104f, 0.012760f, -0.024244f, 0.033176f, 0.000638f},
     {0.041471f, 0.031900f, 0.030624f, -0.008294f, -0.003190f, -0.003190f, -0.018502f, 0.033814f,
      -0.040195f, 0.006380f, -0.021692f, -0.000638f, 0.005742f, -0.026158f, 0.021054f, 0.005104f}},

    {108386, "\xE4\xBD\xA0\xE5\xA5\xBD", // "你好"
     {-0.023532f, -0.019610f, -0.013335f, 0.001569f, 0.018826f, -0.034906f, -0.018041f, -0.021963f,
      0.021179f, 0.025101f, 0.001177f, 0.028238f, 0.012158f, 0.009413f, -0.008236f, -0.000784f},
     {0.016865f, -0.016472f, -0.006275f, 0.012943f, 0.032552f, -0.042357f, -0.009021f, -0.032945f},
     {-0.023448f, 0.016883f, -0.031890f, -0.037049f, 0.014723f, -0.013773f, 0.034195f, 0.022322f},
     {-0.025493f, 0.022748f, 0.031376f, 0.014511f, 0.030199f, -0.006667f, -0.000392f, 0.021179f},
     {0.043142f, 0.033337f, 0.004706f, 0.020787f, 0.018433f, -0.017257f, 0.003922f, -0.024709f,
      0.019218f, -0.032552f, 0.049809f, -0.017649f, 0.033337f, -0.002353f, -0.024316f, 0.042357f}},

    {99489, "\xE4\xB8\x96\xE7\x95\x8C", // "世界"
     {-0.018215f, -0.007473f, -0.004671f, -0.036898f, -0.014946f, 0.026155f, -0.021018f, -0.051844f,
      0.001868f, -0.017748f, -0.032694f, -0.015413f, 0.003736f, 0.003269f, -0.019617f, -0.025688f},
     {0.059317f, 0.052778f, 0.016814f, 0.022886f, -0.015413f, 0.010742f, 0.007473f, -0.003269f},
     {-0.013320f, 0.003229f, -0.017357f, -0.049244f, -0.017563f, 0.031223f, -0.038639f, -0.032004f},
     {-0.010968f, -0.020167f, -0.006369f, 0.000354f, -0.018398f, 0.020521f, -0.044934f, -0.003538f},
     {-0.007430f, -0.012737f, -0.013445f, 0.014860f, 0.014153f, 0.044934f, -0.022290f, -0.016629f,
      0.002831f, -0.002477f, 0.016629f, -0.015568f, -0.000354f, 0.010614f, -0.015568f, 0.016275f}},

    {90316, " \xF0\x9F\x98\x80", // " 😀"
     {0.021191f, 0.031187f, 0.007597f, 0.023990f, 0.033985f, -0.010795f, 0.037184f, -0.030387f,
      -0.045580f, -0.001599f, -0.011595f, -0.005997f, 0.021191f, 0.016793f, -0.050778f, -0.006397f},
     {0.007197f, -0.012794f, -0.009596f, 0.015993f, -0.021191f, 0.000000f, -0.002399f, -0.001999f},
     {-0.032094f, 0.016505f, -0.002445f, 0.023535f, -0.001323f, -0.005622f, 0.039682f, -0.008598f},
     {-0.012437f, 0.011845f, -0.006515f, 0.021320f, 0.002961f, -0.075213f, 0.013621f, 0.010068f},
     {0.011252f, -0.020136f, 0.041456f, 0.030796f, 0.020728f, 0.005922f, 0.022505f, 0.010068f,
      0.037903f, 0.027243f, -0.010068f, 0.039087f, 0.026058f, -0.006515f, -0.021913f, -0.024874f}},

    {145836, "\xF0\x9F\x9A\x80", // "🚀"
     {-0.035286f, 0.013828f, -0.011921f, -0.031948f, -0.002384f, -0.005245f, 0.006676f, -0.016212f,
      -0.008583f, -0.016212f, -0.011921f, -0.015259f, 0.037670f, -0.017166f, -0.012875f, -0.017166f},
     {-0.015736f, -0.022411f, 0.004768f, -0.000477f, 0.052452f, 0.007629f, 0.019550f, -0.008106f},
     {-0.016549f, 0.012513f, 0.000807f, -0.008073f, -0.020892f, -0.002749f, 0.051131f, -0.006598f},
     {0.015869f, 0.044136f, 0.024796f, 0.062981f, 0.001488f, 0.000496f, -0.023804f, 0.019836f},
     {0.021324f, 0.027771f, 0.040665f, -0.006943f, 0.007935f, 0.040169f, -0.032730f, -0.004463f,
      0.041656f, 0.007439f, 0.007439f, 0.032730f, 0.001984f, 0.019836f, 0.006447f, -0.003471f}},
};

TEST_CASE("embed_token matches independent Q8_0 dequant reference", "[embedding]")
{
    HANDLE File = CreateFileA("C:\\Users\\zezo_\\.lmstudio\\models\\lmstudio-community"
                            "\\Qwen2.5-3B-Instruct-GGUF\\Qwen2.5-3B-Instruct-Q8_0.gguf",
                            GENERIC_READ, FILE_SHARE_READ|FILE_SHARE_WRITE,
                            0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
    LARGE_INTEGER size;
    GetFileSizeEx(File, &size);

    HANDLE Mapping = CreateFileMappingA(File, 0, PAGE_READONLY, 0, 0, 0);
    
    Data gguf;
    ModelInfo model;

    u8 *file_base = (u8 *)MapViewOfFile(Mapping, FILE_MAP_READ, 0, 0, 0);
    gguf.ptr = file_base;
    gguf.len = static_cast<i64>(size.QuadPart);
    
    parse_gguf(gguf, model, size.QuadPart, file_base);
    const int n_embd = model.cfg.embedding_length; // expected 2048

    for (const auto &c : kCases)
    {
        DYNAMIC_SECTION("token " << c.token_id << " (" << c.piece << ")")
        {
            std::vector<float> x(n_embd);
            embed_token(model.token_embd, c.token_id, x.data(), n_embd);

            check_slice(x.data(), 0, c.first16);
            check_slice(x.data(), 16, c.mid_a);
            check_slice(x.data(), 1020, c.mid_b);
            check_slice(x.data(), 2024, c.mid_c);
            check_slice(x.data(), 2032, c.last16);
        }
    }
}