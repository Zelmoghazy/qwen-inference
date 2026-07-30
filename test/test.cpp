#include <catch2/catch_test_macros.hpp>
#include "tokenizer.hpp"



TEST_CASE("pretokenize matches regex_golden output", "[regex][pretokenize]") 
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
        { "🏳️‍🌈", { "🏳️‍🌈" } },
    };
    for (const auto& [input, expected] : regex_golden) {
        DYNAMIC_SECTION("input: \"" << input << "\"") {
            auto actual = regex_split(input);
            REQUIRE(actual == expected);
        }
    }
}

TEST_CASE("normalize_nfc converts decomposed sequences to composed form", "[unicode][nfc]")
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