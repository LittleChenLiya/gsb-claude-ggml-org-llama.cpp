// Acceptance test for the "remaining placeholders" diagnostic.
//
// After a chat template is rendered, the diagnostic lists placeholders the
// template referenced but the caller never provided. The diagnostic must:
//   1. not change the rendered text, and
//   2. not report placeholders that are only present as literal text (escaped
//      via a string literal) or explicitly handled (is defined / default).

#undef NDEBUG
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "chat.h"
#include "chat-auto-parser.h"
#include "json.h"

using json = common_json;

static void expect_contains(const std::string & haystack, const std::string & needle, const char * what) {
    if (haystack.find(needle) == std::string::npos) {
        std::cerr << "FAIL: expected rendered text to contain " << what << " ('" << needle << "')\n";
        std::cerr << "--- rendered text ---\n" << haystack << "\n---------------------\n";
        std::exit(1);
    }
}

static void expect_report(const std::vector<std::string> & report,
                          const std::vector<std::string> & expected,
                          const std::string & scenario) {
    auto print = [](const std::vector<std::string> & v) {
        std::string s = "{ ";
        for (size_t i = 0; i < v.size(); ++i) {
            if (i) s += ", ";
            s += v[i];
        }
        s += " }";
        return s;
    };
    std::cout << "[" << scenario << "] unconsumed placeholders: " << print(report) << "\n";

    if (report != expected) {
        std::cerr << "FAIL: unexpected report for scenario '" << scenario << "'\n";
        std::cerr << "  expected: " << print(expected) << "\n";
        std::exit(1);
    }
}

int main() {
    // Fixture: a known placeholder, an unknown placeholder, an escaped placeholder
    // (literal text via a string literal), and explicitly-handled placeholders.
    const std::string fixture =
        "known={{ known }}\n"
        "unknown={{ unknown }}\n"
        "escaped={{ '{{ escaped }}' }}\n"
        "default={{ optional | default('fallback') }}\n"
        "defined={{ checked is defined }}\n"
        "not_defined={{ checked is not defined }}\n";

    common_chat_template tmpl(fixture, "<s>", "</s>");

    autoparser::generation_params params;
    params.messages = json::array();
    params.extra_context = json{
        {"known", "KNOWN"},
    };

    std::vector<std::string> report;
    std::string rendered = common_chat_template_direct_apply(tmpl, params, &report);

    std::cout << "=== rendered text ===\n" << rendered << "=====================\n";

    // Rendering must be unchanged: known value substituted, unknown empty,
    // escaped placeholder rendered literally, defaults/defined handled.
    expect_contains(rendered, "known=KNOWN", "known placeholder substitution");
    expect_contains(rendered, "unknown=\n", "unknown placeholder renders as empty");
    expect_contains(rendered, "escaped={{ escaped }}", "escaped placeholder renders literally");
    expect_contains(rendered, "default=fallback", "default filter fallback");
    expect_contains(rendered, "defined=False", "is defined test");
    expect_contains(rendered, "not_defined=True", "is not defined test");

    // Diagnostics: only the genuinely unconsumed placeholder is reported.
    expect_report(report, {"unknown"}, "basic");

    // Scenario 2: unknown placeholder referenced inside a loop (shared tracking
    // across scopes).
    {
        const std::string loop_fixture =
            "{% for item in items %}{{ item }}{% endfor %}\n"
            "missing={{ missing }}\n";
        common_chat_template loop_tmpl(loop_fixture, "<s>", "</s>");
        autoparser::generation_params loop_params;
        loop_params.messages = json::array();
        loop_params.extra_context = json{
            {"items", json::array({"a", "b", "c"})},
        };
        std::vector<std::string> loop_report;
        std::string loop_rendered = common_chat_template_direct_apply(loop_tmpl, loop_params, &loop_report);
        std::cout << "=== loop rendered text ===\n" << loop_rendered << "==========================\n";
        expect_contains(loop_rendered, "abc", "loop body renders");
        expect_report(loop_report, {"missing"}, "loop-scope");
    }

    // Scenario 3: no unconsumed placeholders -> empty report.
    {
        const std::string clean_fixture = "Hello {{ name }}!";
        common_chat_template clean_tmpl(clean_fixture, "<s>", "</s>");
        autoparser::generation_params clean_params;
        clean_params.messages = json::array();
        clean_params.extra_context = json{{"name", "world"}};
        std::vector<std::string> clean_report;
        std::string clean_rendered = common_chat_template_direct_apply(clean_tmpl, clean_params, &clean_report);
        expect_contains(clean_rendered, "Hello world!", "clean render");
        expect_report(clean_report, {}, "clean");
    }

    std::cout << "\nALL ACCEPTANCE CHECKS PASSED\n";
    return 0;
}
