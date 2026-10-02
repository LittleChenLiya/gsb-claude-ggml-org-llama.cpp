// Test the "unconsumed placeholder" diagnostics reported after chat template expansion.
//
// Fixture templates live in tests/chat-template-fixtures/ and cover three cases:
//   - known-placeholders.jinja:    all placeholders are known and consumed by rendering
//   - unknown-placeholders.jinja:  the template emits literal placeholder fragments
//                                  (e.g. from string literals) that survive rendering
//   - escaped-placeholders.jinja:  user input contains placeholder-like text that must
//                                  pass through untouched and must NOT be reported
//
// The two invariants guarded here:
//   1. computing diagnostics never changes the rendered text
//   2. legitimate user text that merely looks like a placeholder is not misreported

#include "chat.h"
#include "chat-auto-parser.h"
#include "json.h"

#undef NDEBUG
#include <cassert>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using json = common_json;
using autoparser::generation_params;

static std::string read_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static generation_params make_inputs(const std::vector<std::pair<std::string, std::string>> & msgs,
                                     bool add_generation_prompt) {
    generation_params inputs;
    json jmsgs = json::array();
    for (const auto & [role, content] : msgs) {
        jmsgs.push_back({{"role", role}, {"content", content}});
    }
    inputs.messages              = jmsgs;
    inputs.add_generation_prompt = add_generation_prompt;
    return inputs;
}

static std::vector<std::string> diag_texts(const std::vector<common_chat_unconsumed_placeholder> & diags) {
    std::vector<std::string> out;
    for (const auto & d : diags) {
        out.push_back(d.text);
    }
    return out;
}

static void check_offsets(const std::string & rendered,
                          const std::vector<common_chat_unconsumed_placeholder> & diags) {
    for (const auto & d : diags) {
        // the reported offset must point at the reported fragment in the rendered text
        assert(d.offset + d.text.size() <= rendered.size());
        assert(rendered.compare(d.offset, d.text.size(), d.text) == 0);
    }
}

static void dump_render_and_diags(const std::string & rendered,
                                  const std::vector<common_chat_unconsumed_placeholder> & diags) {
    std::cout << "--- rendered prompt ---\n" << rendered << "\n";
    std::cout << "--- diagnostics (" << diags.size() << ") ---\n";
    for (const auto & d : diags) {
        std::cout << "  offset " << d.offset << ": " << d.text << "\n";
    }
}

static void test_scanner_provenance() {
    std::cout << "== test_scanner_provenance ==\n";
    // hand-built rendered output mixing template parts and user-input parts
    jinja::string rendered;
    rendered.parts.push_back({/*is_input=*/false, "a {{ x }} b "});
    rendered.parts.push_back({/*is_input=*/true,  "{{ user_literal }} {% also_user %}"});
    rendered.parts.push_back({/*is_input=*/false, "c {% tag %} d {{ unclosed"});

    const std::string flat  = rendered.str();
    auto              diags = common_chat_find_unconsumed_placeholders(rendered);

    std::vector<std::string> expected = {"{{ x }}", "{% tag %}", "{{"};
    assert(diag_texts(diags) == expected);
    check_offsets(flat, diags);
    // user-input fragments must not be reported
    for (const auto & d : diags) {
        assert(d.text.find("user") == std::string::npos);
    }
    std::cout << "ok: template fragments reported, user-input fragments skipped\n";
}

static void test_known_placeholders() {
    std::cout << "== test_known_placeholders ==\n";
    common_chat_template tmpl(read_file("tests/chat-template-fixtures/known-placeholders.jinja"), "<s>", "</s>");
    auto inputs = make_inputs({{"system", "You are helpful"}, {"user", "Hello"}}, /*add_generation_prompt=*/true);

    std::vector<common_chat_unconsumed_placeholder> diags;
    std::string rendered       = common_chat_template_direct_apply(tmpl, inputs, &diags);
    std::string rendered_plain = common_chat_template_direct_apply(tmpl, inputs);

    // diagnostics must not change the rendered text
    assert(rendered == rendered_plain);

    const std::string expected =
        "<s><|im_start|>system\nYou are helpful<|im_end|>\n"
        "<|im_start|>user\nHello<|im_end|>\n"
        "<|im_start|>assistant\n";
    assert(rendered == expected);
    // all placeholders were known and consumed: nothing to report
    assert(diags.empty());
    dump_render_and_diags(rendered, diags);
    std::cout << "ok: rendered " << rendered.size() << " bytes, 0 unconsumed placeholders\n";
}

static void test_unknown_placeholders() {
    std::cout << "== test_unknown_placeholders ==\n";
    common_chat_template tmpl(read_file("tests/chat-template-fixtures/unknown-placeholders.jinja"), "<s>", "</s>");
    auto inputs = make_inputs({{"user", "Hello"}}, /*add_generation_prompt=*/false);

    std::vector<common_chat_unconsumed_placeholder> diags;
    std::string rendered       = common_chat_template_direct_apply(tmpl, inputs, &diags);
    std::string rendered_plain = common_chat_template_direct_apply(tmpl, inputs);

    assert(rendered == rendered_plain);

    const std::string expected =
        "user: Hello\n"
        "TODO: replace {{ assistant_name }} before release\n"
        "{% if debug %}on{% endif %}";
    assert(rendered == expected);

    // the leftover placeholder fragments emitted by the template itself are reported
    std::vector<std::string> expected_diags = {"{{ assistant_name }}", "{% if debug %}", "{% endif %}"};
    assert(diag_texts(diags) == expected_diags);
    check_offsets(rendered, diags);
    dump_render_and_diags(rendered, diags);
    std::cout << "ok: " << diags.size() << " unconsumed placeholders reported at matching offsets\n";
}

static void test_escaped_placeholders() {
    std::cout << "== test_escaped_placeholders ==\n";
    common_chat_template tmpl(read_file("tests/chat-template-fixtures/escaped-placeholders.jinja"), "<s>", "</s>");
    // user text contains placeholder-looking fragments: legitimate content, not template leftovers
    auto inputs = make_inputs({{"user", "show me {{ not_a_placeholder }} and {% raw %} text"}},
                              /*add_generation_prompt=*/false);

    std::vector<common_chat_unconsumed_placeholder> diags;
    std::string rendered       = common_chat_template_direct_apply(tmpl, inputs, &diags);
    std::string rendered_plain = common_chat_template_direct_apply(tmpl, inputs);

    assert(rendered == rendered_plain);

    const std::string expected =
        "user: show me {{ not_a_placeholder }} and {% raw %} text\n"
        "note: {{ intentional_literal }} came from the template";
    assert(rendered == expected);

    // only the template-side literal is reported; the user text is not misreported
    std::vector<std::string> expected_diags = {"{{ intentional_literal }}"};
    assert(diag_texts(diags) == expected_diags);
    check_offsets(rendered, diags);
    dump_render_and_diags(rendered, diags);
    std::cout << "ok: user text with placeholder-like fragments not misreported\n";
}

static void test_end_to_end_apply() {
    std::cout << "== test_end_to_end_apply ==\n";
    auto tmpls = common_chat_templates_init(/*model=*/nullptr,
                                            read_file("tests/chat-template-fixtures/unknown-placeholders.jinja"));

    common_chat_templates_inputs inputs;
    common_chat_msg msg;
    msg.role    = "user";
    msg.content = "Hello";
    inputs.messages              = {msg};
    inputs.add_generation_prompt = false;

    auto params = common_chat_templates_apply(tmpls.get(), inputs);

    // the rendered prompt keeps the leftover fragments (diagnostics never rewrite it)
    assert(params.prompt.find("{{ assistant_name }}") != std::string::npos);
    assert(params.prompt.find("{% if debug %}") != std::string::npos);

    // ... and the diagnostic list is populated on the resulting params
    std::vector<std::string> expected_diags = {"{{ assistant_name }}", "{% if debug %}", "{% endif %}"};
    assert(diag_texts(params.unconsumed_placeholders) == expected_diags);
    check_offsets(params.prompt, params.unconsumed_placeholders);
    std::cout << "ok: common_chat_templates_apply reports " << params.unconsumed_placeholders.size()
              << " unconsumed placeholders, prompt unchanged\n";
}

int main() {
    test_scanner_provenance();
    test_known_placeholders();
    test_unknown_placeholders();
    test_escaped_placeholders();
    test_end_to_end_apply();
    std::cout << "all unconsumed-placeholder tests passed\n";
    return 0;
}
