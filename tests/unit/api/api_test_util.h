#pragma once
// Shared helpers for the API tests: an ApiServer on an ephemeral 127.0.0.1 port over a
// FakeEngine, an httplib client with bounded timeouts (no test can hang on I/O), an SSE
// parser, and a JSON shape matcher for the hand-written response fixtures.

#include <gtest/gtest.h>
#include <httplib.h>

#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "fake_engine.h"
#include "halo/api/server.h"

namespace halo::test {

using Json = nlohmann::ordered_json;

/// A client whose connect/read/write timeouts are all bounded.
inline std::unique_ptr<httplib::Client> make_client(int port, std::chrono::seconds read_timeout = std::chrono::seconds(20)) {
    auto c = std::make_unique<httplib::Client>("127.0.0.1", port);
    c->set_connection_timeout(std::chrono::seconds(5));
    c->set_read_timeout(read_timeout);
    c->set_write_timeout(std::chrono::seconds(10));
    c->set_keep_alive(false);
    return c;
}

struct TestServer {
    std::unique_ptr<FakeEngine> engine;
    std::unique_ptr<api::ApiServer> server;
    int port = 0;

    explicit TestServer(api::ServerConfig cfg = {}, std::unique_ptr<FakeEngine> e = nullptr) {
        engine = e ? std::move(e) : FakeEngine::synthetic();
        cfg.port = 0;
        server = std::make_unique<api::ApiServer>(*engine, std::move(cfg));
        server->start();
        port = server->port();
    }
    ~TestServer() { server->stop(); }
    TestServer(const TestServer&) = delete;
    TestServer& operator=(const TestServer&) = delete;

    [[nodiscard]] std::unique_ptr<httplib::Client> client() const { return make_client(port); }

    httplib::Result post(const std::string& path, const std::string& body, httplib::Headers h = {},
                         const std::string& ct = "application/json") const {
        return client()->Post(path, h, body, ct);
    }
    httplib::Result post(const std::string& path, const Json& body, httplib::Headers h = {}) const {
        return post(path, body.dump(), std::move(h));
    }
    httplib::Result get(const std::string& path, httplib::Headers h = {}) const { return client()->Get(path, h); }
};

struct SseEvent {
    std::string event;  ///< empty for OpenAI-style data-only events
    std::string data;
};

/// Parses a complete text/event-stream body.
inline std::vector<SseEvent> parse_sse(const std::string& body) {
    std::vector<SseEvent> out;
    SseEvent cur;
    bool any = false;
    std::size_t pos = 0;
    while (pos <= body.size()) {
        auto nl = body.find('\n', pos);
        if (nl == std::string::npos) nl = body.size();
        const std::string line = body.substr(pos, nl - pos);
        pos = nl + 1;
        if (line.empty()) {
            if (any) out.push_back(std::move(cur));
            cur = {};
            any = false;
            if (nl == body.size()) break;
            continue;
        }
        if (line.starts_with("event: ")) {
            cur.event = line.substr(7);
            any = true;
        } else if (line.starts_with("data: ")) {
            if (!cur.data.empty()) cur.data += "\n";
            cur.data += line.substr(6);
            any = true;
        }
        if (nl == body.size()) {
            if (any) out.push_back(std::move(cur));
            break;
        }
    }
    return out;
}

/// Every key the fixture has must exist in `actual` with a compatible JSON type; arrays in
/// the fixture give the element shape (first element) for every actual element. A fixture
/// value of null accepts anything (nullable field); a string fixture value that starts with
/// "=" must match exactly (e.g. "=chat.completion"). Returns "" when the shape matches.
inline std::string shape_mismatch(const Json& fixture, const Json& actual, const std::string& path = "$") {
    if (fixture.is_null()) return {};
    if (fixture.is_string()) {
        const auto& s = fixture.get_ref<const std::string&>();
        if (!actual.is_string()) return path + ": expected string, got " + actual.dump();
        if (s.starts_with("=") && actual.get<std::string>() != s.substr(1)) {
            return path + ": expected \"" + s.substr(1) + "\", got " + actual.dump();
        }
        return {};
    }
    if (fixture.is_number()) return actual.is_number() ? "" : path + ": expected number, got " + actual.dump();
    if (fixture.is_boolean()) return actual.is_boolean() ? "" : path + ": expected boolean, got " + actual.dump();
    if (fixture.is_array()) {
        if (!actual.is_array()) return path + ": expected array, got " + actual.dump();
        if (fixture.empty()) return {};
        for (std::size_t i = 0; i < actual.size(); ++i) {
            if (auto m = shape_mismatch(fixture[0], actual[i], path + "[" + std::to_string(i) + "]"); !m.empty()) return m;
        }
        return {};
    }
    if (!actual.is_object()) return path + ": expected object, got " + actual.dump();
    for (auto it = fixture.begin(); it != fixture.end(); ++it) {
        if (!actual.contains(it.key())) return path + "." + it.key() + ": missing";
        if (auto m = shape_mismatch(it.value(), actual[it.key()], path + "." + it.key()); !m.empty()) return m;
    }
    return {};
}

inline Json load_fixture(const std::string& name) {
    return Json::parse(read_file(std::filesystem::path(HALO_API_FIXTURES) / name));
}

}  // namespace halo::test
