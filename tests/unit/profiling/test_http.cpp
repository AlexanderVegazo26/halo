#include <gtest/gtest.h>

#include <chrono>
#include <thread>

#include "fake_http.h"
#include "halo/profiling/http_client.h"
#include "test_util.h"

using namespace halo::profiling;
using halo::ErrorCode;
using halo::profiling::test::FakeHttpServer;
using namespace std::chrono_literals;

namespace {

HttpRequest get(std::uint16_t port, std::string path = "/x") {
    HttpRequest r;
    r.port = port;
    r.path = std::move(path);
    r.timeout = 3s;
    return r;
}

}  // namespace

TEST(Http, ContentLengthAndPostBody) {
    FakeHttpServer srv([](const auto& m, const auto& p, const auto& b) {
        return FakeHttpServer::json(200, "{\"m\":\"" + m + "\",\"p\":\"" + p + "\",\"n\":" + std::to_string(b.size()) + "}");
    });
    HttpRequest r = get(srv.port(), "/completion");
    r.method = "POST";
    r.body = "{\"prompt\":\"hi\"}";
    const auto resp = http_request(r);
    EXPECT_EQ(resp.status, 200);
    EXPECT_EQ(resp.body, "{\"m\":\"POST\",\"p\":\"/completion\",\"n\":15}");
    EXPECT_EQ(resp.headers.at("content-type"), "application/json");
    ASSERT_EQ(srv.requests().size(), 1u);
    EXPECT_EQ(srv.requests()[0], "POST /completion {\"prompt\":\"hi\"}");
}

TEST(Http, ChunkedAndReadUntilClose) {
    FakeHttpServer chunked([](const auto&, const auto&, const auto&) {
        return std::string("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
                           "5\r\nhello\r\n1;ext=1\r\n \r\n5\r\nworld\r\n0\r\n\r\n");
    });
    EXPECT_EQ(http_request(get(chunked.port())).body, "hello world");
    FakeHttpServer close([](const auto&, const auto&, const auto&) {
        return std::string("HTTP/1.0 503 Loading\r\n\r\n{\"status\":\"loading\"}");
    });
    const auto r = http_request(get(close.port()));
    EXPECT_EQ(r.status, 503);
    EXPECT_EQ(r.body, "{\"status\":\"loading\"}");
}

TEST(Http, MalformedResponsesAreTypedErrors) {
    const auto expect_api = [](std::string raw) {
        FakeHttpServer s([raw](const auto&, const auto&, const auto&) { return raw; });
        test::expect_error(ErrorCode::Api, [&] { (void)http_request(get(s.port())); });
    };
    expect_api("NOTHTTP 200\r\n\r\n");
    expect_api("HTTP/1.1 2x0 OK\r\n\r\n");
    expect_api("HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\nshort");                 // truncated
    expect_api("HTTP/1.1 200 OK\r\nContent-Length: -1\r\n\r\n");                        // not decimal
    expect_api("HTTP/1.1 200 OK\r\nContent-Length: 99999999999999999999999\r\n\r\n");  // overflow
    expect_api("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\nzz\r\nabc\r\n0\r\n\r\n");
    expect_api("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabcXX0\r\n\r\n");
    expect_api("HTTP/1.1 200 OK\r\nno colon here\r\n\r\n");
    expect_api("HTTP/1.1 200 OK\r\nContent-Length: 5\r\n");  // headers never end
}

TEST(Http, BodyCapIsEnforced) {
    FakeHttpServer s([](const auto&, const auto&, const auto&) { return FakeHttpServer::json(200, std::string(5000, 'a')); });
    HttpRequest r = get(s.port());
    r.max_body_bytes = 4096;
    test::expect_error(ErrorCode::Api, [&] { (void)http_request(r); });
    FakeHttpServer c([](const auto&, const auto&, const auto&) {
        return std::string("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n2000\r\n") + std::string(0x2000, 'b') +
               "\r\n0\r\n\r\n";
    });
    r.port = c.port();
    test::expect_error(ErrorCode::Api, [&] { (void)http_request(r); });
    FakeHttpServer eof([](const auto&, const auto&, const auto&) { return "HTTP/1.1 200 OK\r\n\r\n" + std::string(8000, 'c'); });
    r.port = eof.port();
    test::expect_error(ErrorCode::Api, [&] { (void)http_request(r); });
}

TEST(Http, TimeoutRefusedAndNonLoopback) {
    FakeHttpServer slow([](const auto&, const auto&, const auto&) {
        std::this_thread::sleep_for(1500ms);
        return FakeHttpServer::json(200, "{}");
    });
    HttpRequest r = get(slow.port());
    r.timeout = 300ms;
    const auto t0 = std::chrono::steady_clock::now();
    test::expect_error(ErrorCode::Io, [&] { (void)http_request(r); });
    EXPECT_LT(std::chrono::steady_clock::now() - t0, 1200ms);

    std::uint16_t dead = 0;
    { FakeHttpServer tmp([](const auto&, const auto&, const auto&) { return std::string(); }); dead = tmp.port(); }
    test::expect_error(ErrorCode::Io, [&] { (void)http_request(get(dead)); });  // refused

    for (const char* host : {"localhost", "10.0.0.1", "example.com", "0.0.0.0", "127.0.0.2"}) {
        HttpRequest bad = get(1);
        bad.host = host;
        test::expect_error(ErrorCode::Config, [&] { (void)http_request(bad); });
    }
    HttpRequest inj = get(1, "/x\r\nHost: evil");
    test::expect_error(ErrorCode::Config, [&] { (void)http_request(inj); });
}
