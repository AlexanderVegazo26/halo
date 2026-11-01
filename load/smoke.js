// Blocking k6 smoke (ctest `load_smoke`, so scripts/test-agent and both CI jobs): a few virtual
// users for ~20 s over every public route of the real ApiServer, served by halo_load_server
// (FakeEngine, no model). Thresholds are generous so they hold under the ASan/Debug CI job too;
// what they catch is a route that errors, hangs or slows down by an order of magnitude.
// Run it with load/run.py (it starts the server and sets BASE_URL), never against a real host.
import http from 'k6/http';
import { check, group } from 'k6';

const BASE = __ENV.BASE_URL;
if (!BASE || !/^http:\/\/127\.0\.0\.1:\d+$/.test(BASE)) {
  throw new Error(`BASE_URL must be the local halo_load_server (http://127.0.0.1:<port>), got '${BASE}'`);
}

export const options = {
  scenarios: {
    smoke: { executor: 'constant-vus', vus: 4, duration: '20s' },
  },
  thresholds: {
    http_req_failed: ['rate<0.01'],
    http_req_duration: ['p(95)<1000', 'p(99)<2000'],
    'http_req_duration{route:chat_stream}': ['p(95)<1500'],
    checks: ['rate>0.99'],
  },
  summaryTrendStats: ['avg', 'min', 'med', 'p(90)', 'p(95)', 'p(99)', 'max'],
};

const JSON_HEADERS = { 'Content-Type': 'application/json' };
const messages = [{ role: 'user', content: 'Say something short.' }];
// Thinking off on every chat route: the synthetic template opens a think block by default, and
// with a small max_tokens the reasoning-budget warning would then be logged once per request.
const NO_THINK = { chat_template_kwargs: { enable_thinking: false } };

function post(route, path, body) {
  return http.post(`${BASE}${path}`, JSON.stringify(body), { headers: JSON_HEADERS, tags: { route } });
}

export default function () {
  group('health and discovery', () => {
    const h = http.get(`${BASE}/health`, { tags: { route: 'health' } });
    check(h, { 'health is 200 ok': (r) => r.status === 200 && r.json('status') === 'ok' });
    const m = http.get(`${BASE}/v1/models`, { tags: { route: 'models' } });
    check(m, { 'models lists fake-model': (r) => r.status === 200 && r.json('data.0.id') === 'fake-model' });
  });

  group('openai chat', () => {
    const r = post('chat', '/v1/chat/completions', { model: 'fake-model', messages, max_tokens: 64, ...NO_THINK });
    check(r, {
      'chat is 200 with the scripted reply': (x) =>
        x.status === 200 && String(x.json('choices.0.message.content')).includes('quick brown fox'),
    });
    const s = post('chat_stream', '/v1/chat/completions', { model: 'fake-model', messages, max_tokens: 64, stream: true, ...NO_THINK });
    check(s, {
      'chat stream is 200 SSE ending in [DONE]': (x) =>
        x.status === 200 && String(x.headers['Content-Type']).startsWith('text/event-stream') &&
        x.body.includes('data: [DONE]'),
    });
  });

  group('anthropic messages', () => {
    const body = { model: 'fake-model', messages, max_tokens: 64, thinking: { type: 'disabled' } };
    const r = post('messages', '/v1/messages', body);
    check(r, {
      'messages is 200 with the scripted text block': (x) =>
        x.status === 200 && x.json('content.0.type') === 'text' &&
        String(x.json('content.0.text')).includes('quick brown fox'),
    });
    const c = post('count_tokens', '/v1/messages/count_tokens', { model: 'fake-model', messages });
    check(c, { 'count_tokens is 200 with input_tokens > 0': (x) => x.status === 200 && x.json('input_tokens') > 0 });
  });

  group('completions and utilities', () => {
    const r = post('completions', '/v1/completions', { model: 'fake-model', prompt: 'Once upon', max_tokens: 32 });
    check(r, { 'completions is 200 with text': (x) => x.status === 200 && typeof x.json('choices.0.text') === 'string' });
    const t = post('tokenize', '/tokenize', { content: 'hello world' });
    check(t, { 'tokenize is 200 with tokens': (x) => x.status === 200 && Array.isArray(x.json('tokens')) });
    const p = http.get(`${BASE}/metrics`, { tags: { route: 'metrics' } });
    check(p, { 'metrics is 200 prometheus text': (x) => x.status === 200 && x.body.includes('halo_api_') });
  });
}
