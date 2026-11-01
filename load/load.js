// Opt-in heavier load run (not in ctest): ramps past the admission limits of halo_load_server
// (default max_concurrent 4 + max_queue 16) and holds there, then spikes. Under overload the API
// must shed with 429/503 + Retry-After instead of erroring, hanging or crashing, and the requests
// it admits must stay fast. Run:
//   python3 load/run.py --server build/tests/unit/api/halo_load_server \
//       --server-arg=--token-delay-ms=2 load/load.js
// (2 ms per token makes a generation take ~90 ms, so the queue really fills.)
import http from 'k6/http';
import { check } from 'k6';

const BASE = __ENV.BASE_URL;
if (!BASE || !/^http:\/\/127\.0\.0\.1:\d+$/.test(BASE)) {
  throw new Error(`BASE_URL must be the local halo_load_server (http://127.0.0.1:<port>), got '${BASE}'`);
}

// Shedding is the designed answer to overload, not a failure.
http.setResponseCallback(http.expectedStatuses(200, 429, 503));

export const options = {
  scenarios: {
    ramp_soak: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '30s', target: 16 },
        { duration: '2m', target: 16 },
        { duration: '30s', target: 48 },
        { duration: '30s', target: 48 },
        { duration: '30s', target: 0 },
      ],
      gracefulRampDown: '10s',
    },
  },
  thresholds: {
    http_req_failed: ['rate<0.01'],
    'http_req_duration{expected_response:true,status:200}': ['p(95)<3000'],
    checks: ['rate>0.99'],
  },
  summaryTrendStats: ['avg', 'min', 'med', 'p(90)', 'p(95)', 'p(99)', 'max'],
};

const body = JSON.stringify({
  model: 'fake-model',
  messages: [{ role: 'user', content: 'Say something short.' }],
  max_tokens: 64,
  chat_template_kwargs: { enable_thinking: false },
});

export default function () {
  const stream = __ITER % 2 === 1;
  const r = http.post(`${BASE}/v1/chat/completions`, stream ? body.replace(/}$/, ',"stream":true}') : body, {
    headers: { 'Content-Type': 'application/json' },
    tags: { route: stream ? 'chat_stream' : 'chat' },
  });
  check(r, {
    'served (200) or shed with Retry-After (429/503)': (x) =>
      x.status === 200 || ((x.status === 429 || x.status === 503) && x.headers['Retry-After'] !== undefined),
  });
  // A stream carries the reply as one SSE chunk per byte token, so only its terminator is checked.
  check(r, {
    'a served reply is complete': (x) =>
      x.status !== 200 || (stream ? x.body.includes('data: [DONE]') : x.body.includes('quick brown fox')),
  });
}
