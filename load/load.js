// Opt-in heavier load run (not in ctest): ramps past the admission limits of halo_load_server
// (default max_concurrent 4 + max_queue 16) and holds there, then spikes. Under overload the API
// must shed with 429/503 + Retry-After instead of erroring, hanging or crashing, and the requests
// it admits must stay fast. Run:
//   python3 load/run.py --server build/tests/unit/api/halo_load_server \
//       --server-arg=--token-delay-ms=2 load/load.js
// (2 ms per token makes a generation take ~90 ms, so the queue really fills.)
//
// Budgets follow from the admission limits, not from host speed: an admitted request waits behind
// at most max_queue / max_concurrent = 4 generations, so served p95 is ~5 generations (~465 ms,
// load/README.md "Baseline"). p95 < 700 ms / p99 < 1 s fail when admission stops bounding the
// queue (--queue=1000: every request is admitted and waits ~1.1 s) and when a generation gets
// ~1.5x slower; served/s and shed > 0 fail when concurrency collapses or the spike is never shed.
// HALO_LOAD_BUDGET_SCALE (run.py --budget-scale) loosens them for sanitizer/Debug builds.
import http from 'k6/http';
import { check } from 'k6';
import { Counter } from 'k6/metrics';

const BASE = __ENV.BASE_URL;
if (!BASE || !/^http:\/\/127\.0\.0\.1:\d+$/.test(BASE)) {
  throw new Error(`BASE_URL must be the local halo_load_server (http://127.0.0.1:<port>), got '${BASE}'`);
}

// Shedding is the designed answer to overload, not a failure.
http.setResponseCallback(http.expectedStatuses(200, 429, 503));

const SCALE = Number(__ENV.HALO_LOAD_BUDGET_SCALE ?? '1');
if (!Number.isFinite(SCALE) || SCALE < 1) {
  throw new Error(`HALO_LOAD_BUDGET_SCALE must be a number >= 1, got '${__ENV.HALO_LOAD_BUDGET_SCALE}'`);
}
const ms = (budget) => budget * SCALE;

const served = new Counter('served'); // 200 replies
const shed = new Counter('shed'); // 429/503 with Retry-After

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
    'http_req_duration{expected_response:true,status:200}': [`p(95)<${ms(700)}`, `p(99)<${ms(1000)}`],
    checks: ['rate>0.99'],
    // 4 slots x ~11 generations/s = ~44 served/s on any host; 25/s fails with 2 slots or slower generations.
    served: [`rate>${25 / SCALE}`],
    // The 48-VU spike is 2.4x what admission holds (4 + 16): it must be shed, not queued.
    shed: ['count>0'],
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
  if (r.status === 200) served.add(1);
  else if (r.status === 429 || r.status === 503) shed.add(1);
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
