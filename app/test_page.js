// Runs the real page (static/index.html) in a simulated browser (jsdom) against a fake server, using a real result
// built from a live run (fixtures/live_result.json: the billing-questions request, with its real legal wording and
// its real zero-of-nine competitor result). Setup, once:  npm install jsdom   Run:  node test_page.js
const fs = require("fs");
const { JSDOM } = require("jsdom");
const html = fs.readFileSync(__dirname + "/static/index.html", "utf8");
const real = JSON.parse(fs.readFileSync(__dirname + "/fixtures/live_result.json", "utf8"));

let pass = 0, fail = 0;
const check = (name, ok, detail) => { if (ok) { pass++; console.log("PASS  " + name); } else { fail++; console.log("FAIL  " + name + "   -> " + (detail === undefined ? "" : String(detail).slice(0, 400))); } };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const clone = (o) => JSON.parse(JSON.stringify(o));

const dashboard = { stats: { evaluations_saved: 3, awaiting_feedback: 1, decision_quality: { rate: 0.5, met_all: 1, comparable: 2 }, median_seconds_to_result: 52.3, runs_measured: 4 },
  evaluations: [
    { id: "a", feature: "A billing chatbot for SaaS customers", model: "Claude Sonnet 5.5", forecast: { yearly_cost_usd: 1680, latency_ms: 669, latency_kind: "to the first words" }, accepted_at: "2026-10-01T00:00:00+00:00", accepted_misses: [], legal: "transparency_obligated", status: "feedback_due", next_follow_up: { day: 30, due: "2026-10-31T00:00:00+00:00" }, sample: true, outcome: null },
    { id: "b", feature: "Ticket tagger", model: "gpt-6-luna", forecast: { yearly_cost_usd: 66, latency_ms: 773, latency_kind: "for the full reply" }, accepted_at: "2026-09-01T00:00:00+00:00", accepted_misses: ["latency"], legal: "no_specific_obligation_identified", status: "reported", next_follow_up: null, sample: true,
      outcome: { reported_at: "2026-10-02T00:00:00+00:00", latency_ms: 900, yearly_cost_usd: 70, verdict: "missed", targets: { latency: "missed", cost: "met" } } }],
  action_needed: { id: "a", feature: "A billing chatbot for SaaS customers", next_follow_up: { day: 30 } }, top_models: [{ model: "Claude Sonnet 5.5", count: 2 }], has_sample_data: true };

// A result containing hostile text, to prove model output is never turned into markup.
const hostile = clone(real);
hostile.summary = "<img src=x onerror=window.__pwned=1> summary";
hostile.regulatory.answer = "<script>window.__pwned=1</script>Article 50 applies.";
hostile.regulatory.quotes[0].text = "<img src=x onerror=window.__pwned=1> quote";
hostile.cards[0].model = "<b>Model</b>";
hostile.guardrails.lines = ["<img src=x onerror=window.__pwned=1>"];

// A result where two competitor claims were verified, to test the table.
const verified = clone(real);
verified.competitors = [
  { product: "Intercom Fin", vendor: "Intercom", cited_hosts: ["intercom.com"], claims: [
    { field: "pricing", note: "$0.99 per resolved conversation", source_url: "https://www.intercom.com/pricing", check: "verified" },
    { field: "accuracy", note: "No verified source: x", source_url: "https://x.example/a", check: "page_does_not_support_claim" },
    { field: "latency", note: "n", source_url: null, check: "could_not_be_checked (page returned 403)" }] }];
verified.competitor_coverage = { total: 3, verified: 1, not_supported: 1, could_not_check: 1, no_source: 0, unchecked: 0 };
verified.verdicts[2] = { key: "market", heading: "MARKET", title: "1 competitor researched", detail: "1 of 3 claims verified", extra: null, tone: "caution" };

function sse(events) { return events.map((e) => "data: " + JSON.stringify(e) + "\n\n").join(""); }
function streamBody(text, chunk = 97, gate) { // split oddly so the parser must reassemble events; optionally pause once at a marker
  const enc = new TextEncoder(); let bytes = enc.encode(text); let i = 0; const pauseAt = gate ? enc.encode(text).length : -1; let paused = false;
  return { getReader() { return { read: async () => {
    if (gate && !paused && i >= gate.at) { paused = true; await gate.wait; }
    if (i >= bytes.length) return { done: true };
    const end = gate && !paused ? Math.min(i + chunk, gate.at) : i + chunk; const v = bytes.slice(i, end); i = end; return { done: false, value: v }; } }; } };
}

function eventsFor(resultToUse) {
  return [{ type: "reading" }, { type: "note", message: "Plan: I'll run it." },
    { type: "plan", request_type: "new_evaluation", agents: ["regulations", "competitive_analysis", "model_stack_analysis"], message: "Plan: I'll run the regulations check, the competitive analysis and the model-stack analysis now." },
    { type: "agent_started", agent: "regulations" }, { type: "agent_started", agent: "competitive_analysis" }, { type: "agent_started", agent: "model_stack_analysis" },
    { type: "agent_finished", agent: "model_stack_analysis", status: "ok", seconds: 1.2 }, { type: "agent_finished", agent: "competitive_analysis", status: "ok", seconds: 7.8 },
    { type: "agent_finished", agent: "regulations", status: "ok", seconds: 8.1 }, { type: "writing" },
    { type: "note", message: "  (the written summary left out part of the regulatory finding; the system put it back)" },
    { type: "result", run_id: "r1", ...resultToUse }];
}

async function run(resultToUse, opts = {}) {
  const calls = [];
  const dom = new JSDOM(html, { runScripts: "dangerously", url: "https://shipwise.test/", pretendToBeVisual: true, beforeParse(window) {
    window.TextDecoder = TextDecoder; window.TextEncoder = TextEncoder;
    window.Element.prototype.scrollIntoView = function () { window.__scrolledTo = this.id; };
    window.supabase = { createClient: () => ({ auth: { getSession: async () => ({ data: { session: { access_token: "tok" } } }), signOut: async () => {}, signInWithPassword: async () => ({}), signUp: async () => ({}) } }) };
    window.fetch = async (url, o) => {
      calls.push({ url, opts: o });
      const json = (obj, status = 200) => ({ ok: status < 400, status, json: async () => obj });
      if (url === "/config") return json({ supabase_url: "https://x.supabase.co", supabase_key: "k" });
      if (url === "/api/me") return json({ user_id: "u1", full_name: "Priyanka", company: "Northwind Support", company_id: "northwind-support", email: "a@x" });
      if (url === "/api/dashboard") return json(dashboard);
      if (url === "/api/evaluate/stream") {
        const text = ": keepalive\n\n" + sse(eventsFor(resultToUse));
        if (opts.gate) { const cut = text.indexOf('data: {"type":"agent_finished","agent":"competitive_analysis"'); return { ok: true, status: 200, body: streamBody(text, 97, { at: cut, wait: opts.gate }) }; }
        return { ok: true, status: 200, body: streamBody(text) };
      }
      if (url === "/api/accept") return json({ message: "m", recommendation_id: "rec1", chosen: JSON.parse(o.body).option_id, accepted_misses: ["latency"], follow_ups: [{ day: 30, due: "2026-11-05T00:00:00+00:00" }, { day: 60, due: "2026-12-05T00:00:00+00:00" }, { day: 90, due: "2027-01-04T00:00:00+00:00" }], legal: { classification: "transparency_obligated" } });
      return json({}, 404);
    };
  } });
  const w = dom.window, d = w.document;
  w.addEventListener("error", (e) => { fail++; console.log("FAIL  uncaught page error -> " + e.message); });
  await sleep(150);
  return { w, d, calls };
}
const click = (d, re, sel = "button") => [...d.querySelectorAll(sel)].find((b) => re.test(b.textContent)).click();
const text = (d, id) => d.getElementById(id).textContent;

(async () => {
  let { w, d, calls } = await run(real);
  check("after sign-in the app view shows and the sign-in view hides", !d.getElementById("app-view").classList.contains("hidden") && d.getElementById("auth-view").classList.contains("hidden"));
  check("the top bar shows the person and company", d.getElementById("who-name").textContent === "Priyanka" && d.getElementById("who-company").textContent === "Northwind Support");
  check("the Dashboard greets the user by first name", /Welcome back, Priyanka/.test(text(d, "page")));
  check("the Dashboard still shows counted values, sample badges and no removed mock blocks", /Evaluations saved/.test(text(d, "page")) && /50%/.test(text(d, "page")) && d.querySelectorAll(".pill.sample").length === 2 && !/Team Access|Accuracy Trend|Confidence|Version History/.test(d.body.textContent));

  click(d, /New Evaluation/); await sleep(60);
  check("the New Evaluation button opens a full chat box and nothing else yet", w.location.hash === "#/new" && !!d.getElementById("msg") && !!d.getElementById("send") && d.querySelectorAll(".tile").length === 0 && d.querySelectorAll(".sec").length === 0);

  d.getElementById("msg").value = "i want to build an ai feature that allows answering billing questions , should cost less than $40k per year, with latency <900ms"; d.getElementById("send").click();
  await sleep(150);
  const sent = calls.find((c) => c.url === "/api/evaluate/stream");
  check("the message is posted with the token and no session id the first time", sent && sent.opts.headers.Authorization === "Bearer tok" && JSON.parse(sent.opts.body).session_id === null);

  // ---- the answer-first layout
  const page = d.getElementById("page");
  const tiles = [...d.querySelectorAll(".tile")];
  check("three verdict tiles appear: legal, recommended, market", tiles.length === 3 && /LEGAL/.test(tiles[0].textContent) && /RECOMMENDED/.test(tiles[1].textContent) && /MARKET/.test(tiles[2].textContent), tiles.map((t) => t.textContent).join(" | "));
  check("the legal tile names the classification, the act and article, and when it applies", /Transparency duty/.test(tiles[0].textContent) && /EU AI Act, Article 50\(1\)/.test(tiles[0].textContent) && /Applies from 2 August 2026/.test(tiles[0].textContent), tiles[0].textContent);
  check("the recommended tile names the top option with its monthly cost and latency", /Sonnet 5\.5/.test(tiles[1].textContent) && /\$110\/month/.test(tiles[1].textContent) && /669 ms to the first words/.test(tiles[1].textContent) && /2 other options below/.test(tiles[1].textContent), tiles[1].textContent);
  check("the market tile says plainly that nothing could be verified", /3 competitors researched/.test(tiles[2].textContent) && /0 of 9 claims verified/.test(tiles[2].textContent) && /No claim could be backed by its source/.test(tiles[2].textContent), tiles[2].textContent);
  check("the legal tile is above everything except the request bar (the classification is on top)", tiles[0].compareDocumentPosition(d.getElementById("sec-legal")) & 4 && d.getElementById("request-slot").compareDocumentPosition(tiles[0]) & 4);
  check("the request has collapsed to a slim bar with the request text and the targets", /YOUR REQUEST/.test(text(d, "request-slot")) && /billing questions/.test(text(d, "request-slot")) && !d.getElementById("send") && /Latency/.test(text(d, "request-slot")) && /<900ms/.test(text(d, "request-slot")));
  check("chips show the volume as assumed and the reply style as inferred", /20,000 \/ month/.test(text(d, "request-slot")) && /assumed/i.test(text(d, "request-slot")) && /streamed chat/.test(text(d, "request-slot")) && /inferred/i.test(text(d, "request-slot")));
  tiles[2].click();
  check("clicking a tile scrolls to its section", w.__scrolledTo === "sec-market", w.__scrolledTo);

  // ---- 1. regulations
  const legal = d.getElementById("sec-legal");
  check("the regulations section carries its number, the classification pill and the explanation", /^1/.test(legal.textContent.trim()) && /Transparency duty/.test(legal.querySelector(".tonepill").textContent) && /Article 50\(1\) governs it/.test(legal.textContent));
  check("it says when the rules apply", /When it applies: 2 August 2026/.test(legal.textContent), legal.querySelector(".when") && legal.querySelector(".when").textContent);
  const quote = legal.querySelector("details.quote");
  check("the Act's exact wording is available under the citation", !!quote && /Article 50\(1\)/.test(quote.querySelector("summary").textContent) && /Providers shall ensure that AI systems intended to interact directly with natural persons/.test(quote.textContent), quote && quote.textContent.slice(0, 120));
  check("the quote links to the Act only through a safe address", !quote.querySelector("a") || /^https:/.test(quote.querySelector("a").href));
  check("the disclaimer is shown", /not legal advice/i.test(legal.textContent));

  // ---- 2. recommendations
  const reco = d.getElementById("sec-reco");
  const cards = [...reco.querySelectorAll(".scen")];
  check("three cards: A Recommended, B Lowest cost, C Caution", cards.length === 3 && /RECOMMENDED/.test(cards[0].textContent) && /LOWEST COST/.test(cards[1].textContent) && /CAUTION/.test(cards[2].textContent), cards.map((c) => c.querySelector(".badge").textContent).join(","));
  check("the Caution card shows its miss underneath, against the PM's own target", /Misses latency by 296 ms \(1,196 ms against 900 ms\)/.test(cards[2].textContent), cards[2].textContent);
  check("cards show monthly and yearly cost, latency with its kind, and quality marked as not accuracy", /Cost \/ month/.test(cards[0].textContent) && /to the first words/.test(cards[0].textContent) && /not accuracy on your task/.test(cards[0].textContent));
  check("no card shows an accuracy figure", !/Accuracy\b/.test(cards.map((c) => c.textContent).join(" ").replace(/not accuracy on your task/g, "")));
  const grid = reco.querySelector(".recogrid");
  check("'How to read these options' sits right beside the cards", !!grid && !!grid.querySelector(".cards3") && !!grid.querySelector(".reading"));
  const reading = grid.querySelector(".reading").textContent;
  check("it says what was compared and that 66 had no timing data", /102 models compared; 5 meet your targets/.test(reading) && /66 had no timing data/.test(reading), reading);
  check("it says what it can't tell you (accuracy)", /No public benchmark measures accuracy/.test(reading));
  check("it lists the assumptions: volume as assumed, tokens as estimated, what latency means", /20,000 requests a month \(assumed, not given\)/.test(reading) && /1,500 tokens in and 250 out/.test(reading) && /Latency is the time until the first answer token/.test(reading), reading);
  check("it credits the sources", /vendors' own pricing pages/.test(reading) && /Artificial Analysis/.test(reading));
  check("there is no right-hand sidebar any more", !/Agent execution/i.test(d.body.textContent) && !d.querySelector(".grid2"));

  // ---- 3. competitors: the real zero-of-nine result
  const market = d.getElementById("sec-market");
  check("the competitors section states the coverage first", /0 of 9 claims verified/.test(market.querySelector(".cov").textContent) && /9 not supported by their source/.test(market.querySelector(".cov").textContent), market.querySelector(".cov").textContent);
  check("with nothing verified there is one plain explanation and no table, not nine repeated sentences", /None of the competitor claims could be backed by a source/.test(market.textContent) && !market.querySelector("table") && !/the cited page does not exist/.test(market.textContent));
  check("it lists what the research cited, per competitor", /Intercom Fin/.test(market.textContent) && /cited: aerochat\.ai/.test(market.textContent) && market.querySelectorAll(".li").length === 3, market.textContent);

  // ---- how this was produced, and the written summary
  const produced = [...d.querySelectorAll("#sections-slot details")].find((x) => /How this was produced/.test(x.textContent));
  check("'How this was produced' is collapsed and holds the plan with times", !!produced && !produced.open && /Checking the EU AI Act \(8\.1 s\)/.test(produced.textContent) && /Researching competitors \(7\.8 s\)/.test(produced.textContent));
  check("it says how many guardrail checks ran and how many fired, in plain sentences", /16 guardrail checks ran; 5 changed or refused something/.test(produced.textContent) && /raised to the typical range \(input tokens 200 -> 1,500\)/.test(produced.textContent) && /9 competitor claims removed because the cited page did not support them/.test(produced.textContent) && /put back/.test(produced.textContent), produced.textContent);
  check("the written summary is kept, collapsed", [...d.querySelectorAll("#sections-slot details")].some((x) => /Written summary/.test(x.textContent) && !x.open));
  check("the plan card is gone once the result is in (it moved into 'How this was produced')", d.getElementById("plan-slot").children.length === 0);

  // ---- sections come in the order regulations, recommendations, competitors
  const order = [...d.querySelectorAll(".sec")].map((s) => s.id);
  check("sections are in order: regulations, recommendations, competitors", order.join(",") === "sec-legal,sec-reco,sec-market", order.join(","));

  // ---- follow-up and Edit & re-run
  click(d, /^Follow-up$/); await sleep(20);
  check("Follow-up reveals the composer without losing the results", !!d.getElementById("msg") && d.querySelectorAll(".tile").length === 3);
  click(d, /Edit & re-run/); await sleep(20);
  check("Edit & re-run opens two fields filled with the current targets", d.getElementById("rr-lat").value === "<900ms" && d.getElementById("rr-bud").value === "$40k per year");
  d.getElementById("rr-lat").value = "under 500ms"; click(d, /^Re-run$/); await sleep(150);
  const second = calls.filter((c) => c.url === "/api/evaluate/stream")[1];
  check("re-run posts a plain follow-up in the same session", second && /a latency target of under 500ms and a budget of \$40k per year/.test(JSON.parse(second.opts.body).message) && JSON.parse(second.opts.body).session_id === real.session_id, second && second.opts.body);

  // ---- Approve the Caution card
  click(d, /Accept with this trade-off/); await sleep(80);
  const acc = calls.find((c) => c.url === "/api/accept");
  check("Approve posts the session and the option id", acc && JSON.parse(acc.opts.body).session_id === real.session_id && JSON.parse(acc.opts.body).option_id === real.cards[2].option_id, acc && acc.opts.body);
  const done = d.querySelector(".done") ? d.querySelector(".done").textContent : "";
  check("the confirmation names the choice, the accepted miss, the legal result and the three dates", /Recorded: you chose/.test(done) && /misses your latency target/.test(done) && /transparency obligated/.test(done) && /day 30/.test(done) && /day 60/.test(done) && /day 90/.test(done), done);
  check("after approving, the buttons are disabled and the chosen one says Approved", [...d.querySelectorAll(".scen .btn")].every((b) => b.disabled) && /Approved ✓/.test([...d.querySelectorAll(".scen")][2].textContent));
  check("the approval survives the page re-rendering other sections", !!d.querySelector(".done"));

  // ---- New evaluation resets to a full chat box
  click(d, /^New evaluation$/); await sleep(40);
  check("New evaluation clears the results and shows the full chat box again", !!d.getElementById("send") && d.querySelectorAll(".tile").length === 0 && d.querySelectorAll(".sec").length === 0);

  // ---- the dashboard and Report outcome
  w.location.hash = "#/"; await sleep(60);
  check("returning to the Dashboard shows it again", /Welcome back, Priyanka/.test(text(d, "page")));
  click(d, /Report outcome/); await sleep(60);
  check("Report outcome opens the chat with the feature already written in", w.location.hash === "#/new" && /I want to report how our launch went/.test(d.getElementById("msg").value) && /billing chatbot/.test(d.getElementById("msg").value), d.getElementById("msg") && d.getElementById("msg").value);

  // ---- the live plan while a request runs
  let release; const gate = new Promise((r) => { release = r; });
  ({ w, d, calls } = await run(real, { gate }));
  click(d, /New Evaluation/); await sleep(60);
  d.getElementById("msg").value = "x"; d.getElementById("send").click(); await sleep(120);
  const planNow = text(d, "plan-slot");
  check("while the request runs the plan card is shown, with finished steps ticked and running ones pulsing", /Orchestrator plan/.test(planNow) && /Comparing model prices and speed \(1\.2 s\)/.test(planNow) && d.querySelectorAll("#plan-slot .ic.run").length >= 1 && d.querySelectorAll("#plan-slot .ic.ok").length >= 2, planNow);
  check("no verdicts or sections are shown until the result arrives, and the button says Working…", d.querySelectorAll(".tile").length === 0 && d.querySelectorAll(".sec").length === 0 && d.getElementById("send").textContent === "Working…" && d.getElementById("send").disabled);
  release(); await sleep(200);
  check("when the result arrives the plan card goes and the verdicts appear", d.getElementById("plan-slot").children.length === 0 && d.querySelectorAll(".tile").length === 3 && d.getElementById("send") === null);

  // ---- verified competitor claims render as a table, unverified cells stay compact
  ({ w, d, calls } = await run(verified));
  click(d, /New Evaluation/); await sleep(60); d.getElementById("msg").value = "x"; d.getElementById("send").click(); await sleep(150);
  const mk = d.getElementById("sec-market");
  check("with one verified claim a table appears with a source link", !!mk.querySelector("table") && /\$0\.99 per resolved conversation/.test(mk.textContent) && /✓ source checked/.test(mk.textContent) && /^https:\/\/www\.intercom\.com/.test(mk.querySelector("td a").href), mk.textContent);
  check("unverified cells are one short grey line, not the long sentence", /no verified source/.test(mk.textContent) && /source could not be opened/.test(mk.textContent) && !/No verified source: x/.test(mk.textContent) && mk.querySelectorAll("td.unv").length === 2);
  check("the coverage line counts each kind", /1 of 3 claims verified/.test(mk.querySelector(".cov").textContent) && /1 not supported by their source/.test(mk.querySelector(".cov").textContent) && /1 source could not be opened/.test(mk.querySelector(".cov").textContent));

  // ---- hostile text never becomes markup
  ({ w, d, calls } = await run(hostile));
  click(d, /New Evaluation/); await sleep(60); d.getElementById("msg").value = "x"; d.getElementById("send").click(); await sleep(150);
  [...d.querySelectorAll("details")].forEach((x) => { x.open = true; });
  check("model output containing tags is shown as text, never run or parsed", w.__pwned === undefined && !d.querySelector("#sections-slot img") && !d.querySelector("#sections-slot script") && /<b>Model<\/b>/.test(d.getElementById("sections-slot").textContent) && /<img src=x/.test(d.getElementById("sections-slot").textContent), String(w.__pwned));

  console.log(`\n${fail === 0 ? "ALL PASSED" : fail + " FAILED"} (${pass} passed)`);
  process.exit(fail === 0 ? 0 : 1);
})();
