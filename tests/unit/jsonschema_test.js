// harness/jsonschema.js: the JSON Schema subset the API grammar enforces, compiled to a node table.
import { compileSchema, stringCapable, schemaTypes, SchemaError, SCHEMA_CAPS } from "../../harness/jsonschema.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const throwsSchema = (f, re, m) => { try { f(); } catch (e) { ok(e instanceof SchemaError && re.test(e.message), m + ": " + e.message); return; } throw new Error(m + ": no error"); };
const node = (t) => t.nodes[t.root];

Deno.test("jsonschema: objects keep declared keys, required indexes and no free keys", () => {
  const t = compileSchema({ type: "object", properties: { a: { type: "string" }, b: { type: "integer" } }, required: ["b", "zzz"], additionalProperties: true });
  const n = node(t);
  eq(n.k, "obj");
  eq(n.props.map(([k]) => k), ["a", "b"]);
  eq(n.req, [1], "unknown required keys are dropped");
  eq(n.free, null, "declared properties: no free keys even when additionalProperties is not false");
  eq(t.nodes[n.props[1][1]].k, "int");
  const free = node(compileSchema({ type: "object", additionalProperties: { type: "number" } }));
  eq([free.props.length, compileSchema({ type: "object", additionalProperties: { type: "number" } }).nodes[free.free].k], [0, "num"]);
  eq(node(compileSchema({ type: "object", additionalProperties: false })).free, null, "closed and empty");
  eq(node(compileSchema({ type: "object" })).free != null, true, "no properties: free keys, any value");
});
Deno.test("jsonschema: enums, const, type lists, nullable, anyOf merge into first-character unions", () => {
  eq(node(compileSchema({ enum: ["a", "b"] })), { k: "str", en: ["a", "b"] });
  eq(node(compileSchema({ const: 3 })), { k: "lit", texts: ["3"] });
  const u = node(compileSchema({ type: ["string", "null"] }));
  eq([u.k, u.str != null, u.null != null, u.num], ["u", true, true, null]);
  const n = node(compileSchema({ type: "integer", nullable: true }));
  eq([n.k, n.num != null, n.null != null], ["u", true, true]);
  const a = compileSchema({ anyOf: [{ type: "integer" }, { type: "number" }, { enum: ["x"] }, { type: "string" }] });
  const an = node(a);
  eq([a.nodes[an.num].k, a.nodes[an.str].en], ["num", null], "number covers integer; a free string covers the enum");
  const o = compileSchema({ oneOf: [{ type: "object", properties: { a: { type: "string" } }, required: ["a"] }, { type: "object", properties: { b: { type: "boolean" } }, required: ["b"] }] });
  const on = node(o);
  eq([on.k, on.props.map(([k]) => k), on.req], ["obj", ["a", "b"], []], "two object branches: keys united, required only what both require");
  eq(node(compileSchema({ anyOf: [{}, { type: "string" }] })).k, "any");
});
Deno.test("jsonschema: allOf merges objects; $ref / $defs / definitions resolve; recursion bottoms out in any", () => {
  const m = node(compileSchema({ allOf: [{ type: "object", properties: { a: { type: "string" } }, required: ["a"] }, { properties: { b: { type: "integer" } }, required: ["b"] }] }));
  eq([m.props.map(([k]) => k), m.req], [["a", "b"], [0, 1]]);
  const r = compileSchema({ type: "object", properties: { p: { $ref: "#/$defs/P" }, q: { $ref: "#/definitions/Q" } }, $defs: { P: { type: "integer" } }, definitions: { Q: { enum: [1, 2] } } });
  eq(node(r).props.map(([, id]) => r.nodes[id].k), ["int", "lit"]);
  const tree = compileSchema({ $defs: { T: { type: "object", properties: { kids: { type: "array", items: { $ref: "#/$defs/T" } } } } }, $ref: "#/$defs/T" });
  ok(tree.nodes.some((x) => x.k === "any"), "recursion deeper than 6 is any value");
  ok(tree.nodes.length < 200, "and stays small: " + tree.nodes.length);
  throwsSchema(() => compileSchema({ $ref: "https://example.com/s.json" }), /only local references/, "remote ref");
  throwsSchema(() => compileSchema({ $ref: "#/$defs/nope" }), /does not resolve/, "missing ref");
});
Deno.test("jsonschema: keywords that are accepted but not enforced change nothing", () => {
  const plain = compileSchema({ type: "object", properties: { s: { type: "string" }, n: { type: "number" } } });
  const bounded = compileSchema({ $schema: "x", title: "t", description: "d", type: "object", properties: { s: { type: "string", minLength: 2, maxLength: 5, pattern: "^a", format: "email", default: "aa" }, n: { type: "number", minimum: 0, maximum: 9, multipleOf: 3, exclusiveMinimum: 0 } }, examples: [] });
  eq(bounded.nodes, plain.nodes);
});
Deno.test("jsonschema: caps (a request's schema must not stall the room)", () => {
  throwsSchema(() => compileSchema({ enum: Array.from({ length: SCHEMA_CAPS.enum + 1 }, (_, i) => "v" + i) }), /enum has more than/, "enum");
  throwsSchema(() => compileSchema({ anyOf: Array.from({ length: SCHEMA_CAPS.anyOf + 1 }, () => ({ type: "string" })) }), /anyOf/, "anyOf");
  throwsSchema(() => compileSchema({ allOf: Array.from({ length: SCHEMA_CAPS.allOf + 1 }, () => ({ type: "object" })) }), /allOf/, "allOf");
  let deep = { type: "string" };
  for (let i = 0; i < SCHEMA_CAPS.depth + 2; i++) deep = { type: "array", items: deep };
  throwsSchema(() => compileSchema(deep), /deeper than/, "depth");
  const wide = { type: "object", properties: {} };
  for (let i = 0; i < 400; i++) wide.properties["p" + i] = { type: "object", properties: Object.fromEntries(Array.from({ length: 30 }, (_, j) => ["q" + j, { type: "array", items: { type: "object", properties: { x: { type: "integer" } } } }])) };
  throwsSchema(() => compileSchema(wide), /too large/, "nodes");
});
Deno.test("jsonschema: string-capable (XML values written raw) and the types for coercion", () => {
  eq(["string", { type: "string", enum: ["high"] }, { const: "x" }].map((s) => stringCapable(typeof s === "string" ? { type: s } : s)), ["only", "only", "only"]);
  eq([{ type: ["string", "null"] }, { anyOf: [{ type: "string" }, { type: "integer" }] }, {}, undefined].map((s) => stringCapable(s)), ["some", "some", "some", "some"]);
  eq([{ type: "integer" }, { type: "object", properties: {} }, { enum: [1, 2] }, { type: "array" }].map((s) => stringCapable(s)), ["none", "none", "none", "none"]);
  eq(stringCapable({ $ref: "#/$defs/S" }, { $defs: { S: { type: "string" } } }), "only");
  eq([...schemaTypes({ anyOf: [{ type: "integer" }, { type: "null" }] })], ["integer", "null"]);
  eq([...schemaTypes({ enum: ["a", 1, 1.5, null, true] })], ["string", "integer", "number", "null", "boolean"]);
  eq(schemaTypes({ description: "anything" }), null);
});
