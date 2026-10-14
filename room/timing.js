// Bounded local frame records; callers start each attempt before dispatching work.
// Collection does no I/O and can be used unchanged in browsers and room nodes.
export function createTimingCollector({ capacity = 16384 } = {}) {
  if (!Number.isSafeInteger(capacity) || capacity < 1) throw new RangeError("invalid timing capacity");
  let attempt = null, records = [], dropped = 0;
  return {
    begin(id) {
      if (attempt !== null) throw new Error("timing attempt already active");
      if (typeof id !== "string" || !id) throw new TypeError("timing attempt needs an id");
      attempt = id; records = []; dropped = 0;
    },
    record(event) {
      if (attempt === null) return;
      if (records.length < capacity) records.push({ ...event, sequence: records.length });
      else dropped++;
    },
    end() {
      if (attempt === null) throw new Error("no active timing attempt");
      const result = { attempt, records, dropped };
      attempt = null; records = [];
      return result;
    },
  };
}
