const fields = { sent: "framesSent", recv: "framesReceived", dups: "duplicateSlices", skipped: "skippedFrames", kaSent: "keepaliveMessages" };

// These are existing application counters, not network byte or retransmit counts.
export function wireSnapshot(connections, nameOf) {
  const identities = new WeakMap(); let next = 0;
  return () => Object.fromEntries([...connections].map(([id, entry]) => {
    const link = entry.link;
    if (!link) return [nameOf(id), null];
    if (!identities.has(link)) identities.set(link, ++next);
    return [nameOf(id), { generation: identities.get(link),
      ...Object.fromEntries(Object.entries(fields).map(([source, name]) => [name, link[source]])),
      openChannels: link.chans.filter((c) => c.readyState === "open").length }];
  }));
}

export function wireDifference(before, after) {
  if (!before || !after) return { status: "unavailable", reason: "wire snapshots not supplied", links: null };
  const names = Object.keys(before).sort();
  if (JSON.stringify(names) !== JSON.stringify(Object.keys(after).sort())) {
    return { status: "incomplete", reason: "room links changed", links: null };
  }
  const links = {};
  for (const name of names) {
    const a = before[name], b = after[name];
    if (!a || !b || a.generation !== b.generation) return { status: "incomplete", reason: "wire link replaced or unavailable", links: null };
    const delta = {};
    for (const field of Object.values(fields)) {
      if (![a[field], b[field]].every((n) => Number.isSafeInteger(n) && n >= 0) || b[field] < a[field]) {
        return { status: "incomplete", reason: "wire counter reset or invalid", links: null };
      }
      delta[field] = b[field] - a[field];
    }
    links[name] = { ...delta, openChannelsBefore: a.openChannels, openChannelsAfter: b.openChannels };
  }
  return { status: "pass", reason: null, links, wireBytes: null,
    wireBytesReason: "existing counters count frames, duplicate slices and keepalive messages; exclude SCTP/DTLS/IP overhead" };
}
